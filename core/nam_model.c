// neural-amx: model loading. The graph comes from the safetensors metadata key "neural_amx" (JSON) or from a
// one-line spec for plain state_dicts; weights are converted once into the kernel layout chosen by the precision.
#define _POSIX_C_SOURCE 200809L // clock_gettime
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

#include "nam_internal.h"
#include "nam_json.h"
#include "nam_safetensors.h"

#define MAX_LAYERS 256

typedef struct {
  char *err;
  size_t errlen;
} errbuf;

static int fail(errbuf *e, const char *fmt, ...) {
  if (e->err && e->errlen && !e->err[0]) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->err, e->errlen, fmt, ap);
    va_end(ap);
  }
  return 0;
}

static int pad16(int n) { return (n + 15) & ~15; }

static char *dup_string(const char *s) {
  size_t n = strlen(s) + 1;
  char *d = malloc(n);
  if (d) memcpy(d, s, n);
  return d;
}

static double now_ms(void) {
#if defined(_WIN32)
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
#endif
}

void nam_default_options(nam_options *o) {
  o->precision = NAM_PRECISION_STORED;
  o->kernel = NAM_KERNEL_AUTO;
  o->batch = -1;
  o->slots = 33;
}

// ---------- spec -> JSON graph ----------
typedef struct { char *s; size_t len, cap; } sbuf;

static void sb_add(sbuf *b, const char *fmt, ...) {
  va_list ap;
  for (;;) {
    va_start(ap, fmt);
    int n = vsnprintf(b->s ? b->s + b->len : NULL, b->s ? b->cap - b->len : 0, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (b->s && b->len + (size_t)n < b->cap) { b->len += (size_t)n; return; }
    size_t cap = (b->cap ? b->cap * 2 : 256) + (size_t)n;
    char *s = realloc(b->s, cap);
    if (!s) return;
    b->s = s;
    b->cap = cap;
  }
}

static int valid_name(const char *s) {
  if (!*s) return 0;
  for (; *s; s++)
    if (!((*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z') || (*s >= '0' && *s <= '9') || strchr("_.-/:", *s)))
      return 0;
  return 1;
}

// "input(64) dense(fc1) relu conv1d(c, stride=2, padding=1) gru(rnn) layernorm(ln, eps=1e-5) softmax"
static char *spec_to_json(const char *spec, errbuf *e) {
  sbuf b = {0};
  sb_add(&b, "{\"layers\":[");
  int first = 1, inputs = 0;
  const char *p = spec;
  while (*p) {
    while (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n') p++;
    if (!*p) break;
    char op[32] = {0}, args[8][64] = {{0}};
    int nargs = 0, k = 0;
    while (*p && *p != '(' && *p != ' ' && *p != ',' && k < 31) op[k++] = *p++;
    if (*p == '(') {
      p++;
      while (*p && *p != ')') {
        while (*p == ' ' || *p == ',') p++;
        k = 0;
        while (*p && *p != ',' && *p != ')' && k < 63) { if (*p != ' ') args[nargs][k++] = *p; p++; }
        if (k && nargs < 8) nargs++;
      }
      if (*p != ')') { free(b.s); fail(e, "spec: missing ')' after '%s'", op); return NULL; }
      p++;
    }
    const char *name = nargs ? args[0] : "";
    char opts[192] = "";
    for (int i = 1; i < nargs; i++) {
      char *eq = strchr(args[i], '=');
      if (!eq) continue;
      *eq = 0;
      size_t used = strlen(opts);
      snprintf(opts + used, sizeof opts - used, ",\"%s\":%s", args[i], eq + 1);
    }
    if (!strcmp(op, "input")) { inputs = atoi(name); continue; }
    int weighted = !strcmp(op, "dense") || !strcmp(op, "linear") || !strcmp(op, "conv1d") || !strcmp(op, "gru") ||
                   !strcmp(op, "layernorm") || !strcmp(op, "batchnorm");
    if (weighted && !valid_name(name)) { free(b.s); fail(e, "spec: '%s' needs a tensor prefix, e.g. %s(fc1)", op, op); return NULL; }
    if (!first) sb_add(&b, ",");
    first = 0;
    if (!strcmp(op, "dense") || !strcmp(op, "linear"))
      sb_add(&b, "{\"op\":\"dense\",\"weight\":\"%s.weight\",\"bias\":\"%s.bias\",\"bias_optional\":true}", name, name);
    else if (!strcmp(op, "conv1d"))
      sb_add(&b, "{\"op\":\"conv1d\",\"weight\":\"%s.weight\",\"bias\":\"%s.bias\",\"bias_optional\":true%s}", name, name, opts);
    else if (!strcmp(op, "gru"))
      sb_add(&b, "{\"op\":\"gru\",\"weight_ih\":\"%s.weight_ih_l0\",\"weight_hh\":\"%s.weight_hh_l0\","
                 "\"bias_ih\":\"%s.bias_ih_l0\",\"bias_hh\":\"%s.bias_hh_l0\",\"bias_optional\":true}", name, name, name, name);
    else if (!strcmp(op, "layernorm"))
      sb_add(&b, "{\"op\":\"layernorm\",\"weight\":\"%s.weight\",\"bias\":\"%s.bias\"%s}", name, name, opts);
    else if (!strcmp(op, "batchnorm"))
      sb_add(&b, "{\"op\":\"batchnorm\",\"weight\":\"%s.weight\",\"bias\":\"%s.bias\",\"mean\":\"%s.running_mean\","
                 "\"var\":\"%s.running_var\"%s}", name, name, name, name, opts);
    else if (!strcmp(op, "leaky_relu"))
      sb_add(&b, "{\"op\":\"leaky_relu\",\"alpha\":%s}", nargs ? name : "0.01");
    else
      sb_add(&b, "{\"op\":\"%s\"}", op);
  }
  sb_add(&b, "]");
  if (inputs > 0) sb_add(&b, ",\"inputs\":%d", inputs);
  sb_add(&b, "}");
  return b.s;
}

// ---------- weights ----------
static int is_float_dtype(nst_dtype d) { return d == NST_F32 || d == NST_F16 || d == NST_BF16 || d == NST_F64; }

static int8_t nibble(uint8_t b, int high) {
  int v = high ? b >> 4 : b & 0xF;
  return (int8_t)(v >= 8 ? v - 16 : v);
}

// Loads tensor `name` as rows x cols into `m` with `target_bits` (0 = as stored, 32, 8 or 4).
static int load_matrix(nam_matrix *m, const nst_file *f, int layer, const char *name, int rows, int cols,
                       int target_bits, errbuf *e) {
  const nst_tensor *t = nst_find(f, name);
  if (!t) return fail(e, "layer %d: tensor '%s' not found", layer, name);
  size_t cells = (size_t)rows * cols;
  float *w = malloc(cells * sizeof *w), *scale = NULL;
  int16_t *q = NULL;
  int stored_bits = 32;
  if (!w) return fail(e, "out of memory");
  if (is_float_dtype(t->dtype)) {
    if ((size_t)t->elements != cells) { free(w); return fail(e, "layer %d: '%s' has %lld values, expected %dx%d", layer, name, (long long)t->elements, rows, cols); }
    nst_to_f32(t, w);
  } else if (t->dtype == NST_I8 || t->dtype == NST_U8) {
    char sname[256];
    snprintf(sname, sizeof sname, "%s.scale", name);
    const nst_tensor *st = nst_find(f, sname);
    if (!st || st->elements != rows) { free(w); return fail(e, "layer %d: quantized '%s' needs '%s' with %d scales", layer, name, sname, rows); }
    stored_bits = t->dtype == NST_I8 ? 8 : 4;
    size_t row_bytes = stored_bits == 8 ? (size_t)cols : (size_t)(cols + 1) / 2;
    if ((size_t)t->elements != row_bytes * rows) { free(w); return fail(e, "layer %d: '%s' has the wrong size for %dx%d int%d", layer, name, rows, cols, stored_bits); }
    scale = malloc((size_t)rows * sizeof *scale);
    q = malloc(cells * sizeof *q);
    if (!scale || !q) { free(w); free(scale); free(q); return fail(e, "out of memory"); }
    nst_to_f32(st, scale);
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++) {
        const uint8_t *row = t->data + r * row_bytes;
        int v = stored_bits == 8 ? (int8_t)row[c] : nibble(row[c / 2], c & 1);
        q[(size_t)r * cols + c] = (int16_t)v;
        w[(size_t)r * cols + c] = (float)v * scale[r];
      }
  } else {
    free(w);
    return fail(e, "layer %d: '%s' has unsupported dtype %s", layer, name, nst_dtype_name(t->dtype));
  }
  int bits = target_bits ? target_bits : stored_bits;
  if (bits != 32 && !(q && stored_bits <= bits)) { // quantize per row: s = maxabs / qmax, q = floor(w / s + 0.5)
    float qmax = bits == 8 ? 127.0f : 7.0f;
    if (!scale) scale = malloc((size_t)rows * sizeof *scale);
    if (!q) q = malloc(cells * sizeof *q);
    if (!scale || !q) { free(w); free(scale); free(q); return fail(e, "out of memory"); }
    for (int r = 0; r < rows; r++) {
      float mx = 0.0f;
      for (int c = 0; c < cols; c++) { float a = fabsf(w[(size_t)r * cols + c]); if (a > mx) mx = a; }
      float s = mx / qmax;
      scale[r] = s;
      for (int c = 0; c < cols; c++) {
        float v = s > 0.0f ? floorf(w[(size_t)r * cols + c] / s + 0.5f) : 0.0f;
        q[(size_t)r * cols + c] = (int16_t)(v > qmax ? qmax : v < -qmax ? -qmax : v);
      }
    }
  }
  m->rows = rows;
  m->cols = cols;
  m->cols_pad = pad16(cols);
  m->bits = bits;
  size_t padded = (size_t)rows * m->cols_pad;
  if (bits == 32) {
    m->wf = calloc(padded, sizeof *m->wf);
    if (!m->wf) { free(w); free(scale); free(q); return fail(e, "out of memory"); }
    for (int r = 0; r < rows; r++) memcpy(m->wf + (size_t)r * m->cols_pad, w + (size_t)r * cols, (size_t)cols * 4);
    free(scale);
    scale = NULL;
  } else {
    m->wq = calloc(padded, sizeof *m->wq);
    if (!m->wq) { free(w); free(scale); free(q); return fail(e, "out of memory"); }
    int maxw = 0;
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++) {
        int v = q[(size_t)r * cols + c];
        m->wq[(size_t)r * m->cols_pad + c] = (int16_t)v;
        if (abs(v) > maxw) maxw = abs(v);
      }
    int fl = maxw ? 32767 / (maxw * 127) : 1 << 20;
    m->flush = fl >= 4 ? fl : 0; // measured: 2-3 steps (int8 weights) are slower than the i32 dot
    m->scale = scale;
  }
  free(w);
  free(q);
  m->bias = calloc((size_t)rows, sizeof *m->bias);
  return m->bias ? 1 : fail(e, "out of memory");
}

static int load_vector(float **dst, const nst_file *f, int layer, const char *name, int n, int optional, float fill, errbuf *e) {
  *dst = malloc((size_t)(n ? n : 1) * sizeof **dst);
  if (!*dst) return fail(e, "out of memory");
  const nst_tensor *t = name ? nst_find(f, name) : NULL;
  if (!t) {
    if (!optional && name) return fail(e, "layer %d: tensor '%s' not found", layer, name);
    for (int i = 0; i < n; i++) (*dst)[i] = fill;
    return 1;
  }
  if (t->elements != n || !is_float_dtype(t->dtype)) return fail(e, "layer %d: '%s' must be a float vector of %d", layer, name, n);
  return nst_to_f32(t, *dst);
}

static int load_bias(nam_matrix *m, const nst_file *f, int layer, const nj *spec, const char *key, errbuf *e) {
  const char *name = nj_str(spec, key, NULL);
  if (!name) return 1;
  const nst_tensor *t = nst_find(f, name);
  if (!t) return nj_num(spec, "bias_optional", 0) ? 1 : fail(e, "layer %d: tensor '%s' not found", layer, name);
  if (t->elements != m->rows) return fail(e, "layer %d: bias '%s' has %lld values, expected %d", layer, name, (long long)t->elements, m->rows);
  return nst_to_f32(t, m->bias);
}

static const struct { const char *name; nam_op op; } kOps[] = {
  {"dense", NAM_OP_DENSE}, {"linear", NAM_OP_DENSE}, {"conv1d", NAM_OP_CONV1D}, {"gru", NAM_OP_GRU},
  {"layernorm", NAM_OP_LAYERNORM}, {"affine", NAM_OP_AFFINE}, {"batchnorm", NAM_OP_AFFINE}, {"relu", NAM_OP_RELU},
  {"leaky_relu", NAM_OP_LEAKY_RELU}, {"sigmoid", NAM_OP_SIGMOID}, {"tanh", NAM_OP_TANH}, {"gelu", NAM_OP_GELU},
  {"gelu_tanh", NAM_OP_GELU_TANH}, {"silu", NAM_OP_SILU}, {"swish", NAM_OP_SILU}, {"softmax", NAM_OP_SOFTMAX},
  {"transpose", NAM_OP_TRANSPOSE},
};

static const char *op_name(nam_op op) {
  for (size_t i = 0; i < sizeof kOps / sizeof *kOps; i++) if (kOps[i].op == op) return kOps[i].name;
  return "?";
}

static int is_weighted(const char *op) { return !strcmp(op, "dense") || !strcmp(op, "linear") || !strcmp(op, "conv1d") || !strcmp(op, "gru"); }

// target bits of weighted layer `k` of `count` for a precision
static int target_bits(nam_precision p, int k, int count, int recurrent) {
  switch (p) {
    case NAM_PRECISION_F32: return 32;
    case NAM_PRECISION_INT8: return 8;
    case NAM_PRECISION_INT4: return (recurrent || k == 0 || k == count - 1) ? 8 : 4;
    default: return 0;
  }
}

static int build_layer(nam_layer *L, const nj *spec, const nst_file *f, int index, int in, int bits, errbuf *e) {
  const char *op = nj_str(spec, "op", "");
  size_t i;
  for (i = 0; i < sizeof kOps / sizeof *kOps; i++) if (!strcmp(op, kOps[i].name)) break;
  if (i == sizeof kOps / sizeof *kOps) return fail(e, "layer %d: unknown op '%s'", index, op);
  L->op = kOps[i].op;
  L->in = L->out = in;
  if (L->op == NAM_OP_DENSE) {
    const nst_tensor *t = nst_find(f, nj_str(spec, "weight", ""));
    if (!t) return fail(e, "layer %d: tensor '%s' not found", index, nj_str(spec, "weight", ""));
    int rows = (int)t->shape[0];
    if (rows <= 0) return fail(e, "layer %d: bad weight shape", index);
    if (!load_matrix(&L->m, f, index, nj_str(spec, "weight", ""), rows, in, bits, e)) return 0;
    if (!load_bias(&L->m, f, index, spec, "bias", e)) return 0;
    L->out = rows;
  } else if (L->op == NAM_OP_CONV1D) {
    const nst_tensor *t = nst_find(f, nj_str(spec, "weight", ""));
    if (!t) return fail(e, "layer %d: tensor '%s' not found", index, nj_str(spec, "weight", ""));
    L->out_ch = (int)t->shape[0];
    L->in_ch = (int)nj_num(spec, "in_channels", t->ndim == 3 ? (double)t->shape[1] : 0);
    L->kernel = (int)nj_num(spec, "kernel_size", t->ndim == 3 ? (double)t->shape[2] : 0);
    L->stride = (int)nj_num(spec, "stride", 1);
    L->pad = (int)nj_num(spec, "padding", 0);
    if (L->in_ch <= 0 || L->kernel <= 0 || L->stride <= 0 || L->pad < 0 || in % L->in_ch)
      return fail(e, "layer %d: conv1d expects input channels x length (input %d, in_channels %d)", index, in, L->in_ch);
    L->len_in = in / L->in_ch;
    L->len_out = (L->len_in + 2 * L->pad - L->kernel) / L->stride + 1;
    if (L->len_out <= 0) return fail(e, "layer %d: conv1d output length %d", index, L->len_out);
    if (!load_matrix(&L->m, f, index, nj_str(spec, "weight", ""), L->out_ch, L->in_ch * L->kernel, bits, e)) return 0;
    if (!load_bias(&L->m, f, index, spec, "bias", e)) return 0;
    L->out = L->out_ch * L->len_out;
  } else if (L->op == NAM_OP_GRU) {
    const nst_tensor *t = nst_find(f, nj_str(spec, "weight_hh", ""));
    if (!t || t->shape[0] % 3) return fail(e, "layer %d: gru needs weight_hh of shape [3H, H]", index);
    L->hidden = (int)t->shape[0] / 3;
    if (!load_matrix(&L->m, f, index, nj_str(spec, "weight_ih", ""), 3 * L->hidden, in, bits, e)) return 0;
    if (!load_matrix(&L->h, f, index, nj_str(spec, "weight_hh", ""), 3 * L->hidden, L->hidden, bits, e)) return 0;
    if (!load_bias(&L->m, f, index, spec, "bias_ih", e) || !load_bias(&L->h, f, index, spec, "bias_hh", e)) return 0;
    L->out = L->hidden;
  } else if (L->op == NAM_OP_LAYERNORM) {
    L->alpha = (float)nj_num(spec, "eps", 1e-5);
    if (!load_vector(&L->g, f, index, nj_str(spec, "weight", NULL), in, 1, 1.0f, e)) return 0;
    if (!load_vector(&L->b, f, index, nj_str(spec, "bias", NULL), in, 1, 0.0f, e)) return 0;
  } else if (L->op == NAM_OP_AFFINE && !strcmp(op, "batchnorm")) { // y = (x - mean) / sqrt(var + eps) * w + b
    float *mean, *var;
    float eps = (float)nj_num(spec, "eps", 1e-5);
    if (!load_vector(&L->g, f, index, nj_str(spec, "weight", NULL), in, 1, 1.0f, e) ||
        !load_vector(&L->b, f, index, nj_str(spec, "bias", NULL), in, 1, 0.0f, e) ||
        !load_vector(&mean, f, index, nj_str(spec, "mean", ""), in, 0, 0.0f, e) ||
        !load_vector(&var, f, index, nj_str(spec, "var", ""), in, 0, 1.0f, e))
      return 0;
    for (int k = 0; k < in; k++) {
      float s = L->g[k] / sqrtf(var[k] + eps);
      L->b[k] = L->b[k] - mean[k] * s;
      L->g[k] = s;
    }
    free(mean);
    free(var);
  } else if (L->op == NAM_OP_AFFINE) {
    if (!load_vector(&L->g, f, index, nj_str(spec, "scale", ""), in, 0, 1.0f, e)) return 0;
    if (!load_vector(&L->b, f, index, nj_str(spec, "shift", NULL), in, 1, 0.0f, e)) return 0;
  } else if (L->op == NAM_OP_LEAKY_RELU) {
    L->alpha = (float)nj_num(spec, "alpha", 0.01);
  } else if (L->op == NAM_OP_TRANSPOSE) { // channels-last <-> channels-first for conv models from Keras/MLX
    L->trows = (int)nj_num(spec, "rows", 0);
    L->tcols = (int)nj_num(spec, "cols", 0);
    if (L->trows <= 0 || L->tcols <= 0 || L->trows * L->tcols != in)
      return fail(e, "layer %d: transpose %dx%d does not match input %d", index, L->trows, L->tcols, in);
  }
  return 1;
}

static void free_matrix(nam_matrix *m) { free(m->wf); free(m->wq); free(m->scale); free(m->bias); }

void nam_free(nam_model *M) {
  if (!M) return;
  for (int i = 0; i < M->n; i++) {
    free_matrix(&M->layers[i].m);
    free_matrix(&M->layers[i].h);
    free(M->layers[i].g);
    free(M->layers[i].b);
  }
  for (int i = 0; i < M->nlabels; i++) free(M->labels[i]);
  free(M->labels);
  free(M->layers);
  free(M->state);
  free(M->act[0]);
  free(M->act[1]);
  free(M->q);
  free(M->qx);
  free(M->patch);
  free(M->cout);
  free(M->tmp);
  free(M);
}

static size_t matrix_bytes(const nam_matrix *m) {
  return (size_t)m->rows * m->cols_pad * (m->bits == 32 ? 4 : 2) + (m->scale ? (size_t)m->rows * 4 : 0) + (size_t)m->rows * 4;
}

static int allocate(nam_model *M, errbuf *e) {
  int width = pad16(M->inputs), qwidth = 0, patch_rows = NAM_BATCH, cout_width = 0, hidden = 0;
  M->state_size = 0;
  for (int i = 0; i < M->n; i++) {
    nam_layer *L = &M->layers[i];
    if (pad16(L->out) > width) width = pad16(L->out);
    if (L->m.cols_pad > qwidth) qwidth = L->m.cols_pad;
    if (L->h.cols_pad > qwidth) qwidth = L->h.cols_pad;
    if (L->op == NAM_OP_CONV1D) {
      if (L->len_out > patch_rows) patch_rows = L->len_out;
      if (L->out_ch > cout_width) cout_width = L->out_ch;
    }
    if (L->op == NAM_OP_GRU) {
      L->state_offset = M->state_size;
      M->state_size += pad16(L->hidden);
      if (L->hidden > hidden) hidden = L->hidden;
    }
    M->weight_bytes += (L->m.rows ? matrix_bytes(&L->m) : 0) + (L->h.rows ? matrix_bytes(&L->h) : 0);
  }
  if (width > qwidth) qwidth = width;
  M->width = width;
  M->qwidth = qwidth;
  M->patch_rows = patch_rows;
  M->cout_width = cout_width;
  M->act[0] = calloc((size_t)NAM_BATCH * width, 4);
  M->act[1] = calloc((size_t)NAM_BATCH * width, 4);
  M->q = calloc((size_t)patch_rows * qwidth, 2);
  M->qx = calloc((size_t)qwidth, 2);
  M->patch = cout_width ? calloc((size_t)patch_rows * qwidth, 4) : NULL;
  M->cout = cout_width ? calloc((size_t)patch_rows * cout_width, 4) : NULL;
  M->tmp = calloc((size_t)6 * (hidden ? hidden : 1), 4);
  M->state = M->state_size ? calloc((size_t)M->slots * M->state_size, 4) : NULL;
  if (!M->act[0] || !M->act[1] || !M->q || !M->qx || !M->tmp || (cout_width && (!M->patch || !M->cout)) ||
      (M->state_size && !M->state))
    return fail(e, "out of memory");
  return 1;
}

static double time_config(nam_model *M, int acc16, int batch, const float *x, float *y, int reps) {
  M->acc16 = acc16;
  M->batch = batch;
  double best = 1e30;
  for (int t = 0; t < 3; t++) {
    double t0 = now_ms();
    for (int r = 0; r < reps; r++) nam_run(M, x, 2 * NAM_BATCH, y, 0);
    double dt = now_ms() - t0;
    if (dt < best) best = dt;
  }
  return best;
}

int nam_tune(nam_model *M) {
  int has_acc16 = 0;
  for (int i = 0; i < M->n; i++) has_acc16 |= M->layers[i].m.flush || M->layers[i].h.flush;
  float *x = calloc((size_t)2 * NAM_BATCH * M->inputs, 4), *y = malloc((size_t)2 * NAM_BATCH * M->outputs * 4);
  if (!x || !y) { free(x); free(y); return 0; }
  int slots = M->slots;
  if (M->state_size && slots < 2 * NAM_BATCH) { free(x); free(y); return 0; }
  double t0 = now_ms();
  nam_run(M, x, 2 * NAM_BATCH, y, 0);
  double one = now_ms() - t0;
  int reps = one > 0.0 ? (int)(2.0 / one) + 1 : 64; // ~2 ms per timing
  double best = 1e30;
  int best_acc16 = 0, best_batch = 1;
  for (int batch = 0; batch <= 1; batch++) {
    if (!M->auto_batch && batch != M->batch) continue;
    for (int acc16 = 0; acc16 <= has_acc16; acc16++) {
      double t = time_config(M, acc16, batch, x, y, reps);
      if (t < best) { best = t; best_acc16 = acc16; best_batch = batch; }
    }
  }
  M->acc16 = best_acc16;
  M->batch = best_batch;
  nam_reset(M, -1);
  free(x);
  free(y);
  return 1;
}

void nam_set_kernel(nam_model *M, int acc16, int batch) {
  M->acc16 = acc16;
  M->batch = batch;
}

nam_model *nam_load(const void *data, size_t size, const char *spec, const nam_options *options, char *err, size_t errlen) {
  errbuf e = {err, errlen};
  if (err && errlen) err[0] = 0;
  nam_options opt;
  if (options) opt = *options;
  else nam_default_options(&opt);
  if (opt.slots < 1) opt.slots = 1;
  nst_file f;
  if (!nst_open(&f, data, size, err, errlen)) return NULL;
  char *json = NULL;
  if (spec && *spec) {
    json = spec_to_json(spec, &e);
  } else {
    const char *meta = nst_metadata(&f, "neural_amx");
    if (!meta) { nst_close(&f); fail(&e, "no 'neural_amx' graph in the file metadata: pass a spec, e.g. \"dense(fc1) relu dense(fc2)\""); return NULL; }
    json = dup_string(meta);
  }
  nj *graph = json ? nj_parse(json, strlen(json), err, errlen) : NULL;
  free(json);
  nj *layers = graph ? nj_get(graph, "layers") : NULL;
  nam_model *M = calloc(1, sizeof *M);
  if (!graph || !layers || layers->type != NJ_ARR || !M) {
    if (graph && (!layers || layers->type != NJ_ARR)) fail(&e, "graph: 'layers' must be an array");
    nj_free(graph); nst_close(&f); free(M);
    return NULL;
  }
  M->precision = opt.precision;
  M->slots = opt.slots;
  M->layers = calloc((size_t)(layers->count ? layers->count : 1), sizeof *M->layers);
  // inputs: explicit, or inferred from the first weighted layer
  int in = (int)nj_num(graph, "inputs", 0), weighted = 0;
  for (nj *l = layers->child; l; l = l->next) weighted += is_weighted(nj_str(l, "op", ""));
  if (in <= 0 && layers->child) {
    const nj *l0 = layers->child;
    const char *op = nj_str(l0, "op", "");
    const nst_tensor *t = nst_find(&f, nj_str(l0, !strcmp(op, "gru") ? "weight_ih" : "weight", ""));
    if (t && t->ndim == 2 && (!strcmp(op, "dense") || !strcmp(op, "linear") || !strcmp(op, "gru"))) in = (int)t->shape[1];
  }
  int ok = M->layers != NULL;
  if (ok && layers->count > MAX_LAYERS) ok = fail(&e, "graph: more than %d layers", MAX_LAYERS);
  if (ok && in <= 0) ok = fail(&e, "graph: cannot infer the input size; add \"inputs\" or input(N) to the spec");
  M->inputs = in;
  int k = 0;
  for (nj *l = layers->child; ok && l; l = l->next) {
    const char *op = nj_str(l, "op", "");
    if (!strcmp(op, "flatten") || !strcmp(op, "identity") || !strcmp(op, "dropout")) continue;
    int bits = 0;
    if (is_weighted(op)) {
      int stored = (int)nj_num(l, "bits", 0);
      bits = target_bits(opt.precision, k++, weighted, !strcmp(op, "gru"));
      if (opt.precision == NAM_PRECISION_STORED && stored) bits = stored;
    }
    ok = build_layer(&M->layers[M->n], l, &f, M->n, in, bits, &e);
    if (ok) in = M->layers[M->n++].out;
  }
  if (ok && M->n == 0) ok = fail(&e, "graph: no layers");
  M->outputs = in;
  nj *labels = nj_get(graph, "labels");
  if (ok && labels && labels->type == NJ_ARR) {
    M->labels = calloc((size_t)labels->count + 1, sizeof *M->labels);
    for (nj *s = labels->child; s && M->labels; s = s->next)
      if (s->type == NJ_STR) M->labels[M->nlabels++] = dup_string(s->str);
  }
  nj_free(graph);
  nst_close(&f);
  if (ok) ok = allocate(M, &e);
  if (!ok) { nam_free(M); return NULL; }
  M->auto_batch = opt.batch < 0;
  M->batch = opt.batch != 0;
  M->acc16 = opt.kernel == NAM_KERNEL_ACC16;
  if (opt.kernel == NAM_KERNEL_AUTO) nam_tune(M);
  return M;
}

int nam_inputs(const nam_model *M) { return M ? M->inputs : 0; }
int nam_outputs(const nam_model *M) { return M ? M->outputs : 0; }
size_t nam_weight_bytes(const nam_model *M) { return M ? M->weight_bytes : 0; }
const char *nam_label(const nam_model *M, int i) { return M && i >= 0 && i < M->nlabels ? M->labels[i] : NULL; }

int nam_describe(const nam_model *M, char *buf, size_t len) {
  sbuf b = {0};
  sb_add(&b, "neural-amx %s [%s] %d -> %d:", NAM_VERSION, nam_backend(), M->inputs, M->outputs);
  for (int i = 0; i < M->n; i++) {
    const nam_layer *L = &M->layers[i];
    sb_add(&b, " %s", op_name(L->op));
    if (L->m.rows) sb_add(&b, "(%d->%d %s)", L->in, L->out, L->m.bits == 32 ? "f32" : L->m.bits == 8 ? "int8" : "int4");
  }
  sb_add(&b, " | kernel %s, batch %s, weights %zu bytes", M->acc16 ? "acc16" : "dot", M->batch ? "on" : "off", M->weight_bytes);
  int n = b.s ? snprintf(buf, len, "%s", b.s) : 0;
  free(b.s);
  return n;
}
