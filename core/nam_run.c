// neural-amx: forward pass. Each layer maps `rows` (1 or NAM_BATCH) activation rows of stride model->width;
// columns past a layer's output are kept at zero so the next dense kernel can read padded rows.
#include <math.h>
#include <string.h>

#include "nam_internal.h"
#include "nam_math.h"

static void dense_rows(nam_model *M, const nam_matrix *m, const float *x, int rows, float *y) {
  size_t W = (size_t)M->width;
  if (m->bits == 32) {
    if (rows == NAM_BATCH && M->batch) nam_dense_f32_batch(m, x, W, y, W);
    else for (int r = 0; r < rows; r++) nam_dense_f32(m, x + r * W, y + r * W);
    return;
  }
  float sx[NAM_BATCH];
  size_t qs = (size_t)M->qwidth;
  for (int r = 0; r < rows; r++) sx[r] = nam_quantize_row(x + r * W, m->cols, m->cols_pad, M->q + r * qs);
  if (rows == NAM_BATCH && M->batch) nam_dense_int_batch(m, M->q, qs, sx, y, W, M->acc16);
  else for (int r = 0; r < rows; r++) nam_dense_int(m, M->q + r * qs, sx[r], y + r * W, M->acc16);
}

// conv1d as a dense layer over patches; the output positions play the role of batch rows
static void conv1d_row(nam_model *M, const nam_layer *L, const float *x, float *y) {
  const nam_matrix *m = &L->m;
  size_t qs = (size_t)M->qwidth, cw = (size_t)M->cout_width;
  float sx = 0.0f;
  if (m->bits != 32) sx = nam_quantize_row(x, L->in, L->in, M->qx);
  for (int t = 0; t < L->len_out; t++) {
    for (int ci = 0; ci < L->in_ch; ci++)
      for (int j = 0; j < L->kernel; j++) {
        int idx = t * L->stride + j - L->pad, col = ci * L->kernel + j;
        int inside = idx >= 0 && idx < L->len_in;
        if (m->bits == 32) M->patch[t * qs + col] = inside ? x[ci * L->len_in + idx] : 0.0f;
        else M->q[t * qs + col] = inside ? M->qx[ci * L->len_in + idx] : 0;
      }
    if (m->bits == 32) memset(M->patch + t * qs + m->cols, 0, (size_t)(m->cols_pad - m->cols) * 4);
    else memset(M->q + t * qs + m->cols, 0, (size_t)(m->cols_pad - m->cols) * 2);
  }
  float sxs[NAM_BATCH] = {sx, sx, sx, sx};
  int t = 0;
  for (; t + NAM_BATCH <= L->len_out; t += NAM_BATCH) {
    if (m->bits == 32) nam_dense_f32_batch(m, M->patch + t * qs, qs, M->cout + t * cw, cw);
    else nam_dense_int_batch(m, M->q + t * qs, qs, sxs, M->cout + t * cw, cw, M->acc16);
  }
  for (; t < L->len_out; t++) {
    if (m->bits == 32) nam_dense_f32(m, M->patch + t * qs, M->cout + t * cw);
    else nam_dense_int(m, M->q + t * qs, sx, M->cout + t * cw, M->acc16);
  }
  for (int o = 0; o < L->out_ch; o++)
    for (int p = 0; p < L->len_out; p++) y[o * L->len_out + p] = M->cout[p * cw + o];
}

// PyTorch GRU cell: r, z, n gates; h' = (1 - z) * n + z * h
static void gru_row(nam_model *M, const nam_layer *L, const float *x, float *h, float *y) {
  int H = L->hidden;
  float *gi = M->tmp, *gh = M->tmp + 3 * H;
  if (L->m.bits == 32) nam_dense_f32(&L->m, x, gi);
  else nam_dense_int(&L->m, M->qx, nam_quantize_row(x, L->m.cols, L->m.cols_pad, M->qx), gi, M->acc16);
  if (L->h.bits == 32) nam_dense_f32(&L->h, h, gh);
  else nam_dense_int(&L->h, M->qx, nam_quantize_row(h, H, L->h.cols_pad, M->qx), gh, M->acc16);
  for (int j = 0; j < H; j++) {
    float r = nam_sigmoidf(gi[j] + gh[j]);
    float z = nam_sigmoidf(gi[H + j] + gh[H + j]);
    float n = nam_tanhf(gi[2 * H + j] + r * gh[2 * H + j]);
    float v = (1.0f - z) * n + z * h[j];
    y[j] = v;
    h[j] = v;
  }
}

static void layernorm_row(const nam_layer *L, const float *x, float *y) {
  float s = 0.0f;
  for (int k = 0; k < L->in; k++) s += x[k];
  float mean = s / (float)L->in, v = 0.0f;
  for (int k = 0; k < L->in; k++) { float d = x[k] - mean; v += d * d; }
  float inv = 1.0f / sqrtf(v / (float)L->in + L->alpha);
  for (int k = 0; k < L->in; k++) y[k] = ((x[k] - mean) * inv) * L->g[k] + L->b[k];
}

static void softmax_row(const float *x, int n, float *y) {
  float mx = x[0], s = 0.0f;
  for (int k = 1; k < n; k++) if (x[k] > mx) mx = x[k];
  for (int k = 0; k < n; k++) { y[k] = nam_expf(x[k] - mx); s += y[k]; }
  for (int k = 0; k < n; k++) y[k] = y[k] / s;
}

static void elementwise(const nam_layer *L, const float *x, float *y) {
  int n = L->in;
  switch (L->op) {
    case NAM_OP_RELU: for (int k = 0; k < n; k++) y[k] = x[k] > 0.0f ? x[k] : 0.0f; break;
    case NAM_OP_LEAKY_RELU: for (int k = 0; k < n; k++) y[k] = x[k] < 0.0f ? x[k] * L->alpha : x[k]; break;
    case NAM_OP_SIGMOID: for (int k = 0; k < n; k++) y[k] = nam_sigmoidf(x[k]); break;
    case NAM_OP_TANH: for (int k = 0; k < n; k++) y[k] = nam_tanhf(x[k]); break;
    case NAM_OP_GELU: for (int k = 0; k < n; k++) y[k] = nam_gelu_erf(x[k]); break;
    case NAM_OP_GELU_TANH: for (int k = 0; k < n; k++) y[k] = nam_gelu_tanh(x[k]); break;
    case NAM_OP_SILU: for (int k = 0; k < n; k++) y[k] = x[k] * nam_sigmoidf(x[k]); break;
    case NAM_OP_AFFINE: for (int k = 0; k < n; k++) y[k] = x[k] * L->g[k] + L->b[k]; break;
    case NAM_OP_LAYERNORM: layernorm_row(L, x, y); break;
    case NAM_OP_SOFTMAX: softmax_row(x, n, y); break;
    default: break;
  }
}

void nam_forward(nam_model *M, const float *in, int rows, float *out, int first_slot) {
  size_t W = (size_t)M->width;
  float *a = M->act[0], *b = M->act[1];
  for (int r = 0; r < rows; r++) {
    memcpy(a + r * W, in + (size_t)r * M->inputs, (size_t)M->inputs * 4);
    memset(a + r * W + M->inputs, 0, (W - (size_t)M->inputs) * 4);
  }
  for (int i = 0; i < M->n; i++) {
    const nam_layer *L = &M->layers[i];
    switch (L->op) {
      case NAM_OP_DENSE: dense_rows(M, &L->m, a, rows, b); break;
      case NAM_OP_CONV1D: for (int r = 0; r < rows; r++) conv1d_row(M, L, a + r * W, b + r * W); break;
      case NAM_OP_GRU:
        for (int r = 0; r < rows; r++)
          gru_row(M, L, a + r * W, M->state + (size_t)(first_slot + r) * M->state_size + L->state_offset, b + r * W);
        break;
      default: for (int r = 0; r < rows; r++) elementwise(L, a + r * W, b + r * W); break;
    }
    for (int r = 0; r < rows; r++) memset(b + r * W + L->out, 0, (W - (size_t)L->out) * 4);
    float *t = a; a = b; b = t;
  }
  for (int r = 0; r < rows; r++) memcpy(out + (size_t)r * M->outputs, a + r * W, (size_t)M->outputs * 4);
}

int nam_run(nam_model *M, const float *in, int count, float *out, int first_slot) {
  if (!M || count < 0 || first_slot < 0 || (M->state_size && first_slot + count > M->slots)) return -1;
  int i = 0;
  if (M->batch)
    for (; i + NAM_BATCH <= count; i += NAM_BATCH)
      nam_forward(M, in + (size_t)i * M->inputs, NAM_BATCH, out + (size_t)i * M->outputs, first_slot + i);
  for (; i < count; i++) nam_forward(M, in + (size_t)i * M->inputs, 1, out + (size_t)i * M->outputs, first_slot + i);
  return count;
}

void nam_reset(nam_model *M, int slot) {
  if (!M || !M->state_size) return;
  if (slot < 0) memset(M->state, 0, (size_t)M->slots * M->state_size * 4);
  else if (slot < M->slots) memset(M->state + (size_t)slot * M->state_size, 0, (size_t)M->state_size * 4);
}

int nam_argmax(const float *v, int n) {
  int best = 0;
  for (int k = 1; k < n; k++) if (v[k] > v[best]) best = k;
  return n > 0 ? best : -1;
}

void nam_softmax(float *v, int n, float temperature) {
  if (n <= 0) return;
  if (temperature > 0.0f && temperature != 1.0f) for (int k = 0; k < n; k++) v[k] = v[k] / temperature;
  softmax_row(v, n, v);
}

int nam_sample(const float *p, int n, float u) {
  float c = 0.0f;
  for (int k = 0; k < n; k++) { c += p[k]; if (u < c) return k; }
  return n - 1;
}
