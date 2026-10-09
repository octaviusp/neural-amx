// neural-amx: dense kernels. Specification (python/neural_amx/reference.py mirrors it):
//   float: per (row, sample), lanes a += w[k..k+3]*x[k..k+3], b += w[k+4..k+7]*x[k+4..k+7] for k = 0, 8, ...
//          over cols_pad; t = a + b; dot = (t0 + t1) + (t2 + t3); y = dot + bias. Blocking never changes this order.
//   int:   dynamic per-sample activation scale (nam_quantize_row), exact int32 dot of int16 lanes,
//          y = (float)acc * (sx * scale[row]) + bias[row].
#include <math.h>
#include <string.h>

#include "nam_internal.h"
#include "nam_simd.h"

// ---------- float ----------
static inline float dot1_f32(const float *w, const float *x, int n) {
  nam_f4 a = nam_f4_zero(), b = nam_f4_zero();
  for (int k = 0; k < n; k += 8) {
    a = nam_f4_add(a, nam_f4_mul(nam_f4_load(w + k), nam_f4_load(x + k)));
    b = nam_f4_add(b, nam_f4_mul(nam_f4_load(w + k + 4), nam_f4_load(x + k + 4)));
  }
  return nam_f4_sum(nam_f4_add(a, b));
}

// four rows share each activation load
static inline void dot4_f32(const float *w, size_t stride, const float *x, int n, float *r) {
  nam_f4 a0 = nam_f4_zero(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
  const float *w0 = w, *w1 = w + stride, *w2 = w + 2 * stride, *w3 = w + 3 * stride;
  for (int k = 0; k < n; k += 8) {
    nam_f4 x0 = nam_f4_load(x + k), x1 = nam_f4_load(x + k + 4);
    a0 = nam_f4_add(a0, nam_f4_mul(nam_f4_load(w0 + k), x0));
    b0 = nam_f4_add(b0, nam_f4_mul(nam_f4_load(w0 + k + 4), x1));
    a1 = nam_f4_add(a1, nam_f4_mul(nam_f4_load(w1 + k), x0));
    b1 = nam_f4_add(b1, nam_f4_mul(nam_f4_load(w1 + k + 4), x1));
    a2 = nam_f4_add(a2, nam_f4_mul(nam_f4_load(w2 + k), x0));
    b2 = nam_f4_add(b2, nam_f4_mul(nam_f4_load(w2 + k + 4), x1));
    a3 = nam_f4_add(a3, nam_f4_mul(nam_f4_load(w3 + k), x0));
    b3 = nam_f4_add(b3, nam_f4_mul(nam_f4_load(w3 + k + 4), x1));
  }
  r[0] = nam_f4_sum(nam_f4_add(a0, b0));
  r[1] = nam_f4_sum(nam_f4_add(a1, b1));
  r[2] = nam_f4_sum(nam_f4_add(a2, b2));
  r[3] = nam_f4_sum(nam_f4_add(a3, b3));
}

void nam_dense_f32(const nam_matrix *m, const float *x, float *y) {
  int o = 0, n = m->cols_pad;
  for (; o + 4 <= m->rows; o += 4) {
    float r[4];
    dot4_f32(m->wf + (size_t)o * n, (size_t)n, x, n, r);
    for (int j = 0; j < 4; j++) y[o + j] = r[j] + m->bias[o + j];
  }
  for (; o < m->rows; o++) y[o] = dot1_f32(m->wf + (size_t)o * n, x, n) + m->bias[o];
}

// one weight row feeds NAM_BATCH samples
void nam_dense_f32_batch(const nam_matrix *m, const float *x, size_t xs, float *y, size_t ys) {
  int n = m->cols_pad;
  const float *x0 = x, *x1 = x + xs, *x2 = x + 2 * xs, *x3 = x + 3 * xs;
  for (int o = 0; o < m->rows; o++) {
    const float *w = m->wf + (size_t)o * n;
    nam_f4 a0 = nam_f4_zero(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    for (int k = 0; k < n; k += 8) {
      nam_f4 w0 = nam_f4_load(w + k), w1 = nam_f4_load(w + k + 4);
      a0 = nam_f4_add(a0, nam_f4_mul(w0, nam_f4_load(x0 + k)));
      b0 = nam_f4_add(b0, nam_f4_mul(w1, nam_f4_load(x0 + k + 4)));
      a1 = nam_f4_add(a1, nam_f4_mul(w0, nam_f4_load(x1 + k)));
      b1 = nam_f4_add(b1, nam_f4_mul(w1, nam_f4_load(x1 + k + 4)));
      a2 = nam_f4_add(a2, nam_f4_mul(w0, nam_f4_load(x2 + k)));
      b2 = nam_f4_add(b2, nam_f4_mul(w1, nam_f4_load(x2 + k + 4)));
      a3 = nam_f4_add(a3, nam_f4_mul(w0, nam_f4_load(x3 + k)));
      b3 = nam_f4_add(b3, nam_f4_mul(w1, nam_f4_load(x3 + k + 4)));
    }
    float bias = m->bias[o];
    y[o] = nam_f4_sum(nam_f4_add(a0, b0)) + bias;
    y[ys + o] = nam_f4_sum(nam_f4_add(a1, b1)) + bias;
    y[2 * ys + o] = nam_f4_sum(nam_f4_add(a2, b2)) + bias;
    y[3 * ys + o] = nam_f4_sum(nam_f4_add(a3, b3)) + bias;
  }
}

// ---------- integer ----------
// symmetric per-row int8 activations: q = clamp(floor(x * (127 / maxabs) + 0.5)); returns the scale maxabs / 127
float nam_quantize_row(const float *x, int n, int npad, int16_t *q) {
  float maxabs = 0.0f;
  for (int k = 0; k < n; k++) {
    float a = x[k] < 0.0f ? -x[k] : x[k];
    if (a > maxabs) maxabs = a;
  }
  if (!(maxabs > 0.0f) || maxabs > 3.4e38f) { // zero, NaN-only or infinite rows quantize to zero
    memset(q, 0, (size_t)npad * sizeof *q);
    return 0.0f;
  }
  float inv = 127.0f / maxabs;
  for (int k = 0; k < n; k++) {
    float t = floorf(x[k] * inv + 0.5f);
    if (t != t) t = 0.0f;
    q[k] = (int16_t)(t > 127.0f ? 127.0f : t < -127.0f ? -127.0f : t);
  }
  memset(q + n, 0, (size_t)(npad - n) * sizeof *q);
  return maxabs / 127.0f;
}

static inline int32_t dot1_i16(const int16_t *w, const int16_t *x, int n) {
  nam_i4 a = nam_i4_zero(), b = nam_i4_zero();
  for (int k = 0; k < n; k += 16) {
    a = nam_i4_add(a, nam_i4_dot(nam_s8_load(w + k), nam_s8_load(x + k)));
    b = nam_i4_add(b, nam_i4_dot(nam_s8_load(w + k + 8), nam_s8_load(x + k + 8)));
  }
  return nam_i4_sum(nam_i4_add(a, b));
}

static inline void dot4_i16(const int16_t *w, size_t stride, const int16_t *x, int n, int32_t *r) {
  nam_i4 a0 = nam_i4_zero(), a1 = a0, a2 = a0, a3 = a0;
  const int16_t *w0 = w, *w1 = w + stride, *w2 = w + 2 * stride, *w3 = w + 3 * stride;
  for (int k = 0; k < n; k += 8) {
    nam_s8 xv = nam_s8_load(x + k);
    a0 = nam_i4_add(a0, nam_i4_dot(nam_s8_load(w0 + k), xv));
    a1 = nam_i4_add(a1, nam_i4_dot(nam_s8_load(w1 + k), xv));
    a2 = nam_i4_add(a2, nam_i4_dot(nam_s8_load(w2 + k), xv));
    a3 = nam_i4_add(a3, nam_i4_dot(nam_s8_load(w3 + k), xv));
  }
  r[0] = nam_i4_sum(a0); r[1] = nam_i4_sum(a1); r[2] = nam_i4_sum(a2); r[3] = nam_i4_sum(a3);
}

// products accumulate in int16 lanes for `flush` steps, then widen: exact when flush * max|w| * 127 <= 32767
static inline void dot4_acc16(const int16_t *w, size_t stride, const int16_t *x, int n, int flush, int32_t *r) {
  nam_i4 a0 = nam_i4_zero(), a1 = a0, a2 = a0, a3 = a0;
  const int16_t *w0 = w, *w1 = w + stride, *w2 = w + 2 * stride, *w3 = w + 3 * stride;
  for (int k0 = 0; k0 < n; k0 += 8 * flush) {
    int end = k0 + 8 * flush < n ? k0 + 8 * flush : n;
    nam_s8 s0 = nam_s8_zero(), s1 = s0, s2 = s0, s3 = s0;
    for (int k = k0; k < end; k += 8) {
      nam_s8 xv = nam_s8_load(x + k);
      s0 = nam_s8_add(s0, nam_s8_mul(nam_s8_load(w0 + k), xv));
      s1 = nam_s8_add(s1, nam_s8_mul(nam_s8_load(w1 + k), xv));
      s2 = nam_s8_add(s2, nam_s8_mul(nam_s8_load(w2 + k), xv));
      s3 = nam_s8_add(s3, nam_s8_mul(nam_s8_load(w3 + k), xv));
    }
    a0 = nam_i4_add(a0, nam_i4_widen_pairs(s0));
    a1 = nam_i4_add(a1, nam_i4_widen_pairs(s1));
    a2 = nam_i4_add(a2, nam_i4_widen_pairs(s2));
    a3 = nam_i4_add(a3, nam_i4_widen_pairs(s3));
  }
  r[0] = nam_i4_sum(a0); r[1] = nam_i4_sum(a1); r[2] = nam_i4_sum(a2); r[3] = nam_i4_sum(a3);
}

static inline float dequant(int32_t acc, float sx, float scale, float bias) { return (float)acc * (sx * scale) + bias; }

void nam_dense_int(const nam_matrix *m, const int16_t *q, float sx, float *y, int acc16) {
  int o = 0, n = m->cols_pad;
  int flush = acc16 ? m->flush : 0;
  for (; o + 4 <= m->rows; o += 4) {
    int32_t r[4];
    if (flush) dot4_acc16(m->wq + (size_t)o * n, (size_t)n, q, n, flush, r);
    else dot4_i16(m->wq + (size_t)o * n, (size_t)n, q, n, r);
    for (int j = 0; j < 4; j++) y[o + j] = dequant(r[j], sx, m->scale[o + j], m->bias[o + j]);
  }
  for (; o < m->rows; o++) y[o] = dequant(dot1_i16(m->wq + (size_t)o * n, q, n), sx, m->scale[o], m->bias[o]);
}

void nam_dense_int_batch(const nam_matrix *m, const int16_t *q, size_t qs, const float *sx, float *y, size_t ys,
                         int acc16) {
  int n = m->cols_pad, flush = acc16 ? m->flush : 0;
  const int16_t *q0 = q, *q1 = q + qs, *q2 = q + 2 * qs, *q3 = q + 3 * qs;
  for (int o = 0; o < m->rows; o++) {
    const int16_t *w = m->wq + (size_t)o * n;
    nam_i4 a0 = nam_i4_zero(), a1 = a0, a2 = a0, a3 = a0;
    if (flush) {
      for (int k0 = 0; k0 < n; k0 += 8 * flush) {
        int end = k0 + 8 * flush < n ? k0 + 8 * flush : n;
        nam_s8 s0 = nam_s8_zero(), s1 = s0, s2 = s0, s3 = s0;
        for (int k = k0; k < end; k += 8) {
          nam_s8 wv = nam_s8_load(w + k);
          s0 = nam_s8_add(s0, nam_s8_mul(wv, nam_s8_load(q0 + k)));
          s1 = nam_s8_add(s1, nam_s8_mul(wv, nam_s8_load(q1 + k)));
          s2 = nam_s8_add(s2, nam_s8_mul(wv, nam_s8_load(q2 + k)));
          s3 = nam_s8_add(s3, nam_s8_mul(wv, nam_s8_load(q3 + k)));
        }
        a0 = nam_i4_add(a0, nam_i4_widen_pairs(s0));
        a1 = nam_i4_add(a1, nam_i4_widen_pairs(s1));
        a2 = nam_i4_add(a2, nam_i4_widen_pairs(s2));
        a3 = nam_i4_add(a3, nam_i4_widen_pairs(s3));
      }
    } else {
      for (int k = 0; k < n; k += 8) {
        nam_s8 wv = nam_s8_load(w + k);
        a0 = nam_i4_add(a0, nam_i4_dot(wv, nam_s8_load(q0 + k)));
        a1 = nam_i4_add(a1, nam_i4_dot(wv, nam_s8_load(q1 + k)));
        a2 = nam_i4_add(a2, nam_i4_dot(wv, nam_s8_load(q2 + k)));
        a3 = nam_i4_add(a3, nam_i4_dot(wv, nam_s8_load(q3 + k)));
      }
    }
    float scale = m->scale[o], bias = m->bias[o];
    y[o] = dequant(nam_i4_sum(a0), sx[0], scale, bias);
    y[ys + o] = dequant(nam_i4_sum(a1), sx[1], scale, bias);
    y[2 * ys + o] = dequant(nam_i4_sum(a2), sx[2], scale, bias);
    y[3 * ys + o] = dequant(nam_i4_sum(a3), sx[3], scale, bias);
  }
}

const char *nam_backend(void) { return NAM_SIMD_NAME; }
