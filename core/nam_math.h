// neural-amx: deterministic float32 math. Every function is a fixed sequence of IEEE float32 add/mul/div/floor/sqrt
// (no libm transcendental, no fused multiply-add), so results are identical on every CPU, compiler and wasm engine,
// and python/neural_amx/reference.py reproduces them bit for bit. Build with -ffp-contract=off (and -msse2
// -mfpmath=sse on i386, never x87).
#ifndef NAM_MATH_H
#define NAM_MATH_H

#include <math.h>
#include <stdint.h>
#include <string.h>

static inline float nam_bits_to_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// e^x, Cephes-style range reduction and degree-6 polynomial; ~1 ulp on the normal range
static inline float nam_expf(float x) {
  if (x > 88.0f) x = 88.0f;
  if (x < -87.0f) return 0.0f;
  float n = floorf(x * 1.44269504088896341f + 0.5f);
  x = x - n * 0.693359375f;
  x = x - n * -2.12194440e-4f;
  float z = x * x;
  float p = 1.9875691500e-4f;
  p = p * x + 1.3981999507e-3f;
  p = p * x + 8.3334519073e-3f;
  p = p * x + 4.1665795894e-2f;
  p = p * x + 1.6666665459e-1f;
  p = p * x + 5.0000001201e-1f;
  p = p * z + x + 1.0f;
  return p * nam_bits_to_float((uint32_t)((int32_t)n + 127) << 23);
}

static inline float nam_sigmoidf(float x) { return 1.0f / (1.0f + nam_expf(-x)); }

static inline float nam_tanhf(float x) {
  float a = x < 0.0f ? -x : x;
  if (a > 9.0f) return x < 0.0f ? -1.0f : 1.0f;
  if (a < 0.0625f) { // series: avoids the cancellation of (e - 1) near zero
    float x2 = x * x;
    return x + x * x2 * (-0.333333333f + x2 * 0.133333333f);
  }
  float e = nam_expf(2.0f * x);
  return (e - 1.0f) / (e + 1.0f);
}

// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
static inline float nam_erff(float x) {
  float a = x < 0.0f ? -x : x;
  float t = 1.0f / (1.0f + 0.3275911f * a);
  float y = 1.061405429f;
  y = y * t + -1.453152027f;
  y = y * t + 1.421413741f;
  y = y * t + -0.284496736f;
  y = y * t + 0.254829592f;
  y = 1.0f - y * t * nam_expf(-a * a);
  return x < 0.0f ? -y : y;
}

static inline float nam_gelu_erf(float x) { return 0.5f * x * (1.0f + nam_erff(x * 0.70710678118654752f)); }
static inline float nam_gelu_tanh(float x) {
  return 0.5f * x * (1.0f + nam_tanhf(0.79788456080286536f * (x + 0.044715f * x * x * x)));
}

#endif // NAM_MATH_H
