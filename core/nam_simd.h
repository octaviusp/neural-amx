// neural-amx: 128-bit SIMD abstraction with four backends that produce identical bits:
// WebAssembly SIMD128, x86 SSE2, ARM NEON and portable scalar C (lane-exact emulation).
// Only operations with exact, fully specified results are used: f32 add/mul (never fused), wrapping i16 add/mul,
// i16 pair dot products into i32, and i16 pairwise widening adds. Build with -ffp-contract=off.
#ifndef NAM_SIMD_H
#define NAM_SIMD_H

#include <stdint.h>
#include <string.h>

#if defined(NAM_NO_SIMD)
#define NAM_SIMD_SCALAR 1
#elif defined(__wasm_simd128__)
#define NAM_SIMD_WASM 1
#include <wasm_simd128.h>
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define NAM_SIMD_SSE2 1
#include <emmintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#define NAM_SIMD_NEON 1
#include <arm_neon.h>
#else
#define NAM_SIMD_SCALAR 1
#endif

#if NAM_SIMD_WASM
#define NAM_SIMD_NAME "wasm-simd128"
typedef v128_t nam_f4;
typedef v128_t nam_s8; // 8 x i16
typedef v128_t nam_i4;    // 4 x i32
static inline nam_f4 nam_f4_zero(void) { return wasm_f32x4_splat(0.0f); }
static inline nam_f4 nam_f4_load(const float *p) { return wasm_v128_load(p); }
static inline nam_f4 nam_f4_add(nam_f4 a, nam_f4 b) { return wasm_f32x4_add(a, b); }
static inline nam_f4 nam_f4_mul(nam_f4 a, nam_f4 b) { return wasm_f32x4_mul(a, b); }
static inline void nam_f4_store(float *p, nam_f4 a) { wasm_v128_store(p, a); }
static inline nam_s8 nam_s8_zero(void) { return wasm_i16x8_splat(0); }
static inline nam_s8 nam_s8_load(const int16_t *p) { return wasm_v128_load(p); }
static inline nam_s8 nam_s8_add(nam_s8 a, nam_s8 b) { return wasm_i16x8_add(a, b); }
static inline nam_s8 nam_s8_mul(nam_s8 a, nam_s8 b) { return wasm_i16x8_mul(a, b); }
static inline nam_i4 nam_i4_zero(void) { return wasm_i32x4_splat(0); }
static inline nam_i4 nam_i4_add(nam_i4 a, nam_i4 b) { return wasm_i32x4_add(a, b); }
static inline nam_i4 nam_i4_dot(nam_s8 a, nam_s8 b) { return wasm_i32x4_dot_i16x8(a, b); }
static inline nam_i4 nam_i4_widen_pairs(nam_s8 a) { return wasm_i32x4_extadd_pairwise_i16x8(a); }
static inline void nam_i4_store(int32_t *p, nam_i4 a) { wasm_v128_store(p, a); }
#elif NAM_SIMD_SSE2
#define NAM_SIMD_NAME "sse2"
typedef __m128 nam_f4;
typedef __m128i nam_s8;
typedef __m128i nam_i4;
static inline nam_f4 nam_f4_zero(void) { return _mm_setzero_ps(); }
static inline nam_f4 nam_f4_load(const float *p) { return _mm_loadu_ps(p); }
static inline nam_f4 nam_f4_add(nam_f4 a, nam_f4 b) { return _mm_add_ps(a, b); }
static inline nam_f4 nam_f4_mul(nam_f4 a, nam_f4 b) { return _mm_mul_ps(a, b); }
static inline void nam_f4_store(float *p, nam_f4 a) { _mm_storeu_ps(p, a); }
static inline nam_s8 nam_s8_zero(void) { return _mm_setzero_si128(); }
static inline nam_s8 nam_s8_load(const int16_t *p) { return _mm_loadu_si128((const __m128i *)p); }
static inline nam_s8 nam_s8_add(nam_s8 a, nam_s8 b) { return _mm_add_epi16(a, b); }
static inline nam_s8 nam_s8_mul(nam_s8 a, nam_s8 b) { return _mm_mullo_epi16(a, b); }
static inline nam_i4 nam_i4_zero(void) { return _mm_setzero_si128(); }
static inline nam_i4 nam_i4_add(nam_i4 a, nam_i4 b) { return _mm_add_epi32(a, b); }
static inline nam_i4 nam_i4_dot(nam_s8 a, nam_s8 b) { return _mm_madd_epi16(a, b); }
static inline nam_i4 nam_i4_widen_pairs(nam_s8 a) { return _mm_madd_epi16(a, _mm_set1_epi16(1)); }
static inline void nam_i4_store(int32_t *p, nam_i4 a) { _mm_storeu_si128((__m128i *)p, a); }
#elif NAM_SIMD_NEON
#define NAM_SIMD_NAME "neon"
typedef float32x4_t nam_f4;
typedef int16x8_t nam_s8;
typedef int32x4_t nam_i4;
static inline nam_f4 nam_f4_zero(void) { return vdupq_n_f32(0.0f); }
static inline nam_f4 nam_f4_load(const float *p) { return vld1q_f32(p); }
static inline nam_f4 nam_f4_add(nam_f4 a, nam_f4 b) { return vaddq_f32(a, b); }
static inline nam_f4 nam_f4_mul(nam_f4 a, nam_f4 b) { return vmulq_f32(a, b); }
static inline void nam_f4_store(float *p, nam_f4 a) { vst1q_f32(p, a); }
static inline nam_s8 nam_s8_zero(void) { return vdupq_n_s16(0); }
static inline nam_s8 nam_s8_load(const int16_t *p) { return vld1q_s16(p); }
static inline nam_s8 nam_s8_add(nam_s8 a, nam_s8 b) { return vaddq_s16(a, b); }
static inline nam_s8 nam_s8_mul(nam_s8 a, nam_s8 b) { return vmulq_s16(a, b); }
static inline nam_i4 nam_i4_zero(void) { return vdupq_n_s32(0); }
static inline nam_i4 nam_i4_add(nam_i4 a, nam_i4 b) { return vaddq_s32(a, b); }
static inline nam_i4 nam_i4_dot(nam_s8 a, nam_s8 b) {
  int32x4_t lo = vmull_s16(vget_low_s16(a), vget_low_s16(b)), hi = vmull_s16(vget_high_s16(a), vget_high_s16(b));
#if defined(__aarch64__)
  return vpaddq_s32(lo, hi);
#else
  return vcombine_s32(vpadd_s32(vget_low_s32(lo), vget_high_s32(lo)), vpadd_s32(vget_low_s32(hi), vget_high_s32(hi)));
#endif
}
static inline nam_i4 nam_i4_widen_pairs(nam_s8 a) { return vpaddlq_s16(a); }
static inline void nam_i4_store(int32_t *p, nam_i4 a) { vst1q_s32(p, a); }
#else
#define NAM_SIMD_NAME "scalar"
typedef struct { float v[4]; } nam_f4;
typedef struct { int16_t v[8]; } nam_s8;
typedef struct { int32_t v[4]; } nam_i4;
static inline nam_f4 nam_f4_zero(void) { nam_f4 r; memset(&r, 0, sizeof r); return r; }
static inline nam_f4 nam_f4_load(const float *p) { nam_f4 r; memcpy(r.v, p, 16); return r; }
static inline nam_f4 nam_f4_add(nam_f4 a, nam_f4 b) { for (int i = 0; i < 4; i++) a.v[i] += b.v[i]; return a; }
static inline nam_f4 nam_f4_mul(nam_f4 a, nam_f4 b) { for (int i = 0; i < 4; i++) a.v[i] *= b.v[i]; return a; }
static inline void nam_f4_store(float *p, nam_f4 a) { memcpy(p, a.v, 16); }
static inline nam_s8 nam_s8_zero(void) { nam_s8 r; memset(&r, 0, sizeof r); return r; }
static inline nam_s8 nam_s8_load(const int16_t *p) { nam_s8 r; memcpy(r.v, p, 16); return r; }
static inline nam_s8 nam_s8_add(nam_s8 a, nam_s8 b) {
  for (int i = 0; i < 8; i++) a.v[i] = (int16_t)(uint16_t)((uint16_t)a.v[i] + (uint16_t)b.v[i]);
  return a;
}
static inline nam_s8 nam_s8_mul(nam_s8 a, nam_s8 b) {
  for (int i = 0; i < 8; i++) a.v[i] = (int16_t)(uint16_t)((uint32_t)(int32_t)a.v[i] * (uint32_t)(int32_t)b.v[i]);
  return a;
}
static inline nam_i4 nam_i4_zero(void) { nam_i4 r; memset(&r, 0, sizeof r); return r; }
static inline nam_i4 nam_i4_add(nam_i4 a, nam_i4 b) {
  for (int i = 0; i < 4; i++) a.v[i] = (int32_t)((uint32_t)a.v[i] + (uint32_t)b.v[i]);
  return a;
}
static inline nam_i4 nam_i4_dot(nam_s8 a, nam_s8 b) {
  nam_i4 r;
  for (int i = 0; i < 4; i++) r.v[i] = (int32_t)a.v[2 * i] * b.v[2 * i] + (int32_t)a.v[2 * i + 1] * b.v[2 * i + 1];
  return r;
}
static inline nam_i4 nam_i4_widen_pairs(nam_s8 a) {
  nam_i4 r;
  for (int i = 0; i < 4; i++) r.v[i] = (int32_t)a.v[2 * i] + a.v[2 * i + 1];
  return r;
}
static inline void nam_i4_store(int32_t *p, nam_i4 a) { memcpy(p, a.v, 16); }
#endif

// fixed reduction order shared by every backend: (l0 + l1) + (l2 + l3)
static inline float nam_f4_sum(nam_f4 a) {
  float l[4];
  nam_f4_store(l, a);
  return (l[0] + l[1]) + (l[2] + l[3]);
}
static inline int32_t nam_i4_sum(nam_i4 a) {
  int32_t l[4];
  nam_i4_store(l, a);
  return (int32_t)((uint32_t)l[0] + (uint32_t)l[1] + (uint32_t)l[2] + (uint32_t)l[3]);
}

#endif // NAM_SIMD_H
