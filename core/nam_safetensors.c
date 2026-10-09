// neural-amx: safetensors reader.
#include "nam_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nam_math.h"

static const struct { const char *name; nst_dtype dtype; int size; } kDtypes[] = {
  {"F32", NST_F32, 4}, {"F16", NST_F16, 2}, {"BF16", NST_BF16, 2}, {"F64", NST_F64, 8}, {"I8", NST_I8, 1},
  {"U8", NST_U8, 1},   {"I16", NST_I16, 2}, {"I32", NST_I32, 4},   {"I64", NST_I64, 8}, {"BOOL", NST_BOOL, 1},
};

const char *nst_dtype_name(nst_dtype dtype) {
  for (size_t i = 0; i < sizeof kDtypes / sizeof *kDtypes; i++)
    if (kDtypes[i].dtype == dtype) return kDtypes[i].name;
  return "?";
}

static int fail(char *err, size_t errlen, const char *fmt, const char *arg) {
  if (err && errlen) snprintf(err, errlen, fmt, arg);
  return 0;
}

int nst_open(nst_file *f, const uint8_t *buf, size_t len, char *err, size_t errlen) {
  memset(f, 0, sizeof *f);
  if (len < 8) return fail(err, errlen, "safetensors: file too small%s", "");
  uint64_t hlen = 0;
  for (int i = 7; i >= 0; i--) hlen = hlen << 8 | buf[i];
  if (hlen > len - 8 || hlen > (100u << 20)) return fail(err, errlen, "safetensors: bad header size%s", "");
  f->header = nj_parse((const char *)buf + 8, (size_t)hlen, err, errlen);
  if (!f->header) return 0;
  if (f->header->type != NJ_OBJ) { nst_close(f); return fail(err, errlen, "safetensors: header is not an object%s", ""); }
  const uint8_t *data = buf + 8 + hlen;
  size_t data_len = len - 8 - (size_t)hlen;
  f->tensors = calloc((size_t)f->header->count + 1, sizeof *f->tensors);
  for (nj *e = f->header->child; e; e = e->next) {
    if (!strcmp(e->key, "__metadata__")) { f->metadata = e; continue; }
    nst_tensor *t = &f->tensors[f->count];
    t->name = e->key;
    const char *dt = nj_str(e, "dtype", "");
    int size = 0;
    for (size_t i = 0; i < sizeof kDtypes / sizeof *kDtypes; i++)
      if (!strcmp(dt, kDtypes[i].name)) { t->dtype = kDtypes[i].dtype; size = kDtypes[i].size; }
    if (!size) { nst_close(f); return fail(err, errlen, "safetensors: unsupported dtype in '%s'", e->key); }
    nj *shape = nj_get(e, "shape"), *offs = nj_get(e, "data_offsets");
    if (!shape || shape->type != NJ_ARR || shape->count > NST_MAX_DIMS || !offs || offs->count != 2) {
      nst_close(f);
      return fail(err, errlen, "safetensors: bad entry '%s'", e->key);
    }
    t->ndim = shape->count;
    t->elements = 1;
    for (int d = 0; d < t->ndim; d++) {
      double v = nj_at(shape, d)->num;
      if (v < 0 || v > 1e12) { nst_close(f); return fail(err, errlen, "safetensors: bad shape in '%s'", e->key); }
      t->shape[d] = (int64_t)v;
      t->elements *= t->shape[d];
    }
    double b = nj_at(offs, 0)->num, en = nj_at(offs, 1)->num;
    if (b < 0 || en < b || en > (double)data_len || (en - b) != (double)t->elements * size) {
      nst_close(f);
      return fail(err, errlen, "safetensors: data offsets of '%s' do not match its shape", e->key);
    }
    t->data = data + (size_t)b;
    t->nbytes = (size_t)(en - b);
    f->count++;
  }
  return 1;
}

void nst_close(nst_file *f) {
  nj_free(f->header);
  free(f->tensors);
  memset(f, 0, sizeof *f);
}

const nst_tensor *nst_find(const nst_file *f, const char *name) {
  for (int i = 0; i < f->count; i++)
    if (!strcmp(f->tensors[i].name, name)) return &f->tensors[i];
  return NULL;
}

const char *nst_metadata(const nst_file *f, const char *key) { return f->metadata ? nj_str(f->metadata, key, NULL) : NULL; }

static float half_to_float(uint16_t h) { // exact for every finite half; inf/nan map to large finite values
  float f = nam_bits_to_float((uint32_t)(h & 0x7fff) << 13) * nam_bits_to_float(0x77800000u);
  uint32_t u;
  memcpy(&u, &f, 4);
  u |= (uint32_t)(h & 0x8000) << 16;
  return nam_bits_to_float(u);
}

int nst_to_f32(const nst_tensor *t, float *out) {
  const uint8_t *p = t->data;
  for (int64_t i = 0; i < t->elements; i++) {
    switch (t->dtype) {
      case NST_F32: memcpy(&out[i], p + 4 * i, 4); break;
      case NST_F16: { uint16_t h; memcpy(&h, p + 2 * i, 2); out[i] = half_to_float(h); break; }
      case NST_BF16: { uint16_t h; memcpy(&h, p + 2 * i, 2); out[i] = nam_bits_to_float((uint32_t)h << 16); break; }
      case NST_F64: { double d; memcpy(&d, p + 8 * i, 8); out[i] = (float)d; break; }
      case NST_I8: out[i] = (float)(int8_t)p[i]; break;
      case NST_U8: case NST_BOOL: out[i] = (float)p[i]; break;
      case NST_I16: { int16_t v; memcpy(&v, p + 2 * i, 2); out[i] = (float)v; break; }
      case NST_I32: { int32_t v; memcpy(&v, p + 4 * i, 4); out[i] = (float)v; break; }
      case NST_I64: { int64_t v; memcpy(&v, p + 8 * i, 8); out[i] = (float)v; break; }
    }
  }
  return 1;
}
