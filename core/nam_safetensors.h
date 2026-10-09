// neural-amx: safetensors reader (https://github.com/huggingface/safetensors). Tensors point into the caller's buffer.
#ifndef NAM_SAFETENSORS_H
#define NAM_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#include "nam_json.h"

typedef enum { NST_F32, NST_F16, NST_BF16, NST_F64, NST_I8, NST_U8, NST_I16, NST_I32, NST_I64, NST_BOOL } nst_dtype;

#define NST_MAX_DIMS 8

typedef struct {
  const char *name;
  nst_dtype dtype;
  int ndim;
  int64_t shape[NST_MAX_DIMS];
  int64_t elements;
  const uint8_t *data;
  size_t nbytes;
} nst_tensor;

typedef struct {
  nj *header;             // parsed JSON header (owns the names)
  nst_tensor *tensors;
  int count;
  const nj *metadata;     // "__metadata__" object or NULL (string -> string)
} nst_file;

int nst_open(nst_file *file, const uint8_t *buf, size_t len, char *err, size_t errlen);
void nst_close(nst_file *file);
const nst_tensor *nst_find(const nst_file *file, const char *name);
const char *nst_metadata(const nst_file *file, const char *key);
// Converts any floating or integer tensor to float32 (exact for F32/F16/BF16 and integers up to 2^24).
int nst_to_f32(const nst_tensor *tensor, float *out);
const char *nst_dtype_name(nst_dtype dtype);

#endif // NAM_SAFETENSORS_H
