// neural-amx: internal model representation shared by the loader, the kernels and the runner.
#ifndef NAM_INTERNAL_H
#define NAM_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "nam.h"

typedef enum {
  NAM_OP_DENSE, NAM_OP_CONV1D, NAM_OP_GRU, NAM_OP_LAYERNORM, NAM_OP_AFFINE,
  NAM_OP_RELU, NAM_OP_LEAKY_RELU, NAM_OP_SIGMOID, NAM_OP_TANH, NAM_OP_GELU, NAM_OP_GELU_TANH, NAM_OP_SILU,
  NAM_OP_SOFTMAX, NAM_OP_TRANSPOSE,
} nam_op;

// A weight matrix rows x cols prepared for the kernels: rows padded to cols_pad (multiple of 16) with zeros.
typedef struct {
  int rows, cols, cols_pad;
  int bits;      // 32 float, 8 or 4 integer
  float *wf;     // float weights (bits 32)
  int16_t *wq;   // integer weights widened to int16 lanes (bits 8/4)
  float *scale;  // per-row weight scale (integer)
  float *bias;   // rows (zeros when the layer has none)
  int flush;     // int16-lane accumulation steps that stay exact (0 = not applicable)
} nam_matrix;

typedef struct {
  nam_op op;
  int in, out;
  float alpha;                 // leaky_relu slope, layernorm epsilon
  nam_matrix m;                // dense, conv1d (rows = out channels, cols = in channels x kernel), gru input weights
  nam_matrix h;                // gru recurrent weights
  int in_ch, out_ch, kernel, stride, pad, len_in, len_out; // conv1d
  int hidden, state_offset;    // gru
  int trows, tcols;            // transpose: rows x cols -> cols x rows
  float *g, *b;                // layernorm gamma/beta, affine scale/shift
} nam_layer;

struct nam_model {
  int n;
  nam_layer *layers;
  int inputs, outputs, width;  // width: padded row stride of the activation buffers
  int acc16, batch;            // current kernel choices
  int auto_batch;
  int slots, state_size;
  float *state;                // slots x state_size
  float *act[2];               // NAM_BATCH x width each
  int16_t *q;                  // quantized rows: max(NAM_BATCH, patch rows) x qwidth
  int qwidth;
  float *patch;                // conv1d patches (float): patch_rows x qwidth
  float *cout;                 // conv1d outputs per position: patch_rows x max out channels
  int16_t *qx;                 // one quantized activation row (width)
  int patch_rows, cout_width;
  float *tmp;                  // gru scratch: 6 x max hidden
  char **labels;
  int nlabels;
  nam_precision precision;
  size_t weight_bytes;
};

// kernels (nam_kernels.c)
void nam_dense_f32(const nam_matrix *m, const float *x, float *y);
void nam_dense_f32_batch(const nam_matrix *m, const float *x, size_t xstride, float *y, size_t ystride);
float nam_quantize_row(const float *x, int n, int npad, int16_t *q);
void nam_dense_int(const nam_matrix *m, const int16_t *q, float sx, float *y, int acc16);
void nam_dense_int_batch(const nam_matrix *m, const int16_t *q, size_t qstride, const float *sx, float *y,
                         size_t ystride, int acc16);

// runner (nam_run.c)
void nam_forward(nam_model *model, const float *in, int rows, float *out, int first_slot);

#endif // NAM_INTERNAL_H
