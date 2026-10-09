// Test/bench runner for the C core (native and wasm builds).
//   runner <model.safetensors> <inputs.bin> <out.bin> <precision 0-3> <kernel 0-2> <batch -1|0|1> <agents> [spec]
//   runner --bench <model.safetensors> <precision> <kernel> <batch> <samples> <millis>
// inputs.bin: u32 rows, u32 dim, rows x dim float32. Rows run in chunks of `agents` (one call each, slots 0..agents-1),
// so recurrent models see one time step per chunk. out.bin: rows x outputs float32.
#define _POSIX_C_SOURCE 200809L // clock_gettime
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../core/nam.h"

static void *read_file(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
  fseek(f, 0, SEEK_END);
  *len = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  void *buf = malloc(*len ? *len : 1);
  if (fread(buf, 1, *len, f) != *len) { fprintf(stderr, "short read %s\n", path); exit(2); }
  fclose(f);
  return buf;
}

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static nam_model *load(const char *path, int precision, int kernel, int batch, const char *spec) {
  size_t len;
  void *buf = read_file(path, &len);
  nam_options opt;
  nam_default_options(&opt);
  opt.precision = (nam_precision)precision;
  opt.kernel = (nam_kernel)kernel;
  opt.batch = batch;
  char err[256];
  nam_model *m = nam_load(buf, len, spec, &opt, err, sizeof err);
  free(buf);
  if (!m) { fprintf(stderr, "load failed: %s\n", err); exit(1); }
  return m;
}

int main(int argc, char **argv) {
  if (argc >= 8 && !strcmp(argv[1], "--bench")) {
    nam_model *m = load(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), NULL);
    int n = atoi(argv[6]);
    double budget = atof(argv[7]);
    float *x = calloc((size_t)n * nam_inputs(m), 4), *y = malloc((size_t)n * nam_outputs(m) * 4);
    for (int i = 0; i < n * nam_inputs(m); i++) x[i] = (float)((i * 2654435761u) % 2000) / 1000.0f - 1.0f;
    double best = 1e30;
    for (int round = 0; round < 5; round++) {
      int reps = 0;
      double t0 = now_ms(), dt;
      do { nam_run(m, x, n, y, 0); reps++; } while ((dt = now_ms() - t0) < budget);
      double ns = dt * 1e6 / ((double)reps * n);
      if (ns < best) best = ns;
    }
    char desc[512];
    nam_describe(m, desc, sizeof desc);
    printf("{\"ns\":%.1f,\"backend\":\"%s\",\"desc\":\"%s\"}\n", best, nam_backend(), desc);
    nam_free(m);
    return 0;
  }
  if (argc < 8) { fprintf(stderr, "usage: runner model inputs out precision kernel batch agents [spec]\n"); return 2; }
  nam_model *m = load(argv[1], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argc > 8 ? argv[8] : NULL);
  size_t len;
  unsigned char *in = read_file(argv[2], &len);
  unsigned rows, dim;
  memcpy(&rows, in, 4);
  memcpy(&dim, in + 4, 4);
  if ((int)dim != nam_inputs(m)) { fprintf(stderr, "inputs: file has %u, model wants %d\n", dim, nam_inputs(m)); return 1; }
  int agents = atoi(argv[7]);
  if (agents <= 0) agents = (int)rows;
  const float *x = (const float *)(in + 8);
  float *y = malloc((size_t)rows * nam_outputs(m) * 4);
  for (unsigned r = 0; r < rows; r += (unsigned)agents) {
    int count = rows - r < (unsigned)agents ? (int)(rows - r) : agents;
    if (nam_run(m, x + (size_t)r * dim, count, y + (size_t)r * nam_outputs(m), 0) != count) { fprintf(stderr, "run failed\n"); return 1; }
  }
  FILE *f = fopen(argv[3], "wb");
  fwrite(y, 4, (size_t)rows * nam_outputs(m), f);
  fclose(f);
  nam_free(m);
  return 0;
}
