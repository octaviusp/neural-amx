// neural-amx AMX Mod X module: natives over the C core (core/nam.h).
// Pawn Float arrays are 32-bit IEEE cells, so they go to the kernels in place, with no copies. Every array is
// bounds-checked against the plugin's memory before use.
// SPDX-License-Identifier: MIT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "amxxmodule.h"
#include "../core/nam.h"

namespace {

const int kMaxModels = 256;
nam_model *g_models[kMaxModels];
uint32_t g_rng = 0x9E3779B9u;

bool cells_ok(AMX *amx, cell addr, long count) {
  return addr >= 0 && count >= 0 && (long long)addr + (long long)count * (long long)sizeof(cell) <= (long long)amx->stp;
}

float *float_array(AMX *amx, cell addr, long count, const char *what) {
  if (!cells_ok(amx, addr, count)) {
    MF_LogError(amx, AMX_ERR_NATIVE, "neural: %s needs %ld cells inside the plugin memory", what, count);
    return NULL;
  }
  return reinterpret_cast<float *>(MF_GetAmxAddr(amx, addr));
}

nam_model *model_at(AMX *amx, cell h) {
  if (h < 0 || h >= kMaxModels || !g_models[h]) {
    MF_LogError(amx, AMX_ERR_NATIVE, "neural: invalid model handle %d", (int)h);
    return NULL;
  }
  return g_models[h];
}

void *read_file(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void *buf = n > 0 ? malloc((size_t)n) : NULL;
  if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); buf = NULL; }
  fclose(f);
  *len = (size_t)n;
  return buf;
}

// Neural:neural_load(const file[], NeuralPrecision:precision = NEURAL_STORED, const spec[] = "")
cell AMX_NATIVE_CALL n_load(AMX *amx, cell *params) {
  int flen, slen;
  const char *file = MF_GetAmxString(amx, params[1], 0, &flen);
  char spec[1024];
  snprintf(spec, sizeof spec, "%s", MF_GetAmxString(amx, params[3], 1, &slen));
  if (strstr(file, "..")) {
    MF_LogError(amx, AMX_ERR_NATIVE, "neural: '%s' must stay inside the data directory", file);
    return -1;
  }
  const char *path = MF_BuildPathname("%s/%s", MF_GetLocalInfo("amxx_datadir", "addons/amxmodx/data"), file);
  size_t len = 0;
  void *buf = read_file(path, &len);
  if (!buf) {
    MF_LogError(amx, AMX_ERR_NATIVE, "neural: cannot read %s", path);
    return -1;
  }
  nam_options opt;
  nam_default_options(&opt);
  opt.precision = (nam_precision)params[2];
  char err[256];
  nam_model *m = nam_load(buf, len, spec[0] ? spec : NULL, &opt, err, sizeof err);
  free(buf);
  if (!m) {
    MF_LogError(amx, AMX_ERR_NATIVE, "neural: %s: %s", file, err);
    return -1;
  }
  for (int i = 0; i < kMaxModels; i++)
    if (!g_models[i]) { g_models[i] = m; return i; }
  nam_free(m);
  MF_LogError(amx, AMX_ERR_NATIVE, "neural: more than %d models loaded", kMaxModels);
  return -1;
}

// neural_free(&Neural:model)
cell AMX_NATIVE_CALL n_free(AMX *amx, cell *params) {
  if (!cells_ok(amx, params[1], 1)) return 0;
  cell *h = MF_GetAmxAddr(amx, params[1]);
  if (*h >= 0 && *h < kMaxModels && g_models[*h]) {
    nam_free(g_models[*h]);
    g_models[*h] = NULL;
  }
  *h = -1;
  return 1;
}

cell AMX_NATIVE_CALL n_inputs(AMX *amx, cell *params) { nam_model *m = model_at(amx, params[1]); return m ? nam_inputs(m) : -1; }
cell AMX_NATIVE_CALL n_outputs(AMX *amx, cell *params) { nam_model *m = model_at(amx, params[1]); return m ? nam_outputs(m) : -1; }

// neural_run(Neural:model, const Float:input[], Float:output[], slot = 0) -> argmax of the output
cell AMX_NATIVE_CALL n_run(AMX *amx, cell *params) {
  nam_model *m = model_at(amx, params[1]);
  if (!m) return -1;
  const float *in = float_array(amx, params[2], nam_inputs(m), "input");
  float *out = float_array(amx, params[3], nam_outputs(m), "output");
  if (!in || !out) return -1;
  if (nam_run(m, in, 1, out, params[4]) != 1) {
    MF_LogError(amx, AMX_ERR_NATIVE, "neural: slot %d out of range", (int)params[4]);
    return -1;
  }
  return nam_argmax(out, nam_outputs(m));
}

// neural_run_batch(Neural:model, const Float:inputs[], count, Float:outputs[], first_slot = 0) -> count
cell AMX_NATIVE_CALL n_run_batch(AMX *amx, cell *params) {
  nam_model *m = model_at(amx, params[1]);
  long count = params[3];
  if (!m || count < 0) return -1;
  const float *in = float_array(amx, params[2], count * nam_inputs(m), "inputs");
  float *out = float_array(amx, params[4], count * nam_outputs(m), "outputs");
  if (!in || !out) return -1;
  if (nam_run(m, in, (int)count, out, params[5]) != count) {
    MF_LogError(amx, AMX_ERR_NATIVE, "neural: slots %d..%d out of range", (int)params[5], (int)(params[5] + count - 1));
    return -1;
  }
  return (cell)count;
}

cell AMX_NATIVE_CALL n_reset(AMX *amx, cell *params) {
  nam_model *m = model_at(amx, params[1]);
  if (m) nam_reset(m, params[2]);
  return m != NULL;
}

cell AMX_NATIVE_CALL n_tune(AMX *amx, cell *params) {
  nam_model *m = model_at(amx, params[1]);
  return m ? nam_tune(m) : 0;
}

// neural_info(Neural:model, buffer[], maxlen)
cell AMX_NATIVE_CALL n_info(AMX *amx, cell *params) {
  nam_model *m = model_at(amx, params[1]);
  if (!m) return 0;
  char buf[1024];
  nam_describe(m, buf, sizeof buf);
  return MF_SetAmxString(amx, params[2], buf, params[3]);
}

// neural_label(Neural:model, index, buffer[], maxlen)
cell AMX_NATIVE_CALL n_label(AMX *amx, cell *params) {
  nam_model *m = model_at(amx, params[1]);
  if (!m) return 0;
  const char *s = nam_label(m, params[2]);
  return MF_SetAmxString(amx, params[3], s ? s : "", params[4]);
}

// neural_argmax(const Float:values[], count)
cell AMX_NATIVE_CALL n_argmax(AMX *amx, cell *params) {
  const float *v = float_array(amx, params[1], params[2], "values");
  return v ? nam_argmax(v, params[2]) : -1;
}

// neural_softmax(Float:values[], count, Float:temperature = 1.0)
cell AMX_NATIVE_CALL n_softmax(AMX *amx, cell *params) {
  float *v = float_array(amx, params[1], params[2], "values");
  if (!v) return 0;
  nam_softmax(v, params[2], amx_ctof(params[3]));
  return 1;
}

// neural_sample(const Float:probabilities[], count, Float:uniform = -1.0)
cell AMX_NATIVE_CALL n_sample(AMX *amx, cell *params) {
  const float *p = float_array(amx, params[1], params[2], "probabilities");
  if (!p) return -1;
  float u = amx_ctof(params[3]);
  if (u < 0.0f) { // xorshift32
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    u = (float)(g_rng >> 8) / 16777216.0f;
  }
  return nam_sample(p, params[2], u);
}

// neural_clock() -> monotonic microseconds (wraps every ~71 minutes); for per-agent budgets
cell AMX_NATIVE_CALL n_clock(AMX *amx, cell *params) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (cell)(uint32_t)((uint64_t)t.tv_sec * 1000000u + (uint64_t)t.tv_nsec / 1000u);
}

AMX_NATIVE_INFO g_natives[] = {
  {"neural_load", n_load},       {"neural_free", n_free},         {"neural_inputs", n_inputs},
  {"neural_outputs", n_outputs}, {"neural_run", n_run},           {"neural_run_batch", n_run_batch},
  {"neural_reset", n_reset},     {"neural_tune", n_tune},         {"neural_info", n_info},
  {"neural_label", n_label},     {"neural_argmax", n_argmax},     {"neural_softmax", n_softmax},
  {"neural_sample", n_sample},   {"neural_clock", n_clock},       {NULL, NULL},
};

} // namespace

void OnAmxxAttach() {
  g_rng ^= (uint32_t)time(NULL);
  MF_AddNatives(g_natives);
}

void OnPluginsUnloaded() { // map change: plugins reload and load their models again
  for (int i = 0; i < kMaxModels; i++) {
    nam_free(g_models[i]);
    g_models[i] = NULL;
  }
}
