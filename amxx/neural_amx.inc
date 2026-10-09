// neural-amx: fast, deterministic neural-network inference for AMX Mod X.
// Models are safetensors files under addons/amxmodx/data/ (see https://github.com/octaviusp/neural-amx).
//
//   new Neural:g_policy;
//   public plugin_init() { g_policy = neural_load("neural/policy.safetensors", NEURAL_INT4); }
//   ...
//   new Float:features[64], Float:actions[8];
//   new best = neural_run(g_policy, features, actions, .slot = id);
#if defined _neural_amx_included
  #endinput
#endif
#define _neural_amx_included

#pragma reqlib neural_amx
#if !defined AMXMODX_NOAUTOLOAD
  #pragma loadlib neural_amx
#endif

enum Neural
{
	Invalid_Neural = -1
};

enum NeuralPrecision
{
	NEURAL_STORED = 0, // as the file stores it
	NEURAL_F32,        // float32 everywhere
	NEURAL_INT8,       // int8 weights, dynamic int8 activations
	NEURAL_INT4        // int4 hidden layers, int8 first/last layer and recurrences (smallest, fastest when wide)
};

/**
 * Loads a model from <amxx_datadir>/<file>. The fastest kernel for this host is picked automatically (~10-30 ms).
 *
 * @param file       Path relative to the data directory, e.g. "neural/policy.safetensors".
 * @param precision  Weight precision (quantization at load needs no calibration).
 * @param spec       Layer list for a plain PyTorch state_dict without neural-amx metadata,
 *                   e.g. "input(64) dense(fc1) relu dense(fc2) softmax".
 * @return           Model handle, or Invalid_Neural (an error is logged).
 */
native Neural:neural_load(const file[], NeuralPrecision:precision = NEURAL_STORED, const spec[] = "");

/** Frees a model and sets the handle to Invalid_Neural. Models are also freed on map change. */
native neural_free(&Neural:model);

native neural_inputs(Neural:model);
native neural_outputs(Neural:model);

/**
 * Runs one inference. Recurrent models (GRU) keep a hidden state per slot: use the player id as the slot.
 *
 * @return  Index of the largest output, or -1 on error.
 */
native neural_run(Neural:model, const Float:input[], Float:output[], slot = 0);

/**
 * Runs `count` inferences in one call, the fastest way to drive several agents. Rows are flat
 * (count x inputs); a 2D array Float:x[32][INPUTS] works: pass x[0]. Row i uses slot first_slot + i.
 *
 * @return  count, or -1 on error.
 */
native neural_run_batch(Neural:model, const Float:inputs[], count, Float:outputs[], first_slot = 0);

/** Clears recurrent state (slot -1: all slots), e.g. when a player respawns. */
native neural_reset(Neural:model, slot = -1);

/** Re-times the kernel choices on this host. */
native neural_tune(Neural:model);

/** Writes a one-line description: layers, precision per layer, kernel, batch, weight bytes. */
native neural_info(Neural:model, buffer[], maxlen);

/** Copies the name of output `index` (from the file metadata) or an empty string. */
native neural_label(Neural:model, index, buffer[], maxlen);

native neural_argmax(const Float:values[], count);

/** In-place softmax with temperature (> 1 flattens, < 1 sharpens). */
native neural_softmax(Float:values[], count, Float:temperature = 1.0);

/** Samples an index from probabilities; uniform in [0, 1), or a negative value to draw one. */
native neural_sample(const Float:probabilities[], count, Float:uniform = -1.0);

/** Monotonic microseconds (wraps every ~71 minutes): measure what your agents cost per frame. */
native neural_clock();
