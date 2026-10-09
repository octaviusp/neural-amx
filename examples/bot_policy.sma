// Example: a learned policy for every alive bot, decided 10 times per second and staggered across frames.
// Put policy.safetensors (examples/train_torch.py) in addons/amxmodx/data/neural/.
#include <amxmodx>
#include <engine>
#include <neural_amx>

#define FEATURES 64
#define ACTIONS 8
#define GROUPS 3 // bots are split into groups; one group decides per 0.033 s tick -> each bot at ~10 Hz

new Neural:g_policy = Invalid_Neural;
new Float:g_features[32][FEATURES], Float:g_actions[32][ACTIONS];
new g_ids[32], g_group;

public plugin_init()
{
	register_plugin("neural bots", "1.0", "neural-amx");
	g_policy = neural_load("neural/policy.safetensors", NEURAL_INT4);
	if (g_policy == Invalid_Neural)
		set_fail_state("neural/policy.safetensors could not be loaded");
	new info[256];
	neural_info(g_policy, info, charsmax(info));
	log_amx("%s", info);
	set_task(0.033, "decide", .flags = "b");
}

public decide()
{
	new players[32], num, count = 0;
	get_players(players, num, "ad"); // alive bots
	for (new i = 0; i < num; i++)
	{
		if (players[i] % GROUPS != g_group) continue;
		build_features(players[i], g_features[count]);
		g_ids[count++] = players[i];
	}
	g_group = (g_group + 1) % GROUPS;
	if (!count) return;
	neural_run_batch(g_policy, g_features[0], count, g_actions[0]); // one call for the whole group
	for (new i = 0; i < count; i++)
	{
		new action = neural_sample(g_actions[i], ACTIONS); // the model ends in softmax
		apply_action(g_ids[i], action);
	}
}

build_features(id, Float:f[])
{
	new Float:origin[3], Float:velocity[3];
	entity_get_vector(id, EV_VEC_origin, origin);
	entity_get_vector(id, EV_VEC_velocity, velocity);
	for (new c = 0; c < 3; c++) f[c] = origin[c] / 4096.0;
	for (new c = 0; c < 3; c++) f[3 + c] = velocity[c] / 320.0;
	f[6] = float(get_user_health(id)) / 100.0;
	for (new k = 7; k < FEATURES; k++) f[k] = 0.0; // add enemies, traces, weapon state... exactly as in training
}

apply_action(id, action)
{
	new label[32];
	neural_label(g_policy, action, label, charsmax(label));
	// steer your bot here (e.g. RunPlayerMove-based bots or a YaPB command bridge)
	#pragma unused id
}
