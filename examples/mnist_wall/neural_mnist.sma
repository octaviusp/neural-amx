// Neural MNIST wall: draw a digit on a wall with bullets and a neural network reads it.
//
//   say /mnist          open a 28x28 canvas on the wall you are looking at (a frame is drawn around it)
//   say /mnist ok       the network's answer goes to the chat
//   say /mnist clear    erase the drawing          say /mnist off    close the canvas
// While the canvas is open, every shot that lands inside it is a dot; the HUD keeps the network's top 3 on screen and
// the server console prints each reading as "[MNIST] <name> PREDICTED: 3 (91.2%), 8 (5.0%), 5 (2.1%) | 14 shots".
//
// cvars: mnist_model "neural/mnist.safetensors" (under the data dir), mnist_cell 4.0 (units per canvas cell),
//        mnist_mode 0 (0 = bullet impacts, where the decals are; 1 = exact crosshair point, ignores spread),
//        mnist_live 1 (live HUD + console line), mnist_debug 0 (1: print why each shot is kept or dropped).
// Admin/test: mnist_open <id>, mnist_guess <id>, mnist_clear <id>, mnist_status; public mnist_api_* functions for callfunc.
//
// The renderer below (render_canvas) is the one examples/mnist_wall/train.py trains on: keep them identical.
// Needs the neural-amx module (https://github.com/octaviusp/neural-amx). Optional config: <configsdir>/neural-mnist.cfg.
// License: MIT.

#include <amxmodx>
#include <amxmisc>
#include <fakemeta>
#include <hamsandwich>
#include <neural_amx>

#define PLUGIN_NAME    "Neural MNIST wall"
#define PLUGIN_VERSION "1.0.0"
#define PLUGIN_AUTHOR  "neural-amx"

#define CELLS 28
#define PIXELS 784
#define MAX_DOTS 200
#define CLASSES 10

const Float:BRUSH = 1.7;           // dot paints full intensity within 0.7 cells, fades to 0 at 1.7
const Float:BOX = 18.0;            // dots are fitted into an 18-cell box, then centered on the center of mass
const Float:PLANE_TOLERANCE = 6.0; // a hit belongs to the canvas when it is this close to its plane
const TASK_FRAME = 6100;           // + player id
const TASK_READ = 6200;            // + player id: one reading per burst of shots
const Float:READ_DELAY = 0.15;     // a burst of shots (automatic fire, shotgun pellets) gives one reading
const Float:HUD_HOLD = 6.0;        // the reading stays this long...
const Float:HUD_RESEND = 4.0;      // ...and is sent again before it expires while the canvas is open
const Float:HUD_Y = 0.72;

new Neural:g_model = Invalid_Neural;
new g_beam;
new g_cvar_model, g_cvar_cell, g_cvar_mode, g_cvar_live, g_cvar_debug;

new bool:g_open[MAX_PLAYERS + 1];
new Float:g_corner[MAX_PLAYERS + 1][3]; // top-left corner, 1 unit off the wall
new Float:g_u[MAX_PLAYERS + 1][3];      // canvas right
new Float:g_n[MAX_PLAYERS + 1][3];      // wall normal, toward the player
new Float:g_cell[MAX_PLAYERS + 1];
new Float:g_last_shot[MAX_PLAYERS + 1];
new g_reading[MAX_PLAYERS + 1][96];     // the HUD text of the latest reading, "" when there is none
new Float:g_resend_at[MAX_PLAYERS + 1];
new bool:g_hud_flip[MAX_PLAYERS + 1];
new g_dots[MAX_PLAYERS + 1];
new Float:g_dot[MAX_PLAYERS + 1][MAX_DOTS][2]; // (column, row) in canvas cells

new Float:g_px[MAX_DOTS], Float:g_py[MAX_DOTS];
new Float:g_img[PIXELS], Float:g_probs[CLASSES];

public plugin_precache()
{
	g_beam = precache_model("sprites/laserbeam.spr");
}

public plugin_init()
{
	register_plugin(PLUGIN_NAME, PLUGIN_VERSION, PLUGIN_AUTHOR);
	g_cvar_model = create_cvar("mnist_model", "neural/mnist.safetensors", FCVAR_NONE, "Model file under the AMX Mod X data directory");
	g_cvar_cell = create_cvar("mnist_cell", "4.0", FCVAR_NONE, "World units per canvas cell (the canvas is 28 cells wide)", true, 2.0, true, 8.0);
	g_cvar_mode = create_cvar("mnist_mode", "0", FCVAR_NONE, "0: bullet impacts, 1: exact crosshair point", true, 0.0, true, 1.0);
	g_cvar_live = create_cvar("mnist_live", "1", FCVAR_NONE, "Live top 3 on the HUD and the server console while drawing", true, 0.0, true, 1.0);
	g_cvar_debug = create_cvar("mnist_debug", "0", FCVAR_NONE, "Print every shot's canvas position", true, 0.0, true, 1.0);
	register_clcmd("say", "CmdSay");
	register_clcmd("say_team", "CmdSay");
	register_concmd("mnist_open", "CmdAdmin", ADMIN_RCON, "<player id> - open a canvas where the player looks");
	register_concmd("mnist_guess", "CmdAdmin", ADMIN_RCON, "<player id> - print the network's answer");
	register_concmd("mnist_clear", "CmdAdmin", ADMIN_RCON, "<player id> - erase the drawing");
	register_concmd("mnist_status", "CmdStatus", ADMIN_RCON, "- print the loaded model");
	RegisterHam(Ham_TraceAttack, "worldspawn", "OnTraceAttack", 1);
	RegisterHam(Ham_TraceAttack, "func_wall", "OnTraceAttack", 1);
}

public plugin_cfg()
{
	// a server or mod pack may ship neural-mnist.cfg next to the other configs
	new cfg[160];
	get_configsdir(cfg, charsmax(cfg));
	add(cfg, charsmax(cfg), "/neural-mnist.cfg");
	if (file_exists(cfg))
	{
		server_cmd("exec %s", cfg);
		server_exec();
	}
	new file[128];
	get_pcvar_string(g_cvar_model, file, charsmax(file));
	g_model = neural_load(file, NEURAL_STORED);
	if (g_model == Invalid_Neural)
	{
		log_amx("[MNIST] cannot load %s: put mnist.safetensors under the data directory", file);
		return;
	}
	new info[256];
	neural_info(g_model, info, charsmax(info));
	log_amx("[MNIST] %s", info);
}

public client_disconnected(id)
{
	g_open[id] = false;
	g_reading[id][0] = EOS;
	remove_task(TASK_FRAME + id);
	remove_task(TASK_READ + id);
}

// ---------- commands ----------
public CmdSay(id)
{
	new text[32];
	read_args(text, charsmax(text));
	remove_quotes(text);
	if (!equali(text, "/mnist", 6))
		return PLUGIN_CONTINUE;
	if (equali(text[6], " ok"))
		announce(id);
	else if (equali(text[6], " clear"))
		clear_canvas(id);
	else if (equali(text[6], " off"))
		close_canvas(id);
	else if (!open_canvas(id))
		client_print(id, print_chat, "[MNIST] Look at a flat wall about 112 units wide (within 600 units) and try again.");
	else
		client_print(id, print_chat, "[MNIST] Draw a digit with your shots inside the frame, then say /mnist ok.");
	return PLUGIN_HANDLED;
}

public CmdAdmin(id, level, cid)
{
	if (!cmd_access(id, level, cid, 2))
		return PLUGIN_HANDLED;
	new cmd[16], arg[8];
	read_argv(0, cmd, charsmax(cmd));
	read_argv(1, arg, charsmax(arg));
	new player = str_to_num(arg);
	if (player < 1 || player > MaxClients || !is_user_connected(player))
	{
		console_print(id, "[MNIST] no player %d", player);
		return PLUGIN_HANDLED;
	}
	if (equal(cmd, "mnist_open"))
		console_print(id, "[MNIST] open: %d", open_canvas(player));
	else if (equal(cmd, "mnist_guess"))
		announce(player);
	else
		clear_canvas(player);
	return PLUGIN_HANDLED;
}

public CmdStatus(id, level, cid)
{
	if (!cmd_access(id, level, cid, 1))
		return PLUGIN_HANDLED;
	if (g_model == Invalid_Neural)
	{
		new file[128];
		get_pcvar_string(g_cvar_model, file, charsmax(file));
		console_print(id, "[MNIST] model not loaded (%s)", file);
		return PLUGIN_HANDLED;
	}
	new info[256];
	neural_info(g_model, info, charsmax(info));
	console_print(id, "[MNIST] %s", info);
	return PLUGIN_HANDLED;
}

// callfunc API (tests and other plugins)
public mnist_api_open(id) { return open_canvas(id); }
public mnist_api_clear(id) { clear_canvas(id); return 1; }
public mnist_api_close(id) { close_canvas(id); return 1; }
public mnist_api_guess(id) { return announce(id); }
public mnist_api_dots(id) { return g_dots[id]; }

// ---------- canvas ----------
eye_trace(id, Float:eye[3], Float:fwd[3], Float:hit[3], Float:normal[3])
{
	new Float:origin[3], Float:ofs[3], Float:angles[3], Float:end[3];
	pev(id, pev_origin, origin);
	pev(id, pev_view_ofs, ofs);
	pev(id, pev_v_angle, angles);
	engfunc(EngFunc_MakeVectors, angles);
	global_get(glb_v_forward, fwd);
	for (new i = 0; i < 3; i++)
	{
		eye[i] = origin[i] + ofs[i];
		end[i] = eye[i] + fwd[i] * 8192.0;
	}
	engfunc(EngFunc_TraceLine, eye, end, IGNORE_MONSTERS, id, 0);
	get_tr2(0, TR_vecEndPos, hit);
	get_tr2(0, TR_vecPlaneNormal, normal);
	new Float:fraction;
	get_tr2(0, TR_flFraction, fraction);
	return fraction < 1.0;
}

Float:plane_distance(id, const Float:p[3])
{
	return (p[0] - g_corner[id][0]) * g_n[id][0] + (p[1] - g_corner[id][1]) * g_n[id][1] + (p[2] - g_corner[id][2]) * g_n[id][2];
}

// true when a ray from the eye toward `target` stops on the canvas plane
bool:reaches_plane(id, const Float:eye[3], const Float:target[3])
{
	new Float:end[3], Float:hit[3];
	for (new i = 0; i < 3; i++)
		end[i] = eye[i] + (target[i] - eye[i]) * 1.05;
	engfunc(EngFunc_TraceLine, eye, end, IGNORE_MONSTERS, id, 0);
	get_tr2(0, TR_vecEndPos, hit);
	return floatabs(plane_distance(id, hit)) < PLANE_TOLERANCE;
}

bool:open_canvas(id)
{
	if (!is_user_alive(id))
		return false;
	new Float:eye[3], Float:fwd[3], Float:hit[3], Float:normal[3];
	if (!eye_trace(id, eye, fwd, hit, normal) || floatabs(normal[2]) > 0.3 || get_distance_f(eye, hit) > 600.0)
		return false;
	new Float:len = floatsqroot(normal[0] * normal[0] + normal[1] * normal[1]);
	new Float:cell = floatmax_(get_pcvar_float(g_cvar_cell), 1.0), Float:half = 14.0 * cell;
	// wall normal made horizontal; canvas right = (-n) x up, canvas up = world up
	g_n[id][0] = normal[0] / len;
	g_n[id][1] = normal[1] / len;
	g_n[id][2] = 0.0;
	g_u[id][0] = -g_n[id][1];
	g_u[id][1] = g_n[id][0];
	g_u[id][2] = 0.0;
	for (new i = 0; i < 3; i++)
		g_corner[id][i] = hit[i] + g_n[id][i] - g_u[id][i] * half + (i == 2 ? half : 0.0);
	g_cell[id] = cell;
	// the whole canvas must be on the same wall: check the four corners and the center
	new Float:p[3];
	for (new c = 0; c < 5; c++)
	{
		new Float:cu = c == 4 ? 14.0 : (c & 1) ? 27.5 : 0.5, Float:cv = c == 4 ? 14.0 : (c & 2) ? 27.5 : 0.5;
		canvas_point(id, cu, cv, p);
		if (!reaches_plane(id, eye, p))
			return false;
	}
	g_open[id] = true;
	g_dots[id] = 0;
	g_reading[id][0] = EOS;
	remove_task(TASK_FRAME + id);
	draw_frame(TASK_FRAME + id);
	set_task(2.0, "draw_frame", TASK_FRAME + id, _, _, "b");
	return true;
}

canvas_point(id, Float:col, Float:row, Float:p[3])
{
	for (new i = 0; i < 3; i++)
		p[i] = g_corner[id][i] + g_u[id][i] * col * g_cell[id] - (i == 2 ? row * g_cell[id] : 0.0) - g_n[id][i] * 0.5;
}

clear_canvas(id)
{
	g_dots[id] = 0;
	remove_task(TASK_READ + id);
	hide_reading(id);
	if (g_open[id] && is_user_connected(id) && !is_user_bot(id))
		client_print(id, print_center, "MNIST: canvas cleared");
}

close_canvas(id)
{
	g_open[id] = false;
	remove_task(TASK_FRAME + id);
	remove_task(TASK_READ + id);
	hide_reading(id);
}

public draw_frame(task)
{
	new id = task - TASK_FRAME;
	if (!g_open[id] || !is_user_connected(id))
	{
		remove_task(task);
		return;
	}
	if (is_user_bot(id))
		return;
	if (g_reading[id][0] && get_gametime() >= g_resend_at[id])
		send_reading(id);
	new Float:a[3], Float:b[3];
	new const Float:corners[5][2] = { {0.0, 0.0}, {28.0, 0.0}, {28.0, 28.0}, {0.0, 28.0}, {0.0, 0.0} };
	for (new k = 0; k < 4; k++)
	{
		canvas_point(id, corners[k][0], corners[k][1], a);
		canvas_point(id, corners[k + 1][0], corners[k + 1][1], b);
		message_begin(MSG_ONE_UNRELIABLE, SVC_TEMPENTITY, _, id);
		write_byte(TE_BEAMPOINTS);
		for (new i = 0; i < 3; i++) write_coord_f(a[i]);
		for (new i = 0; i < 3; i++) write_coord_f(b[i]);
		write_short(g_beam);
		write_byte(0);   // start frame
		write_byte(0);   // frame rate
		write_byte(22);  // life, 0.1 s units (refreshed every 2 s)
		write_byte(10);  // width
		write_byte(0);   // noise
		write_byte(80);  // r
		write_byte(255); // g
		write_byte(120); // b
		write_byte(180); // brightness
		write_byte(0);   // scroll
		message_end();
	}
}

// ---------- shots ----------
public OnTraceAttack(victim, attacker, Float:damage, Float:direction[3], trace, damagebits)
{
	if (attacker < 1 || attacker > MaxClients || !g_open[attacker] || !(damagebits & DMG_BULLET))
		return HAM_IGNORED;
	new Float:hit[3];
	if (get_pcvar_num(g_cvar_mode) == 1)
	{
		new Float:now = get_gametime(), Float:eye[3], Float:fwd[3], Float:normal[3];
		if (now == g_last_shot[attacker]) // shotgun pellets of one shot: one aim point
			return HAM_IGNORED;
		g_last_shot[attacker] = now;
		eye_trace(attacker, eye, fwd, hit, normal);
	}
	else
	{
		get_tr2(trace, TR_vecEndPos, hit);
	}
	add_dot(attacker, hit);
	return HAM_IGNORED;
}

add_dot(id, const Float:hit[3])
{
	new Float:d[3];
	for (new i = 0; i < 3; i++)
		d[i] = hit[i] - g_corner[id][i];
	new Float:col = (d[0] * g_u[id][0] + d[1] * g_u[id][1]) / g_cell[id], Float:row = -d[2] / g_cell[id];
	new Float:off = plane_distance(id, hit);
	new bool:keep = floatabs(off) <= PLANE_TOLERANCE && col >= 0.0 && col < 28.0 && row >= 0.0 && row < 28.0 && g_dots[id] < MAX_DOTS;
	if (get_pcvar_num(g_cvar_debug))
		server_print("MNIST shot id=%d col=%.2f row=%.2f plane=%.2f kept=%d", id, col, row, off, keep);
	if (!keep)
		return;
	g_dot[id][g_dots[id]][0] = col;
	g_dot[id][g_dots[id]][1] = row;
	g_dots[id]++;
	if (get_pcvar_num(g_cvar_live) && !task_exists(TASK_READ + id))
		set_task(READ_DELAY, "live_reading", TASK_READ + id);
}

// ---------- recognition ----------
// Writes the 28x28 input for player id's dots into g_img; mirrors render() in train.py.
render_canvas(id)
{
	new n = g_dots[id];
	new Float:xmin = 1.0e9, Float:xmax = -1.0e9, Float:ymin = 1.0e9, Float:ymax = -1.0e9;
	for (new k = 0; k < n; k++)
	{
		xmin = floatmin_(xmin, g_dot[id][k][0]); xmax = floatmax_(xmax, g_dot[id][k][0]);
		ymin = floatmin_(ymin, g_dot[id][k][1]); ymax = floatmax_(ymax, g_dot[id][k][1]);
	}
	new Float:s = BOX / floatmax_(floatmax_(xmax - xmin, ymax - ymin), 3.0);
	new Float:cx = (xmin + xmax) / 2.0, Float:cy = (ymin + ymax) / 2.0;
	for (new k = 0; k < n; k++)
	{
		g_px[k] = (g_dot[id][k][0] - cx) * s + 14.0;
		g_py[k] = (g_dot[id][k][1] - cy) * s + 14.0;
	}
	splat(n, 0.0, 0.0);
	new Float:total = 0.0, Float:mx = 0.0, Float:my = 0.0;
	for (new i = 0; i < PIXELS; i++)
	{
		if (g_img[i] <= 0.0) continue;
		total += g_img[i];
		mx += g_img[i] * (float(i % CELLS) + 0.5);
		my += g_img[i] * (float(i / CELLS) + 0.5);
	}
	total = floatmax_(total, 0.000001);
	splat(n, 14.0 - mx / total, 14.0 - my / total);
}

splat(n, Float:dx, Float:dy)
{
	for (new i = 0; i < PIXELS; i++)
		g_img[i] = 0.0;
	for (new k = 0; k < n; k++)
	{
		new Float:px = g_px[k] + dx, Float:py = g_py[k] + dy;
		new x0 = max(floatround(px - BRUSH, floatround_floor), 0), x1 = min(floatround(px + BRUSH, floatround_floor), CELLS - 1);
		new y0 = max(floatround(py - BRUSH, floatround_floor), 0), y1 = min(floatround(py + BRUSH, floatround_floor), CELLS - 1);
		for (new y = y0; y <= y1; y++)
		{
			for (new x = x0; x <= x1; x++)
			{
				new Float:ex = float(x) + 0.5 - px, Float:ey = float(y) + 0.5 - py;
				new Float:v = BRUSH - floatsqroot(ex * ex + ey * ey);
				if (v <= 0.0) continue;
				if (v > 1.0) v = 1.0;
				if (v > g_img[y * CELLS + x]) g_img[y * CELLS + x] = v;
			}
		}
	}
}

// runs the network on player id's drawing; returns the digit or -1
classify(id)
{
	if (g_model == Invalid_Neural || g_dots[id] < 3)
		return -1;
	render_canvas(id);
	return neural_run(g_model, g_img, g_probs);
}

top3(best[3])
{
	for (new r = 0; r < 3; r++)
	{
		best[r] = -1;
		for (new c = 0; c < CLASSES; c++)
		{
			if ((r > 0 && c == best[0]) || (r > 1 && c == best[1])) continue;
			if (best[r] < 0 || g_probs[c] > g_probs[best[r]]) best[r] = c;
		}
	}
}

// one reading after a burst of shots: the top 3 to the server console and, for a human, to the HUD
public live_reading(task)
{
	new id = task - TASK_READ;
	if (!g_open[id] || !is_user_connected(id) || classify(id) < 0)
		return;
	new best[3], name[32], line[160];
	top3(best);
	get_user_name(id, name, charsmax(name));
	format_top3(best, line, charsmax(line));
	server_print("[MNIST] %s PREDICTED: %s | %d shots", name, line, g_dots[id]);
	if (is_user_bot(id))
		return;
	formatex(g_reading[id], charsmax(g_reading[]), "MNIST  %d (%d%%)   %d (%d%%)   %d (%d%%)^n%d shots", best[0], percent(best[0]),
		best[1], percent(best[1]), best[2], percent(best[2]), g_dots[id]);
	send_reading(id);
}

format_top3(const best[3], line[], len)
{
	formatex(line, len, "%d (%.1f%%), %d (%.1f%%), %d (%.1f%%)", best[0], g_probs[best[0]] * 100.0, best[1],
		g_probs[best[1]] * 100.0, best[2], g_probs[best[2]] * 100.0);
}

// Auto channel, same x/y: the new message replaces the old one with a fresh timer (a fixed channel keeps the first
// send's timer and blinks). The client drops a text equal to the one on screen, so a resend toggles a trailing space.
send_reading(id)
{
	g_hud_flip[id] = !g_hud_flip[id];
	set_hudmessage(80, 255, 120, -1.0, HUD_Y, 0, 0.0, HUD_HOLD, 0.0, 0.3, -1);
	show_hudmessage(id, "%s%s", g_reading[id], g_hud_flip[id] ? " " : "");
	g_resend_at[id] = get_gametime() + HUD_RESEND;
}

// the client ignores an empty message: a blank one at the same x/y replaces the reading
hide_reading(id)
{
	if (!g_reading[id][0] || !is_user_connected(id))
		return;
	g_reading[id][0] = EOS;
	set_hudmessage(80, 255, 120, -1.0, HUD_Y, 0, 0.0, 0.1, 0.0, 0.0, -1);
	show_hudmessage(id, " ");
}

percent(c) { return floatround(g_probs[c] * 100.0); }

// final answer to the chat and the server log; returns the digit or -1
announce(id)
{
	new digit = classify(id);
	if (digit < 0)
	{
		if (is_user_connected(id) && !is_user_bot(id))
			client_print(id, print_chat, "[MNIST] Draw at least 3 shots inside the frame first.");
		server_print("[MNIST] #%d ANSWER: none (%d shots, needs at least 3)", id, g_dots[id]);
		return -1;
	}
	new name[32], best[3], line[160];
	get_user_name(id, name, charsmax(name));
	top3(best);
	format_top3(best, line, charsmax(line));
	client_print(0, print_chat, "[MNIST] %s drew a %d (%d%% sure).", name, digit, percent(digit));
	server_print("[MNIST] %s ANSWER: %d | PREDICTED: %s | %d shots", name, digit, line, g_dots[id]);
	return digit;
}

Float:floatmax_(Float:a, Float:b) { return a > b ? a : b; }
Float:floatmin_(Float:a, Float:b) { return a < b ? a : b; }
