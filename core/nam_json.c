// neural-amx: minimal JSON DOM parser.
#include "nam_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const char *p, *end;
  char *err;
  size_t errlen;
  int depth;
} parser;

static int fail(parser *ps, const char *msg) {
  if (ps->err && ps->errlen && !ps->err[0]) snprintf(ps->err, ps->errlen, "json: %s", msg);
  return 0;
}

static void skip_ws(parser *ps) {
  while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')) ps->p++;
}

static int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static int put_utf8(char *o, unsigned cp) {
  if (cp < 0x80) { o[0] = (char)cp; return 1; }
  if (cp < 0x800) { o[0] = (char)(0xC0 | cp >> 6); o[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
  if (cp < 0x10000) {
    o[0] = (char)(0xE0 | cp >> 12); o[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
  }
  o[0] = (char)(0xF0 | cp >> 18); o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
  o[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (char)(0x80 | (cp & 0x3F));
  return 4;
}

static int read_hex4(parser *ps, unsigned *cp) {
  if (ps->end - ps->p < 4) return fail(ps, "short \\u escape");
  *cp = 0;
  for (int i = 0; i < 4; i++) {
    int h = hexval(ps->p[i]);
    if (h < 0) return fail(ps, "bad \\u escape");
    *cp = *cp << 4 | (unsigned)h;
  }
  ps->p += 4;
  return 1;
}

// parses a string at ps->p (after the opening quote check) into a new buffer
static char *parse_string(parser *ps) {
  if (ps->p >= ps->end || *ps->p != '"') { fail(ps, "expected string"); return NULL; }
  ps->p++;
  const char *start = ps->p;
  size_t cap = 0;
  while (ps->p < ps->end && *ps->p != '"') { if (*ps->p == '\\') ps->p++; ps->p++; cap++; }
  if (ps->p >= ps->end) { fail(ps, "unterminated string"); return NULL; }
  char *out = malloc(cap * 4 + 1), *o = out;
  if (!out) { fail(ps, "out of memory"); return NULL; }
  const char *q = start;
  while (q < ps->p) {
    char c = *q++;
    if (c != '\\') { *o++ = c; continue; }
    char e = *q++;
    switch (e) {
      case '"': *o++ = '"'; break;
      case '\\': *o++ = '\\'; break;
      case '/': *o++ = '/'; break;
      case 'b': *o++ = '\b'; break;
      case 'f': *o++ = '\f'; break;
      case 'n': *o++ = '\n'; break;
      case 'r': *o++ = '\r'; break;
      case 't': *o++ = '\t'; break;
      case 'u': {
        parser sub = {q, ps->p, ps->err, ps->errlen, 0};
        unsigned cp;
        if (!read_hex4(&sub, &cp)) { free(out); return NULL; }
        q = sub.p;
        if (cp >= 0xD800 && cp < 0xDC00 && ps->p - q >= 6 && q[0] == '\\' && q[1] == 'u') {
          unsigned lo;
          sub.p = q + 2;
          if (!read_hex4(&sub, &lo)) { free(out); return NULL; }
          if (lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); q = sub.p; }
        }
        o += put_utf8(o, cp);
        break;
      }
      default: free(out); fail(ps, "bad escape"); return NULL;
    }
  }
  *o = 0;
  ps->p++; // closing quote
  return out;
}

static nj *parse_value(parser *ps);

static nj *new_node(parser *ps, nj_type t) {
  nj *n = calloc(1, sizeof *n);
  if (!n) fail(ps, "out of memory");
  else n->type = t;
  return n;
}

static nj *parse_container(parser *ps, int object) {
  nj *node = new_node(ps, object ? NJ_OBJ : NJ_ARR), **tail;
  if (!node) return NULL;
  tail = &node->child;
  ps->p++;
  skip_ws(ps);
  if (ps->p < ps->end && *ps->p == (object ? '}' : ']')) { ps->p++; return node; }
  for (;;) {
    char *key = NULL;
    skip_ws(ps);
    if (object) {
      if (!(key = parse_string(ps))) break;
      skip_ws(ps);
      if (ps->p >= ps->end || *ps->p != ':') { free(key); fail(ps, "expected ':'"); break; }
      ps->p++;
    }
    nj *child = parse_value(ps);
    if (!child) { free(key); break; }
    child->key = key;
    *tail = child;
    tail = &child->next;
    node->count++;
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
    if (ps->p < ps->end && *ps->p == (object ? '}' : ']')) { ps->p++; return node; }
    fail(ps, object ? "expected ',' or '}'" : "expected ',' or ']'");
    break;
  }
  nj_free(node);
  return NULL;
}

static nj *parse_value(parser *ps) {
  skip_ws(ps);
  if (ps->p >= ps->end) { fail(ps, "unexpected end"); return NULL; }
  if (++ps->depth > 64) { fail(ps, "nesting too deep"); return NULL; }
  nj *n = NULL;
  char c = *ps->p;
  if (c == '{' || c == '[') {
    n = parse_container(ps, c == '{');
  } else if (c == '"') {
    char *s = parse_string(ps);
    if (s && (n = new_node(ps, NJ_STR))) n->str = s;
    else free(s);
  } else if (ps->end - ps->p >= 4 && !memcmp(ps->p, "true", 4)) {
    if ((n = new_node(ps, NJ_BOOL))) n->num = 1;
    ps->p += 4;
  } else if (ps->end - ps->p >= 5 && !memcmp(ps->p, "false", 5)) {
    n = new_node(ps, NJ_BOOL);
    ps->p += 5;
  } else if (ps->end - ps->p >= 4 && !memcmp(ps->p, "null", 4)) {
    n = new_node(ps, NJ_NULL);
    ps->p += 4;
  } else if (c == '-' || (c >= '0' && c <= '9')) {
    char buf[64];
    size_t k = 0;
    while (ps->p < ps->end && k < sizeof buf - 1 && strchr("+-0123456789.eE", *ps->p)) buf[k++] = *ps->p++;
    buf[k] = 0;
    char *endp;
    double v = strtod(buf, &endp);
    if (endp != buf + k) fail(ps, "bad number");
    else if ((n = new_node(ps, NJ_NUM))) n->num = v;
  } else {
    fail(ps, "unexpected character");
  }
  ps->depth--;
  return n;
}

nj *nj_parse(const char *text, size_t len, char *err, size_t errlen) {
  if (err && errlen) err[0] = 0;
  parser ps = {text, text + len, err, errlen, 0};
  nj *root = parse_value(&ps);
  if (!root) return NULL;
  skip_ws(&ps);
  if (ps.p != ps.end) { nj_free(root); fail(&ps, "trailing characters"); return NULL; }
  return root;
}

void nj_free(nj *node) {
  while (node) {
    nj *next = node->next;
    nj_free(node->child);
    free(node->str);
    free(node->key);
    free(node);
    node = next;
  }
}

nj *nj_get(const nj *object, const char *key) {
  if (!object || object->type != NJ_OBJ) return NULL;
  for (nj *c = object->child; c; c = c->next)
    if (c->key && !strcmp(c->key, key)) return c;
  return NULL;
}

nj *nj_at(const nj *array, int index) {
  if (!array || (array->type != NJ_ARR && array->type != NJ_OBJ) || index < 0) return NULL;
  nj *c = array->child;
  while (c && index--) c = c->next;
  return c;
}

double nj_num(const nj *object, const char *key, double fallback) {
  nj *v = nj_get(object, key);
  return v && (v->type == NJ_NUM || v->type == NJ_BOOL) ? v->num : fallback;
}

const char *nj_str(const nj *object, const char *key, const char *fallback) {
  nj *v = nj_get(object, key);
  return v && v->type == NJ_STR ? v->str : fallback;
}
