// neural-amx: minimal JSON DOM parser (RFC 8259 subset sufficient for safetensors headers and model graphs).
#ifndef NAM_JSON_H
#define NAM_JSON_H

#include <stddef.h>

typedef enum { NJ_NULL, NJ_BOOL, NJ_NUM, NJ_STR, NJ_ARR, NJ_OBJ } nj_type;

typedef struct nj {
  nj_type type;
  double num;       // NJ_NUM, NJ_BOOL (0/1)
  char *str;        // NJ_STR (UTF-8, NUL-terminated)
  char *key;        // member name when the parent is an object
  int count;        // children of NJ_ARR / NJ_OBJ
  struct nj *child; // first child
  struct nj *next;  // next sibling
} nj;

// Parses `len` bytes; returns NULL and fills `err` on failure. Free with nj_free.
nj *nj_parse(const char *text, size_t len, char *err, size_t errlen);
void nj_free(nj *node);
nj *nj_get(const nj *object, const char *key); // NULL when absent or not an object
nj *nj_at(const nj *array, int index);
// typed getters with defaults
double nj_num(const nj *object, const char *key, double fallback);
const char *nj_str(const nj *object, const char *key, const char *fallback);

#endif // NAM_JSON_H
