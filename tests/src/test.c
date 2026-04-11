#include "cf_memory.h"
#include "cf_string.h"

#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Minimal test harness                                               */
/* ------------------------------------------------------------------ */

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(label, expr)                                         \
  do {                                                             \
    if (expr) {                                                    \
      printf("  [PASS] %s\n", label);                             \
      g_passed++;                                                  \
    } else {                                                       \
      printf("  [FAIL] %s  (line %d)\n", label, __LINE__);        \
      g_failed++;                                                  \
    }                                                              \
  } while (0)

#define SECTION(name) printf("\n=== %s ===\n", name)

/* ------------------------------------------------------------------ */
/*  cf_bytes tests                                                     */
/* ------------------------------------------------------------------ */

static void test_cf_bytes(void)
{
  SECTION("cf_bytes");

  /* cf_bytes_empty */
  cf_bytes empty = cf_bytes_empty();
  CHECK("bytes_empty: data is null",  empty.data == CF_NULL);
  CHECK("bytes_empty: len is 0",      empty.len  == 0);

  /* cf_bytes_from */
  const cf_u8 raw[] = {1, 2, 3, 4, 5};
  cf_bytes b = cf_bytes_from(raw, 5);
  CHECK("bytes_from: data pointer",   b.data == raw);
  CHECK("bytes_from: len",            b.len  == 5);

  /* cf_bytes_is_valid */
  CHECK("bytes_is_valid: normal",     cf_bytes_is_valid(b)     == CF_TRUE);
  CHECK("bytes_is_valid: empty",      cf_bytes_is_valid(empty) == CF_TRUE);
  cf_bytes bad = {CF_NULL, 3};   /* non-zero len with null ptr */
  CHECK("bytes_is_valid: bad",        cf_bytes_is_valid(bad)   == CF_FALSE);

  /* cf_bytes_is_empty */
  CHECK("bytes_is_empty: empty",      cf_bytes_is_empty(empty) == CF_TRUE);
  CHECK("bytes_is_empty: non-empty",  cf_bytes_is_empty(b)     == CF_FALSE);

  /* cf_bytes_eq */
  const cf_u8 raw2[] = {1, 2, 3, 4, 5};
  const cf_u8 raw3[] = {1, 2, 9, 4, 5};
  cf_bytes b2 = cf_bytes_from(raw2, 5);
  cf_bytes b3 = cf_bytes_from(raw3, 5);
  cf_bool eq;
  CHECK("bytes_eq: equal slices",     cf_bytes_eq(b, b2, &eq) == CF_OK && eq == CF_TRUE);
  CHECK("bytes_eq: unequal slices",   cf_bytes_eq(b, b3, &eq) == CF_OK && eq == CF_FALSE);
  CHECK("bytes_eq: diff lengths",     cf_bytes_eq(b, cf_bytes_from(raw, 3), &eq) == CF_OK && eq == CF_FALSE);
  CHECK("bytes_eq: null out -> ERR_NULL", cf_bytes_eq(b, b2, CF_NULL) == CF_ERR_NULL);

  /* cf_bytes_slice */
  cf_bytes sl;
  CHECK("bytes_slice: ok",            cf_bytes_slice(b, 1, 3, &sl) == CF_OK);
  CHECK("bytes_slice: correct data",  sl.data == raw + 1 && sl.len == 3);
  CHECK("bytes_slice: out of bounds", cf_bytes_slice(b, 3, 4, &sl) == CF_ERR_BOUNDS);
  CHECK("bytes_slice: null dst",      cf_bytes_slice(b, 0, 1, CF_NULL) == CF_ERR_NULL);
  CHECK("bytes_slice: zero-len ok",   cf_bytes_slice(b, 5, 0, &sl) == CF_OK && sl.len == 0);
}

/* ------------------------------------------------------------------ */
/*  cf_bytes_mut tests                                                 */
/* ------------------------------------------------------------------ */

static void test_cf_bytes_mut(void)
{
  SECTION("cf_bytes_mut");

  cf_bytes_mut empty = cf_bytes_mut_empty();
  CHECK("bytes_mut_empty: null data", empty.data == CF_NULL);
  CHECK("bytes_mut_empty: len 0",     empty.len  == 0);

  cf_u8 buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  cf_bytes_mut bm = cf_bytes_mut_from(buf, 8);
  CHECK("bytes_mut_from: data ptr",   bm.data == buf);
  CHECK("bytes_mut_from: len",        bm.len  == 8);

  CHECK("bytes_mut_is_valid: normal", cf_bytes_mut_is_valid(bm)    == CF_TRUE);
  CHECK("bytes_mut_is_valid: empty",  cf_bytes_mut_is_valid(empty) == CF_TRUE);
  cf_bytes_mut bad = {CF_NULL, 3};
  CHECK("bytes_mut_is_valid: bad",    cf_bytes_mut_is_valid(bad)   == CF_FALSE);

  CHECK("bytes_mut_is_empty: empty",     cf_bytes_mut_is_empty(empty) == CF_TRUE);
  CHECK("bytes_mut_is_empty: non-empty", cf_bytes_mut_is_empty(bm)    == CF_FALSE);

  /* cf_bytes_mut_zero */
  CHECK("bytes_mut_zero: ok",         cf_bytes_mut_zero(bm) == CF_OK);
  int all_zero = 1;
  for (int i = 0; i < 8; i++) if (buf[i] != 0) { all_zero = 0; break; }
  CHECK("bytes_mut_zero: all zeroed", all_zero);

  CHECK("bytes_mut_zero: empty ok",   cf_bytes_mut_zero(empty) == CF_OK);
  CHECK("bytes_mut_zero: bad ERR_STATE", cf_bytes_mut_zero(bad) == CF_ERR_STATE);
}

/* ------------------------------------------------------------------ */
/*  cf_buffer tests                                                    */
/* ------------------------------------------------------------------ */

static void test_cf_buffer(void)
{
  SECTION("cf_buffer");

  /* cf_buffer_empty */
  cf_buffer empty = cf_buffer_empty();
  CHECK("buffer_empty: null data",  empty.data == CF_NULL);
  CHECK("buffer_empty: len 0",      empty.len  == 0);
  CHECK("buffer_empty: cap 0",      empty.cap  == 0);

  /* cf_buffer_is_valid */
  CHECK("buffer_is_valid: empty",   cf_buffer_is_valid(empty) == CF_TRUE);
  cf_buffer bad = {CF_NULL, 1, 4};
  CHECK("buffer_is_valid: bad",     cf_buffer_is_valid(bad)   == CF_FALSE);

  /* cf_buffer_is_empty */
  CHECK("buffer_is_empty: empty",   cf_buffer_is_empty(empty) == CF_TRUE);

  /* cf_buffer_init */
  cf_buffer buf;
  CHECK("buffer_init: ok",          cf_buffer_init(&buf, 16) == CF_OK);
  CHECK("buffer_init: cap >= 16",   buf.cap >= 16);
  CHECK("buffer_init: len == 0",    buf.len == 0);
  CHECK("buffer_init: data != null",buf.data != CF_NULL);
  CHECK("buffer_init: null -> ERR_NULL", cf_buffer_init(CF_NULL, 16) == CF_ERR_NULL);

  /* cf_buffer_reserve */
  CHECK("buffer_reserve: already enough", cf_buffer_reserve(&buf, 8) == CF_OK);
  CHECK("buffer_reserve: grow",           cf_buffer_reserve(&buf, 64) == CF_OK && buf.cap >= 64);
  CHECK("buffer_reserve: null -> ERR_NULL", cf_buffer_reserve(CF_NULL, 8) == CF_ERR_NULL);

  /* cf_buffer_append_byte */
  CHECK("buffer_append_byte: ok",   cf_buffer_append_byte(&buf, 0xAB) == CF_OK);
  CHECK("buffer_append_byte: len",  buf.len == 1);
  CHECK("buffer_append_byte: data", buf.data[0] == 0xAB);
  CHECK("buffer_append_byte: null -> ERR_NULL", cf_buffer_append_byte(CF_NULL, 0) == CF_ERR_NULL);

  /* cf_buffer_append_bytes */
  const cf_u8 extra[] = {0x01, 0x02, 0x03};
  cf_bytes src = cf_bytes_from(extra, 3);
  CHECK("buffer_append_bytes: ok",  cf_buffer_append_bytes(&buf, src) == CF_OK);
  CHECK("buffer_append_bytes: len", buf.len == 4);
  CHECK("buffer_append_bytes: data[1]", buf.data[1] == 0x01);
  CHECK("buffer_append_bytes: null -> ERR_NULL", cf_buffer_append_bytes(CF_NULL, src) == CF_ERR_NULL);

  /* grow past initial capacity */
  cf_buffer_clear(&buf);
  for (int i = 0; i < 200; i++)
    CHECK("buffer auto-grow", cf_buffer_append_byte(&buf, (cf_u8)i) == CF_OK);
  CHECK("buffer auto-grow: len 200", buf.len == 200);

  /* cf_buffer_clear */
  CHECK("buffer_clear: ok",         cf_buffer_clear(&buf) == CF_OK);
  CHECK("buffer_clear: len 0",      buf.len == 0);
  CHECK("buffer_clear: cap kept",   buf.cap >= 200);
  CHECK("buffer_clear: null -> ERR_NULL", cf_buffer_clear(CF_NULL) == CF_ERR_NULL);

  /* cf_buffer_destroy */
  cf_buffer_destroy(&buf);
  CHECK("buffer_destroy: null data", buf.data == CF_NULL);
  CHECK("buffer_destroy: len 0",     buf.len  == 0);
  CHECK("buffer_destroy: cap 0",     buf.cap  == 0);
  cf_buffer_destroy(CF_NULL); /* must not crash */
  CHECK("buffer_destroy: null safe", CF_TRUE);
}

/* ------------------------------------------------------------------ */
/*  cf_str tests                                                       */
/* ------------------------------------------------------------------ */

static void test_cf_str(void)
{
  SECTION("cf_str");

  /* cf_str_empty */
  cf_str empty = cf_str_empty();
  CHECK("str_empty: null data", empty.data == CF_NULL);
  CHECK("str_empty: len 0",     empty.len  == 0);

  /* cf_str_from */
  const char *hello = "hello";
  cf_str s = cf_str_from(hello, 5);
  CHECK("str_from: data ptr", s.data == hello);
  CHECK("str_from: len",      s.len  == 5);

  /* cf_str_is_valid */
  CHECK("str_is_valid: normal", cf_str_is_valid(s)     == CF_TRUE);
  CHECK("str_is_valid: empty",  cf_str_is_valid(empty) == CF_TRUE);
  cf_str bad = {CF_NULL, 3};
  CHECK("str_is_valid: bad",    cf_str_is_valid(bad)   == CF_FALSE);

  /* cf_str_is_empty */
  CHECK("str_is_empty: empty",     cf_str_is_empty(empty) == CF_TRUE);
  CHECK("str_is_empty: non-empty", cf_str_is_empty(s)     == CF_FALSE);

  /* cf_str_eq */
  cf_str s2 = cf_str_from("hello", 5);
  cf_str s3 = cf_str_from("world", 5);
  cf_bool eq;
  CHECK("str_eq: equal",          cf_str_eq(s, s2, &eq) == CF_OK && eq == CF_TRUE);
  CHECK("str_eq: unequal",        cf_str_eq(s, s3, &eq) == CF_OK && eq == CF_FALSE);
  CHECK("str_eq: diff lengths",   cf_str_eq(s, cf_str_from("hi", 2), &eq) == CF_OK && eq == CF_FALSE);
  CHECK("str_eq: null out",       cf_str_eq(s, s2, CF_NULL) == CF_ERR_NULL);

  /* cf_str_slice */
  cf_str sl;
  CHECK("str_slice: ok",          cf_str_slice(s, 1, 3, &sl) == CF_OK);
  CHECK("str_slice: data",        sl.data == hello + 1 && sl.len == 3);
  CHECK("str_slice: out of bounds", cf_str_slice(s, 3, 4, &sl) == CF_ERR_BOUNDS);
  CHECK("str_slice: null dst",    cf_str_slice(s, 0, 1, CF_NULL) == CF_ERR_NULL);
  CHECK("str_slice: zero-len ok", cf_str_slice(s, 5, 0, &sl) == CF_OK && sl.len == 0);
}

/* ------------------------------------------------------------------ */
/*  cf_string tests                                                    */
/* ------------------------------------------------------------------ */

static void test_cf_string(void)
{
  SECTION("cf_string");

  /* cf_string_empty */
  cf_string empty = cf_string_empty();
  CHECK("string_empty: null data", empty.data == CF_NULL);
  CHECK("string_empty: len 0",     empty.len  == 0);
  CHECK("string_empty: cap 0",     empty.cap  == 0);

  /* cf_string_is_valid */
  CHECK("string_is_valid: empty",  cf_string_is_valid(empty) == CF_TRUE);

  /* cf_string_is_empty */
  CHECK("string_is_empty: empty",  cf_string_is_empty(empty) == CF_TRUE);

  /* cf_string_init */
  cf_string str;
  CHECK("string_init: ok",          cf_string_init(&str, 16) == CF_OK);
  CHECK("string_init: cap >= 16",   str.cap >= 16);
  CHECK("string_init: len 0",       str.len == 0);
  CHECK("string_init: null term",   str.data != CF_NULL && str.data[0] == '\0');
  CHECK("string_init: null -> ERR_NULL", cf_string_init(CF_NULL, 16) == CF_ERR_NULL);

  /* cf_string_reserve */
  CHECK("string_reserve: already enough", cf_string_reserve(&str, 8) == CF_OK);
  CHECK("string_reserve: grow",           cf_string_reserve(&str, 64) == CF_OK && str.cap >= 64);
  CHECK("string_reserve: null -> ERR_NULL", cf_string_reserve(CF_NULL, 8) == CF_ERR_NULL);

  /* cf_string_append_char */
  CHECK("string_append_char: 'h'",  cf_string_append_char(&str, 'h') == CF_OK);
  CHECK("string_append_char: 'i'",  cf_string_append_char(&str, 'i') == CF_OK);
  CHECK("string_append_char: len",  str.len == 2);
  CHECK("string_append_char: content", str.data[0] == 'h' && str.data[1] == 'i');
  CHECK("string_append_char: null term", str.data[2] == '\0');
  CHECK("string_append_char: null -> ERR_NULL", cf_string_append_char(CF_NULL, 'x') == CF_ERR_NULL);

  /* cf_string_append_str */
  cf_str suffix = cf_str_from(" world", 6);
  CHECK("string_append_str: ok",     cf_string_append_str(&str, suffix) == CF_OK);
  CHECK("string_append_str: len",    str.len == 8);
  CHECK("string_append_str: content", memcmp(str.data, "hi world", 8) == 0);
  CHECK("string_append_str: null term", str.data[8] == '\0');
  CHECK("string_append_str: null -> ERR_NULL", cf_string_append_str(CF_NULL, suffix) == CF_ERR_NULL);

  /* empty cf_str appended — must be a no-op */
  cf_usize len_before = str.len;
  CHECK("string_append_str: empty no-op", cf_string_append_str(&str, cf_str_empty()) == CF_OK);
  CHECK("string_append_str: len unchanged", str.len == len_before);

  /* grow past initial capacity */
  cf_string_clear(&str);
  for (int i = 0; i < 200; i++)
    CHECK("string auto-grow", cf_string_append_char(&str, 'x') == CF_OK);
  CHECK("string auto-grow: len 200",  str.len == 200);
  CHECK("string auto-grow: null term", str.data[200] == '\0');

  /* cf_string_clear */
  CHECK("string_clear: ok",          cf_string_clear(&str) == CF_OK);
  CHECK("string_clear: len 0",       str.len == 0);
  CHECK("string_clear: null term",   str.data[0] == '\0');
  CHECK("string_clear: cap kept",    str.cap >= 200);
  CHECK("string_clear: null -> ERR_NULL", cf_string_clear(CF_NULL) == CF_ERR_NULL);

  /* cf_string_is_valid after clear */
  CHECK("string_is_valid: after clear", cf_string_is_valid(str) == CF_TRUE);

  /* cf_string_is_empty after clear */
  CHECK("string_is_empty: after clear", cf_string_is_empty(str) == CF_TRUE);

  /* cf_string_destroy */
  cf_string_destroy(&str);
  CHECK("string_destroy: null data", str.data == CF_NULL);
  CHECK("string_destroy: len 0",     str.len  == 0);
  CHECK("string_destroy: cap 0",     str.cap  == 0);
  cf_string_destroy(CF_NULL); /* must not crash */
  CHECK("string_destroy: null safe", CF_TRUE);
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
  printf("Running cf tests...\n");

  test_cf_bytes();
  test_cf_bytes_mut();
  test_cf_buffer();
  test_cf_str();
  test_cf_string();

  printf("\n==============================\n");
  printf("Results: %d passed, %d failed\n", g_passed, g_failed);
  printf("==============================\n");

  return g_failed == 0 ? 0 : 1;
}