#include "cf_memory.h"
#include "cf_string.h"
#include "cf_status.h"

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

  cf_bytes empty = cf_bytes_empty();
  CHECK("bytes_empty: data is null",  empty.data == CF_NULL);
  CHECK("bytes_empty: len is 0",      empty.len  == 0);

  const cf_u8 raw[] = {1, 2, 3, 4, 5};
  cf_bytes b = cf_bytes_from(raw, 5);
  CHECK("bytes_from: data pointer",   b.data == raw);
  CHECK("bytes_from: len",            b.len  == 5);

  CHECK("bytes_is_valid: normal",     cf_bytes_is_valid(b)     == CF_TRUE);
  CHECK("bytes_is_valid: empty",      cf_bytes_is_valid(empty) == CF_TRUE);
  cf_bytes bad = {CF_NULL, 3};
  CHECK("bytes_is_valid: bad",        cf_bytes_is_valid(bad)   == CF_FALSE);

  CHECK("bytes_is_empty: empty",      cf_bytes_is_empty(empty) == CF_TRUE);
  CHECK("bytes_is_empty: non-empty",  cf_bytes_is_empty(b)     == CF_FALSE);

  const cf_u8 raw2[] = {1, 2, 3, 4, 5};
  const cf_u8 raw3[] = {1, 2, 9, 4, 5};
  cf_bytes b2 = cf_bytes_from(raw2, 5);
  cf_bytes b3 = cf_bytes_from(raw3, 5);
  cf_bool eq;
  CHECK("bytes_eq: equal slices",     cf_bytes_eq(b, b2, &eq) == CF_OK && eq == CF_TRUE);
  CHECK("bytes_eq: unequal slices",   cf_bytes_eq(b, b3, &eq) == CF_OK && eq == CF_FALSE);
  CHECK("bytes_eq: diff lengths",     cf_bytes_eq(b, cf_bytes_from(raw, 3), &eq) == CF_OK && eq == CF_FALSE);
  CHECK("bytes_eq: null out -> ERR_NULL", cf_bytes_eq(b, b2, CF_NULL) == CF_ERR_NULL);

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

  cf_buffer empty = cf_buffer_empty();
  CHECK("buffer_empty: null data",  empty.data == CF_NULL);
  CHECK("buffer_empty: len 0",      empty.len  == 0);
  CHECK("buffer_empty: cap 0",      empty.cap  == 0);

  CHECK("buffer_is_valid: empty",   cf_buffer_is_valid(empty) == CF_TRUE);
  cf_buffer bad = {CF_NULL, 1, 4};
  CHECK("buffer_is_valid: bad",     cf_buffer_is_valid(bad)   == CF_FALSE);

  CHECK("buffer_is_empty: empty",   cf_buffer_is_empty(empty) == CF_TRUE);

  cf_buffer buf;
  CHECK("buffer_init: ok",          cf_buffer_init(&buf, 16) == CF_OK);
  CHECK("buffer_init: cap >= 16",   buf.cap >= 16);
  CHECK("buffer_init: len == 0",    buf.len == 0);
  CHECK("buffer_init: data != null",buf.data != CF_NULL);
  CHECK("buffer_init: null -> ERR_NULL", cf_buffer_init(CF_NULL, 16) == CF_ERR_NULL);

  CHECK("buffer_reserve: already enough", cf_buffer_reserve(&buf, 8) == CF_OK);
  CHECK("buffer_reserve: grow",           cf_buffer_reserve(&buf, 64) == CF_OK && buf.cap >= 64);
  CHECK("buffer_reserve: null -> ERR_NULL", cf_buffer_reserve(CF_NULL, 8) == CF_ERR_NULL);

  CHECK("buffer_append_byte: ok",   cf_buffer_append_byte(&buf, 0xAB) == CF_OK);
  CHECK("buffer_append_byte: len",  buf.len == 1);
  CHECK("buffer_append_byte: data", buf.data[0] == 0xAB);
  CHECK("buffer_append_byte: null -> ERR_NULL", cf_buffer_append_byte(CF_NULL, 0) == CF_ERR_NULL);

  const cf_u8 extra[] = {0x01, 0x02, 0x03};
  cf_bytes src = cf_bytes_from(extra, 3);
  CHECK("buffer_append_bytes: ok",  cf_buffer_append_bytes(&buf, src) == CF_OK);
  CHECK("buffer_append_bytes: len", buf.len == 4);
  CHECK("buffer_append_bytes: data[1]", buf.data[1] == 0x01);
  CHECK("buffer_append_bytes: null -> ERR_NULL", cf_buffer_append_bytes(CF_NULL, src) == CF_ERR_NULL);

  cf_buffer_clear(&buf);
  for (int i = 0; i < 200; i++)
    CHECK("buffer auto-grow", cf_buffer_append_byte(&buf, (cf_u8)i) == CF_OK);
  CHECK("buffer auto-grow: len 200", buf.len == 200);

  CHECK("buffer_clear: ok",         cf_buffer_clear(&buf) == CF_OK);
  CHECK("buffer_clear: len 0",      buf.len == 0);
  CHECK("buffer_clear: cap kept",   buf.cap >= 200);
  CHECK("buffer_clear: null -> ERR_NULL", cf_buffer_clear(CF_NULL) == CF_ERR_NULL);

  cf_buffer_destroy(&buf);
  CHECK("buffer_destroy: null data", buf.data == CF_NULL);
  CHECK("buffer_destroy: len 0",     buf.len  == 0);
  CHECK("buffer_destroy: cap 0",     buf.cap  == 0);
  cf_buffer_destroy(CF_NULL);
  CHECK("buffer_destroy: null safe", CF_TRUE);
}

/* ------------------------------------------------------------------ */
/*  cf_str tests                                                       */
/* ------------------------------------------------------------------ */

static void test_cf_str(void)
{
  SECTION("cf_str");

  cf_str empty = cf_str_empty();
  CHECK("str_empty: null data", empty.data == CF_NULL);
  CHECK("str_empty: len 0",     empty.len  == 0);

  const char *hello = "hello";
  cf_str s = cf_str_from(hello, 5);
  CHECK("str_from: data ptr", s.data == hello);
  CHECK("str_from: len",      s.len  == 5);

  CHECK("str_is_valid: normal", cf_str_is_valid(s)     == CF_TRUE);
  CHECK("str_is_valid: empty",  cf_str_is_valid(empty) == CF_TRUE);
  cf_str bad = {CF_NULL, 3};
  CHECK("str_is_valid: bad",    cf_str_is_valid(bad)   == CF_FALSE);

  CHECK("str_is_empty: empty",     cf_str_is_empty(empty) == CF_TRUE);
  CHECK("str_is_empty: non-empty", cf_str_is_empty(s)     == CF_FALSE);

  cf_str s2 = cf_str_from("hello", 5);
  cf_str s3 = cf_str_from("world", 5);
  cf_bool eq;
  CHECK("str_eq: equal",          cf_str_eq(s, s2, &eq) == CF_OK && eq == CF_TRUE);
  CHECK("str_eq: unequal",        cf_str_eq(s, s3, &eq) == CF_OK && eq == CF_FALSE);
  CHECK("str_eq: diff lengths",   cf_str_eq(s, cf_str_from("hi", 2), &eq) == CF_OK && eq == CF_FALSE);
  CHECK("str_eq: null out",       cf_str_eq(s, s2, CF_NULL) == CF_ERR_NULL);

  cf_str sl;
  CHECK("str_slice: ok",          cf_str_slice(s, 1, 3, &sl) == CF_OK);
  CHECK("str_slice: data",        sl.data == hello + 1 && sl.len == 3);
  CHECK("str_slice: out of bounds", cf_str_slice(s, 3, 4, &sl) == CF_ERR_BOUNDS);
  CHECK("str_slice: null dst",    cf_str_slice(s, 0, 1, CF_NULL) == CF_ERR_NULL);
  CHECK("str_slice: zero-len ok", cf_str_slice(s, 5, 0, &sl) == CF_OK && sl.len == 0);
}

static void test_cf_buffer_set_and_views(void)
{
  SECTION("cf_buffer set/view");

  cf_buffer buf;
  CHECK("buffer_set/view: init", cf_buffer_init(&buf, 4) == CF_OK);

  const cf_u8 first_raw[] = {10, 20, 30};
  const cf_u8 second_raw[] = {7, 8};
  cf_bytes first  = cf_bytes_from(first_raw,  3);
  cf_bytes second = cf_bytes_from(second_raw, 2);

  CHECK("buffer_set_bytes: first set ok",   cf_buffer_set_bytes(&buf, first) == CF_OK);
  CHECK("buffer_set_bytes: len == 3",       buf.len == 3);
  CHECK("buffer_set_bytes: content first",  buf.data[0] == 10 && buf.data[1] == 20 && buf.data[2] == 30);
  CHECK("buffer_set_bytes: replace ok",     cf_buffer_set_bytes(&buf, second) == CF_OK);
  CHECK("buffer_set_bytes: len == 2",       buf.len == 2);
  CHECK("buffer_set_bytes: content replaced", buf.data[0] == 7 && buf.data[1] == 8);
  CHECK("buffer_set_bytes: set empty ok",   cf_buffer_set_bytes(&buf, cf_bytes_empty()) == CF_OK);
  CHECK("buffer_set_bytes: set empty len",  buf.len == 0);
  CHECK("buffer_set_bytes: null -> ERR_NULL",   cf_buffer_set_bytes(CF_NULL, first) == CF_ERR_NULL);
  CHECK("buffer_set_bytes: bad -> ERR_STATE",   cf_buffer_set_bytes(&buf, (cf_bytes){CF_NULL, 2}) == CF_ERR_STATE);

  CHECK("buffer_set_bytes: restore", cf_buffer_set_bytes(&buf, first) == CF_OK);
  cf_bytes view = cf_buffer_as_bytes(buf);
  CHECK("buffer_as_bytes: len",  view.len  == buf.len);
  CHECK("buffer_as_bytes: ptr",  view.data == buf.data);

  cf_bytes_mut mview = cf_buffer_as_bytes_mut(&buf);
  CHECK("buffer_as_bytes_mut: len", mview.len  == buf.len);
  CHECK("buffer_as_bytes_mut: ptr", mview.data == buf.data);
  if (mview.len > 0) mview.data[0] = 99;
  CHECK("buffer_as_bytes_mut: mutation visible", buf.data[0] == 99);

  mview = cf_buffer_as_bytes_mut(CF_NULL);
  CHECK("buffer_as_bytes_mut: null -> empty", mview.data == CF_NULL && mview.len == 0);

  cf_buffer_destroy(&buf);
}

static void test_cf_buffer_fill_and_truncate(void)
{
  SECTION("cf_buffer fill/truncate");

  cf_u8 raw[5] = {1, 2, 3, 4, 5};
  cf_bytes_mut bm = cf_bytes_mut_from(raw, 5);

  CHECK("bytes_mut_fill: ok",         cf_bytes_mut_fill(bm, 0xAA) == CF_OK);
  CHECK("bytes_mut_fill: all filled",
        raw[0] == 0xAA && raw[1] == 0xAA && raw[2] == 0xAA &&
        raw[3] == 0xAA && raw[4] == 0xAA);
  CHECK("bytes_mut_fill: empty ok",   cf_bytes_mut_fill(cf_bytes_mut_empty(), 0x11) == CF_OK);
  CHECK("bytes_mut_fill: bad -> ERR_STATE",
        cf_bytes_mut_fill((cf_bytes_mut){CF_NULL, 3}, 0x22) == CF_ERR_STATE);

  cf_buffer buf;
  CHECK("buffer_fill/truncate: init",      cf_buffer_init(&buf, 8) == CF_OK);
  CHECK("buffer_fill/truncate: set bytes",
        cf_buffer_set_bytes(&buf, cf_bytes_from((const cf_u8 *)"abcd", 4)) == CF_OK);
  CHECK("buffer_fill: ok",                 cf_buffer_fill(&buf, 0x55) == CF_OK);
  CHECK("buffer_fill: filled",
        buf.data[0] == 0x55 && buf.data[1] == 0x55 &&
        buf.data[2] == 0x55 && buf.data[3] == 0x55);
  CHECK("buffer_fill: null -> ERR_NULL",   cf_buffer_fill(CF_NULL, 0x33) == CF_ERR_NULL);
  CHECK("buffer_truncate: shrink to 2",    cf_buffer_truncate(&buf, 2) == CF_OK);
  CHECK("buffer_truncate: len == 2",       buf.len == 2);
  CHECK("buffer_truncate: to zero",        cf_buffer_truncate(&buf, 0) == CF_OK);
  CHECK("buffer_truncate: len == 0",       buf.len == 0);
  CHECK("buffer_truncate: too large -> ERR_BOUNDS", cf_buffer_truncate(&buf, 1) == CF_ERR_BOUNDS);
  CHECK("buffer_truncate: null -> ERR_NULL",        cf_buffer_truncate(CF_NULL, 0) == CF_ERR_NULL);

  cf_buffer_destroy(&buf);
}

/* ------------------------------------------------------------------ */
/*  cf_string tests                                                    */
/* ------------------------------------------------------------------ */

static void test_cf_string(void)
{
  SECTION("cf_string");

  cf_string empty = cf_string_empty();
  CHECK("string_empty: null data", empty.data == CF_NULL);
  CHECK("string_empty: len 0",     empty.len  == 0);
  CHECK("string_empty: cap 0",     empty.cap  == 0);
  CHECK("string_is_valid: empty",  cf_string_is_valid(empty) == CF_TRUE);
  CHECK("string_is_empty: empty",  cf_string_is_empty(empty) == CF_TRUE);

  cf_string str;
  CHECK("string_init: ok",          cf_string_init(&str, 16) == CF_OK);
  CHECK("string_init: cap >= 16",   str.cap >= 16);
  CHECK("string_init: len 0",       str.len == 0);
  CHECK("string_init: null term",   str.data != CF_NULL && str.data[0] == '\0');
  CHECK("string_init: null -> ERR_NULL", cf_string_init(CF_NULL, 16) == CF_ERR_NULL);

  CHECK("string_reserve: already enough", cf_string_reserve(&str, 8) == CF_OK);
  CHECK("string_reserve: grow",           cf_string_reserve(&str, 64) == CF_OK && str.cap >= 64);
  CHECK("string_reserve: null -> ERR_NULL", cf_string_reserve(CF_NULL, 8) == CF_ERR_NULL);

  CHECK("string_append_char: 'h'",  cf_string_append_char(&str, 'h') == CF_OK);
  CHECK("string_append_char: 'i'",  cf_string_append_char(&str, 'i') == CF_OK);
  CHECK("string_append_char: len",  str.len == 2);
  CHECK("string_append_char: content", str.data[0] == 'h' && str.data[1] == 'i');
  CHECK("string_append_char: null term", str.data[2] == '\0');
  CHECK("string_append_char: null -> ERR_NULL", cf_string_append_char(CF_NULL, 'x') == CF_ERR_NULL);

  cf_str suffix = cf_str_from(" world", 6);
  CHECK("string_append_str: ok",     cf_string_append_str(&str, suffix) == CF_OK);
  CHECK("string_append_str: len",    str.len == 8);
  CHECK("string_append_str: content", memcmp(str.data, "hi world", 8) == 0);
  CHECK("string_append_str: null term", str.data[8] == '\0');
  CHECK("string_append_str: null -> ERR_NULL", cf_string_append_str(CF_NULL, suffix) == CF_ERR_NULL);

  cf_usize len_before = str.len;
  CHECK("string_append_str: empty no-op", cf_string_append_str(&str, cf_str_empty()) == CF_OK);
  CHECK("string_append_str: len unchanged", str.len == len_before);

  cf_string_clear(&str);
  for (int i = 0; i < 200; i++)
    CHECK("string auto-grow", cf_string_append_char(&str, 'x') == CF_OK);
  CHECK("string auto-grow: len 200",   str.len == 200);
  CHECK("string auto-grow: null term", str.data[200] == '\0');

  CHECK("string_clear: ok",        cf_string_clear(&str) == CF_OK);
  CHECK("string_clear: len 0",     str.len == 0);
  CHECK("string_clear: null term", str.data[0] == '\0');
  CHECK("string_clear: cap kept",  str.cap >= 200);
  CHECK("string_clear: null -> ERR_NULL", cf_string_clear(CF_NULL) == CF_ERR_NULL);
  CHECK("string_is_valid: after clear", cf_string_is_valid(str) == CF_TRUE);
  CHECK("string_is_empty: after clear", cf_string_is_empty(str) == CF_TRUE);

  cf_string_destroy(&str);
  CHECK("string_destroy: null data", str.data == CF_NULL);
  CHECK("string_destroy: len 0",     str.len  == 0);
  CHECK("string_destroy: cap 0",     str.cap  == 0);
  cf_string_destroy(CF_NULL);
  CHECK("string_destroy: null safe", CF_TRUE);
}

static void test_cf_string_set_and_views(void)
{
  SECTION("cf_string set/view");

  cf_string str;
  CHECK("string_set/view: init", cf_string_init(&str, 4) == CF_OK);

  cf_str hello = cf_str_from("hello", 5);
  cf_str hi    = cf_str_from("hi",    2);

  CHECK("string_set_str: first set ok",       cf_string_set_str(&str, hello) == CF_OK);
  CHECK("string_set_str: len == 5",           str.len == 5);
  CHECK("string_set_str: content hello",      memcmp(str.data, "hello", 5) == 0);
  CHECK("string_set_str: null term hello",    str.data[5] == '\0');
  CHECK("string_set_str: replace ok",         cf_string_set_str(&str, hi) == CF_OK);
  CHECK("string_set_str: len == 2",           str.len == 2);
  CHECK("string_set_str: content replaced",   memcmp(str.data, "hi", 2) == 0);
  CHECK("string_set_str: null term replace",  str.data[2] == '\0');
  CHECK("string_set_str: set empty ok",       cf_string_set_str(&str, cf_str_empty()) == CF_OK);
  CHECK("string_set_str: set empty len",      str.len == 0);
  CHECK("string_set_str: set empty nul",      str.data[0] == '\0');
  CHECK("string_set_str: null -> ERR_NULL",   cf_string_set_str(CF_NULL, hello) == CF_ERR_NULL);
  CHECK("string_set_str: bad -> ERR_STATE",   cf_string_set_str(&str, (cf_str){CF_NULL, 3}) == CF_ERR_STATE);

  CHECK("string_set_str: restore",   cf_string_set_str(&str, hello) == CF_OK);
  cf_str view = cf_string_as_str(str);
  CHECK("string_as_str: len",        view.len  == str.len);
  CHECK("string_as_str: ptr",        view.data == str.data);
  CHECK("string_as_str: content",    memcmp(view.data, "hello", 5) == 0);

  cf_string_destroy(&str);
}

static void test_cf_string_queries_and_truncate(void)
{
  SECTION("cf_string queries/truncate");

  cf_string str;
  CHECK("init", cf_string_init(&str, 8) == CF_OK);
  CHECK("set hello", cf_string_set_str(&str, cf_str_from("hello", 5)) == CF_OK);

  cf_str view = cf_string_as_str(str);
  CHECK("string_as_str: len",     view.len == 5);
  CHECK("string_as_str: ptr",     view.data == str.data);
  CHECK("string_as_str: content", memcmp(view.data, "hello", 5) == 0);

  {
    char ch = 0;
    CHECK("str_at: index 1",             cf_str_at(view, 1, &ch) == CF_OK);
    CHECK("str_at: got 'e'",             ch == 'e');
    CHECK("str_at: null -> ERR_NULL",    cf_str_at(view, 0, CF_NULL) == CF_ERR_NULL);
    CHECK("str_at: bounds -> ERR_BOUNDS",cf_str_at(view, 5, &ch) == CF_ERR_BOUNDS);
    CHECK("str_at: bad -> ERR_STATE",    cf_str_at((cf_str){CF_NULL, 3}, 0, &ch) == CF_ERR_STATE);
  }

  {
    char ch = 0;
    CHECK("string_at: index 4",             cf_string_at(str, 4, &ch) == CF_OK);
    CHECK("string_at: got 'o'",             ch == 'o');
    CHECK("string_at: null -> ERR_NULL",    cf_string_at(str, 0, CF_NULL) == CF_ERR_NULL);
    CHECK("string_at: bounds -> ERR_BOUNDS",cf_string_at(str, 5, &ch) == CF_ERR_BOUNDS);
  }

  {
    cf_bool out = CF_FALSE;
    CHECK("starts_with: he",      cf_str_starts_with(view, cf_str_from("he", 2), &out) == CF_OK && out == CF_TRUE);
    CHECK("starts_with: hi false",cf_str_starts_with(view, cf_str_from("hi", 2), &out) == CF_OK && out == CF_FALSE);
    CHECK("starts_with: longer",  cf_str_starts_with(view, cf_str_from("hello!", 6), &out) == CF_OK && out == CF_FALSE);
    CHECK("starts_with: null",    cf_str_starts_with(view, cf_str_from("he", 2), CF_NULL) == CF_ERR_NULL);
  }

  {
    cf_bool out = CF_FALSE;
    CHECK("ends_with: lo",       cf_str_ends_with(view, cf_str_from("lo", 2), &out) == CF_OK && out == CF_TRUE);
    CHECK("ends_with: xx false", cf_str_ends_with(view, cf_str_from("xx", 2), &out) == CF_OK && out == CF_FALSE);
    CHECK("ends_with: longer",   cf_str_ends_with(view, cf_str_from("ohhello", 7), &out) == CF_OK && out == CF_FALSE);
    CHECK("ends_with: null",     cf_str_ends_with(view, cf_str_from("lo", 2), CF_NULL) == CF_ERR_NULL);
  }

  CHECK("string_truncate: shrink to 2",    cf_string_truncate(&str, 2) == CF_OK);
  CHECK("string_truncate: len == 2",       str.len == 2);
  CHECK("string_truncate: nul kept",       str.data[2] == '\0');
  CHECK("string_truncate: content kept",   str.data[0] == 'h' && str.data[1] == 'e');
  CHECK("string_truncate: to zero",        cf_string_truncate(&str, 0) == CF_OK);
  CHECK("string_truncate: len == 0",       str.len == 0);
  CHECK("string_truncate: data[0] nul",    str.data[0] == '\0');
  CHECK("string_truncate: too large",      cf_string_truncate(&str, 1) == CF_ERR_BOUNDS);
  CHECK("string_truncate: null -> ERR_NULL", cf_string_truncate(CF_NULL, 0) == CF_ERR_NULL);

  cf_string_destroy(&str);
}

static void test_cf_str_find_trim_split_and_ignore_case(void)
{
  SECTION("cf_str find/trim/split/ignore-case");

  {
    cf_bool eq = CF_FALSE;
    CHECK("eq_ignore_case: hello/HELLO",
          cf_str_eq_ignore_case(cf_str_from("hello", 5), cf_str_from("HELLO", 5), &eq) == CF_OK && eq == CF_TRUE);
    CHECK("eq_ignore_case: AbC/aBc",
          cf_str_eq_ignore_case(cf_str_from("AbC", 3), cf_str_from("aBc", 3), &eq) == CF_OK && eq == CF_TRUE);
    CHECK("eq_ignore_case: mixed symbols equal",
          cf_str_eq_ignore_case(cf_str_from("A1!", 3), cf_str_from("a1!", 3), &eq) == CF_OK && eq == CF_TRUE);
    CHECK("eq_ignore_case: mixed symbols not equal",
          cf_str_eq_ignore_case(cf_str_from("A1!", 3), cf_str_from("a1?", 3), &eq) == CF_OK && eq == CF_FALSE);
    CHECK("eq_ignore_case: different lengths",
          cf_str_eq_ignore_case(cf_str_from("hi", 2), cf_str_from("HIGH", 4), &eq) == CF_OK && eq == CF_FALSE);
    CHECK("eq_ignore_case: null out",
          cf_str_eq_ignore_case(cf_str_from("abc", 3), cf_str_from("ABC", 3), CF_NULL) == CF_ERR_NULL);
    CHECK("eq_ignore_case: bad state",
          cf_str_eq_ignore_case((cf_str){CF_NULL, 3}, cf_str_from("ABC", 3), &eq) == CF_ERR_STATE);
  }

  {
    cf_usize index = 999;
    cf_bool found = CF_FALSE;
    cf_str s = cf_str_from("hello world", 11);

    CHECK("find_char: found 'o'",    cf_str_find_char(s, 'o', &index, &found) == CF_OK && found == CF_TRUE  && index == 4);
    CHECK("find_char: found 'h'",    cf_str_find_char(s, 'h', &index, &found) == CF_OK && found == CF_TRUE  && index == 0);
    CHECK("find_char: not found",    cf_str_find_char(s, 'z', &index, &found) == CF_OK && found == CF_FALSE && index == 0);
    CHECK("find_char: null index",   cf_str_find_char(s, 'h', CF_NULL, &found) == CF_ERR_NULL);
    CHECK("find_char: null found",   cf_str_find_char(s, 'h', &index, CF_NULL) == CF_ERR_NULL);
    CHECK("find_char: bad state",    cf_str_find_char((cf_str){CF_NULL, 3}, 'h', &index, &found) == CF_ERR_STATE);

    CHECK("find_str: basic match",   cf_str_find_str(s, cf_str_from("world", 5), &index, &found) == CF_OK && found == CF_TRUE && index == 6);
    CHECK("find_str: prefix match",  cf_str_find_str(s, cf_str_from("hello", 5), &index, &found) == CF_OK && found == CF_TRUE && index == 0);
    CHECK("find_str: overlap",       cf_str_find_str(cf_str_from("ababa", 5), cf_str_from("aba", 3), &index, &found) == CF_OK && found == CF_TRUE && index == 0);
    CHECK("find_str: not found",     cf_str_find_str(s, cf_str_from("xyz", 3), &index, &found) == CF_OK && found == CF_FALSE && index == 0);
    CHECK("find_str: needle longer", cf_str_find_str(cf_str_from("hi", 2), cf_str_from("hello", 5), &index, &found) == CF_OK && found == CF_FALSE);
    CHECK("find_str: empty needle",  cf_str_find_str(s, cf_str_empty(), &index, &found) == CF_OK && found == CF_FALSE);
    CHECK("find_str: null index",    cf_str_find_str(s, cf_str_from("he", 2), CF_NULL, &found) == CF_ERR_NULL);
    CHECK("find_str: null found",    cf_str_find_str(s, cf_str_from("he", 2), &index, CF_NULL) == CF_ERR_NULL);
  }

  {
    cf_str out = cf_str_empty();
    CHECK("trim_left: leading spaces",   cf_str_trim_left(cf_str_from("  hello", 7), &out) == CF_OK && out.len == 5 && memcmp(out.data, "hello", 5) == 0);
    CHECK("trim_right: trailing spaces", cf_str_trim_right(cf_str_from("hello  ", 7), &out) == CF_OK && out.len == 5 && memcmp(out.data, "hello", 5) == 0);
    CHECK("trim: both sides",            cf_str_trim(cf_str_from("  hello  ", 9), &out) == CF_OK && out.len == 5 && memcmp(out.data, "hello", 5) == 0);
    CHECK("trim: tabs/newlines",         cf_str_trim(cf_str_from("\t\r\n hello \r\n", 12), &out) == CF_OK && out.len == 5 && memcmp(out.data, "hello", 5) == 0);
    CHECK("trim: empty stays empty",     cf_str_trim(cf_str_empty(), &out) == CF_OK && out.len == 0);
    CHECK("trim: all whitespace",        cf_str_trim(cf_str_from(" \t\r\n ", 5), &out) == CF_OK && out.len == 0);
    CHECK("trim_left: null out",         cf_str_trim_left(cf_str_from("x", 1), CF_NULL) == CF_ERR_NULL);
    CHECK("trim_right: null out",        cf_str_trim_right(cf_str_from("x", 1), CF_NULL) == CF_ERR_NULL);
    CHECK("trim: null out",              cf_str_trim(cf_str_from("x", 1), CF_NULL) == CF_ERR_NULL);
  }

  {
    cf_str left = cf_str_empty(), right = cf_str_empty();
    cf_bool found = CF_FALSE;
    CHECK("split_once: key=value",  cf_str_split_once_char(cf_str_from("key=value", 9), '=', &left, &right, &found) == CF_OK && found == CF_TRUE  && left.len == 3 && memcmp(left.data, "key", 3) == 0 && right.len == 5 && memcmp(right.data, "value", 5) == 0);
    CHECK("split_once: =value",     cf_str_split_once_char(cf_str_from("=value", 6),    '=', &left, &right, &found) == CF_OK && found == CF_TRUE  && left.len == 0 && right.len == 5 && memcmp(right.data, "value", 5) == 0);
    CHECK("split_once: key=",       cf_str_split_once_char(cf_str_from("key=", 4),       '=', &left, &right, &found) == CF_OK && found == CF_TRUE  && left.len == 3 && memcmp(left.data, "key", 3) == 0 && right.len == 0);
    CHECK("split_once: not found",  cf_str_split_once_char(cf_str_from("keyvalue", 8),   '=', &left, &right, &found) == CF_OK && found == CF_FALSE && left.len == 0 && right.len == 0);
    CHECK("split_once: null left",  cf_str_split_once_char(cf_str_from("a=b", 3), '=', CF_NULL, &right, &found) == CF_ERR_NULL);
    CHECK("split_once: null right", cf_str_split_once_char(cf_str_from("a=b", 3), '=', &left, CF_NULL, &found) == CF_ERR_NULL);
    CHECK("split_once: null found", cf_str_split_once_char(cf_str_from("a=b", 3), '=', &left, &right, CF_NULL) == CF_ERR_NULL);
  }
}

/* ------------------------------------------------------------------ */
/*  cf_status tests                                                    */
/* ------------------------------------------------------------------ */

static cf_status    g_handler_last_status  = CF_OK;
static const char  *g_handler_last_context = CF_NULL;
static cf_bool      g_handler_should_claim = CF_TRUE;

static cf_bool test_handler(cf_status status, const char *context)
{
  g_handler_last_status  = status;
  g_handler_last_context = context;
  return g_handler_should_claim;
}

static void test_cf_status(void)
{
  SECTION("cf_status_print");

  /* all built-in codes must not crash */
  printf("  (following lines go to stderr — expected)\n");
  cf_status_print(CF_OK,              CF_NULL);
  cf_status_print(CF_ERR_INVALID,     CF_NULL);
  cf_status_print(CF_ERR_NULL,        CF_NULL);
  cf_status_print(CF_ERR_OOM,         CF_NULL);
  cf_status_print(CF_ERR_OVERFLOW,    CF_NULL);
  cf_status_print(CF_ERR_BOUNDS,      CF_NULL);
  cf_status_print(CF_ERR_STATE,       CF_NULL);
  cf_status_print(CF_ERR_UNSUPPORTED, CF_NULL);
  cf_status_print(CF_ERR_DENIED,      CF_NULL);
  cf_status_print(CF_ERR_INTERNAL,    CF_NULL);
  CHECK("status_print: all built-ins don't crash", CF_TRUE);

  /* context prefix */
  cf_status_print(CF_ERR_OOM, "test_context");
  CHECK("status_print: context prefix doesn't crash", CF_TRUE);

  /* unknown code without a handler falls back to "unknown status" */
  cf_status_print((cf_status)999, CF_NULL);
  CHECK("status_print: unknown code doesn't crash", CF_TRUE);

  /* custom handler is invoked and receives the right arguments */
  cf_status_set_handler(test_handler);

  g_handler_should_claim = CF_TRUE;
  cf_status_print(CF_ERR_OOM, "ctx_a");
  CHECK("handler: was called",        g_handler_last_status == CF_ERR_OOM);
  CHECK("handler: context forwarded", g_handler_last_context != CF_NULL &&
        memcmp(g_handler_last_context, "ctx_a", 5) == 0);

  /* handler claiming CF_TRUE suppresses the default path */
  g_handler_should_claim = CF_TRUE;
  cf_status_print(CF_ERR_NULL, "ctx_b");
  CHECK("handler: CF_TRUE claimed",   g_handler_last_status == CF_ERR_NULL);

  /* handler returning CF_FALSE still records the call and falls through */
  g_handler_should_claim = CF_FALSE;
  cf_status_print(CF_ERR_BOUNDS, "ctx_c");
  CHECK("handler: CF_FALSE records call", g_handler_last_status == CF_ERR_BOUNDS);

  /* handler receives unknown/extended status codes */
  g_handler_should_claim = CF_TRUE;
  cf_status_print((cf_status)42, "ctx_ext");
  CHECK("handler: extended code received",    g_handler_last_status == (cf_status)42);
  CHECK("handler: extended context forwarded",
        g_handler_last_context != CF_NULL &&
        memcmp(g_handler_last_context, "ctx_ext", 7) == 0);

  /* removing the handler restores default behaviour */
  cf_status_set_handler(CF_NULL);
  g_handler_last_status = CF_OK;          /* reset sentinel */
  cf_status_print(CF_ERR_INTERNAL, CF_NULL);
  CHECK("handler: removed, sentinel unchanged", g_handler_last_status == CF_OK);
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
  test_cf_buffer_set_and_views();
  test_cf_string_set_and_views();
  test_cf_buffer_fill_and_truncate();
  test_cf_string_queries_and_truncate();
  test_cf_str_find_trim_split_and_ignore_case();
  test_cf_status();

  printf("\n==============================\n");
  printf("Results: %d passed, %d failed\n", g_passed, g_failed);
  printf("==============================\n");

  return g_failed == 0 ? 0 : 1;
}