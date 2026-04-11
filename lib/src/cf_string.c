#include "cf_string.h"
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */

static cf_usize cf_next_cap(cf_usize current, cf_usize required)
{
  cf_usize cap = current == 0 ? 32 : current;
  while (cap < required && cap < 2048) cap *= 2;
  if(cap < required) cap = required;
  return cap;
}

static cf_bool cf_char_is_alphabet(char c)
{
  return (cf_bool) (('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z'));
}

static cf_status cf_char_to_lower(char c, char *out_ch)
{
  if(out_ch == CF_NULL) return CF_ERR_NULL;
  if(!cf_char_is_alphabet(c)) return CF_ERR_INVALID;
  *out_ch = c;
  if('A' <= c && c <= 'Z')
    *out_ch = c - 'A' + 'a';
  return CF_OK;
}

static cf_bool cf_char_is_space(char c)
{
  return (c == ' ' || c == '\t' || c == '\r' || c == '\n') ? CF_TRUE : CF_FALSE;
}

/* ------------------------------------------------------------------ */
/* Construction                                                        */
/* ------------------------------------------------------------------ */

cf_str cf_str_empty(void) {return (cf_str) {CF_NULL, 0};}

cf_string cf_string_empty(void) {return (cf_string) {CF_NULL, 0, 0};}

cf_str cf_str_from(const char *data, cf_usize len) {return (cf_str) {data, len};}

/* ------------------------------------------------------------------ */
/* Validation                                                          */
/* ------------------------------------------------------------------ */

cf_bool cf_str_is_valid(cf_str str) 
{
  if(str.len > 0 && str.data == CF_NULL)
    return CF_FALSE;
  return CF_TRUE;
}

cf_bool cf_string_is_valid(cf_string str) 
{
  if(str.len > str.cap) return CF_FALSE;
  if(str.data == CF_NULL) 
  {
    if(str.len > 0 || str.cap > 0)
      return CF_FALSE;
  }
  else
  {
    if(str.data[str.len] != '\0')
      return CF_FALSE;
  }
  return CF_TRUE;
}

/* ------------------------------------------------------------------ */
/* Emptiness                                                           */
/* ------------------------------------------------------------------ */

cf_bool cf_str_is_empty(cf_str str) {return str.len == 0;}

cf_bool cf_string_is_empty(cf_string str) {return str.len == 0;}

/* ------------------------------------------------------------------ */
/* Equality / comparison                                               */
/* ------------------------------------------------------------------ */

cf_status cf_str_eq(cf_str s1, cf_str s2, cf_bool *out_eq)
{
  if(out_eq == CF_NULL) return CF_ERR_NULL;
  if(!cf_str_is_valid(s1) || !cf_str_is_valid(s2))
    return CF_ERR_STATE;
  *out_eq = CF_FALSE;
  if(s1.len != s2.len) return CF_OK;
  for (cf_usize i = 0; i < s1.len; i++)
    if(s1.data[i] != s2.data[i]) 
      return CF_OK;
  *out_eq = CF_TRUE;
  return CF_OK;
}

cf_status cf_str_eq_ignore_case(cf_str s1, cf_str s2, cf_bool *out_eq)
{
  if(out_eq == CF_NULL) return CF_ERR_NULL;
  if(!cf_str_is_valid(s1) || !cf_str_is_valid(s2))
    return CF_ERR_STATE;
  *out_eq = CF_FALSE;
  if(s1.len != s2.len) return CF_OK;
  for (cf_usize i = 0; i < s1.len; i++)
  {
    char c1, c2;
    if(
        cf_char_to_lower(s1.data[i], &c1) == CF_OK 
      && 
        cf_char_to_lower(s2.data[i], &c2) == CF_OK
      )
    { 
      if(c1 != c2) return CF_OK;
    }
    else if(s1.data[i] != s2.data[i]) return CF_OK;
  }
  *out_eq = CF_TRUE;
  return CF_OK;
}

/* ------------------------------------------------------------------ */
/* Prefix / suffix checks                                              */
/* ------------------------------------------------------------------ */

cf_status cf_str_starts_with(cf_str s, cf_str prefix, cf_bool *out)
{
  if (out == CF_NULL) return CF_ERR_NULL;
  if(!cf_str_is_valid(s) || !cf_str_is_valid(prefix)) return CF_ERR_STATE;
  *out = CF_FALSE;
  if(s.len < prefix.len) return CF_OK;
  for (cf_usize i = 0; i < prefix.len; i++)
    if(s.data[i] != prefix.data[i])
      return CF_OK;
  *out = CF_TRUE;
  return CF_OK;
}

cf_status cf_str_ends_with(cf_str s, cf_str suffix, cf_bool *out)
{
  if (out == CF_NULL) return CF_ERR_NULL;
  if(!cf_str_is_valid(s) || !cf_str_is_valid(suffix)) return CF_ERR_STATE;
  *out = CF_FALSE;
  if(s.len < suffix.len) return CF_OK;
  cf_usize start = s.len - suffix.len;
  for (cf_usize i = 0; i < suffix.len ; i++)
    if(s.data[start + i] != suffix.data[i])
      return CF_OK;
  *out = CF_TRUE;
  return CF_OK;
}

/* ------------------------------------------------------------------ */
/* Slicing / indexing                                                  */
/* ------------------------------------------------------------------ */

cf_status cf_str_slice(cf_str src, cf_usize offset, cf_usize len, cf_str *dst)
{
  if(dst == CF_NULL) return CF_ERR_NULL;
  if(!cf_str_is_valid(src)) return CF_ERR_STATE;
  if(offset > src.len || len > src.len - offset) return CF_ERR_BOUNDS;
  dst->data = src.data + offset;
  dst->len = len;
  return CF_OK;
}

cf_status cf_str_at(cf_str s, cf_usize index, char *out_ch)
{
  if(out_ch == CF_NULL) return CF_ERR_NULL;
  if(!cf_str_is_valid(s)) return CF_ERR_STATE;
  if(index >= s.len) return CF_ERR_BOUNDS;
  *out_ch = s.data[index];
  return CF_OK;
}

cf_status cf_string_at(cf_string str, cf_usize index, char *out_ch)
{
  if(out_ch == CF_NULL) return CF_ERR_NULL;
  if(!cf_string_is_valid(str)) return CF_ERR_STATE;
  if(index >= str.len) return CF_ERR_BOUNDS;
  *out_ch = str.data[index];
  return CF_OK;
}

/* ------------------------------------------------------------------ */
/* Search                                                              */
/* ------------------------------------------------------------------ */

cf_status cf_str_find_char(cf_str s, char ch, cf_usize *out_index, cf_bool *out_found)
{
  if(out_index == CF_NULL || out_found == CF_NULL) return CF_ERR_NULL;
  if(!cf_str_is_valid(s)) return CF_ERR_STATE;
  *out_index = 0; 
  *out_found = CF_FALSE;
  for (cf_usize i = 0; i < s.len; i++)
    if(s.data[i] == ch)
    {
      *out_index = i; 
      *out_found = CF_TRUE;
      return CF_OK;
    }
  return CF_OK;
}

cf_status cf_str_find_str(cf_str s, cf_str needle, cf_usize *out_index, cf_bool *out_found)
{
  if(out_index == CF_NULL || out_found == CF_NULL) return CF_ERR_NULL;
  if(!cf_str_is_valid(s) || !cf_str_is_valid(needle)) return CF_ERR_STATE;
  *out_index = 0; 
  *out_found = CF_FALSE;
  if(needle.len > s.len) return CF_OK;
  if(cf_str_is_empty(s) || cf_str_is_empty(needle)) return CF_OK;
  for (cf_usize i = 0; i < s.len; i++)
  {
    cf_bool match = CF_TRUE;
    if((s.len - i) >= needle.len && s.data[i] == needle.data[0])
    {
      for (cf_usize j = 0; j < needle.len; j++)
      {
        if(s.data[i + j] != needle.data[j]) 
        {match = CF_FALSE; break;}
      }
      if(match)
      {*out_index = i; *out_found = CF_TRUE; return CF_OK;}
    }
  }
  return CF_OK;
}

/* ------------------------------------------------------------------ */
/* Trim                                                                */
/* ------------------------------------------------------------------ */

cf_status cf_str_trim_left(cf_str s, cf_str *out)
{
  if(out == CF_NULL) return CF_ERR_NULL;
  if(!cf_str_is_valid(s)) return CF_ERR_STATE;
  cf_usize offset = 0;
  while (offset < s.len && cf_char_is_space(s.data[offset]))
    offset++;
  *out = cf_str_from(s.data + offset, s.len - offset);
  return CF_OK;
}

cf_status cf_str_trim_right(cf_str s, cf_str *out)
{
  if(out == CF_NULL) return CF_ERR_NULL;
  if(!cf_str_is_valid(s)) return CF_ERR_STATE;
  cf_usize offset = s.len;
  while (offset > 0 && cf_char_is_space(s.data[offset - 1]))
    offset--;
  *out = cf_str_from(s.data, offset);
  return CF_OK;
}

cf_status cf_str_trim(cf_str s, cf_str *out)
{
  if(out == CF_NULL) return CF_ERR_NULL;
  if(!cf_str_is_valid(s)) return CF_ERR_STATE;
  cf_status state;
  cf_str tmp = cf_str_empty();
  if((state = cf_str_trim_left(s, &tmp)) != CF_OK) return state;
  if((state = cf_str_trim_right(tmp, out)) != CF_OK) return state;
  return CF_OK;
}

/* ------------------------------------------------------------------ */
/* Split                                                               */
/* ------------------------------------------------------------------ */

cf_status cf_str_split_once_char(cf_str s, char sep, cf_str *left, cf_str *right, cf_bool *out_found)
{
  if(left == CF_NULL || right == CF_NULL || out_found == CF_NULL) return CF_ERR_NULL;
  if(!cf_str_is_valid(s)) return CF_ERR_STATE;
  *right = *left = cf_str_empty();
  cf_status state; cf_usize index;
  if((state = cf_str_find_char(s, sep, &index, out_found)) != CF_OK) return state;
  if(*out_found == CF_FALSE) return CF_OK;
  cf_usize newlen = index + 1;
  *right = cf_str_from(s.data + newlen, s.len - newlen);
  *left = cf_str_from(s.data, index);
  return CF_OK;
}

/* ------------------------------------------------------------------ */
/* String lifecycle                                                    */
/* ------------------------------------------------------------------ */

cf_status cf_string_init(cf_string *str, cf_usize cap)
{
  if(str == CF_NULL) return CF_ERR_NULL;
  *str = cf_string_empty();
  return cf_string_reserve(str, cap);
}

cf_status cf_string_reserve(cf_string *str, cf_usize min_cap)
{
  if(str == CF_NULL) return CF_ERR_NULL;
  if(!cf_string_is_valid(*str)) return CF_ERR_STATE;
  if(str->cap >= min_cap) return CF_OK;
  char *tmp;
  if(str->data == CF_NULL) tmp = malloc((min_cap + 1) * sizeof (char));
  else tmp = realloc(str->data, (min_cap + 1) * sizeof (char));
  if(tmp == CF_NULL) return CF_ERR_OOM;
  str->data = tmp;
  str->data[str->len] = '\0';
  str->cap = min_cap;
  return CF_OK;
}

cf_status cf_string_clear(cf_string *str)
{
  if(str == CF_NULL) return CF_ERR_NULL;
  if(!cf_string_is_valid(*str)) return CF_ERR_STATE;
  if(str->data != CF_NULL) str->data[0] = '\0';
  str->len = 0;
  return CF_OK;
}

void cf_string_destroy(cf_string *str)
{
  if(str == CF_NULL) return;
  free(str->data);
  *str = cf_string_empty();
}

/* ------------------------------------------------------------------ */
/* String append / set                                                 */
/* ------------------------------------------------------------------ */

cf_status cf_string_append_char(cf_string *str, char s)
{
  if(str == CF_NULL) return CF_ERR_NULL;
  if(!cf_string_is_valid(*str)) return CF_ERR_STATE;
  if(str->cap < str->len + 1) 
  {
    cf_status state;
    if((state = cf_string_reserve(
    str, 
    cf_next_cap(str->cap, str->len + 1)
    )) != CF_OK) return state;
  }
  str->data[str->len] = s;
  str->len++;
  str->data[str->len] = '\0';
  return CF_OK;
}

cf_status cf_string_append_str(cf_string *str, cf_str s)
{
  if(str == CF_NULL) return CF_ERR_NULL;
  if(!cf_string_is_valid(*str) || !cf_str_is_valid(s)) return CF_ERR_STATE;
  if(s.len == 0) return CF_OK;
  if(str->cap < (str->len + s.len))
  {
    cf_status state;
    if((state = cf_string_reserve(
      str, 
      cf_next_cap(str->cap, str->len + s.len)
    )) != CF_OK) return state;
  }
  memcpy(str->data + str->len, s.data, s.len);
  str->len += s.len;
  str->data[str->len] = '\0';
  return CF_OK;
}

cf_status cf_string_set_str(cf_string *str, cf_str s)
{
  if(str == CF_NULL) return CF_ERR_NULL;
  if(!cf_string_is_valid(*str) || !cf_str_is_valid(s)) return CF_ERR_STATE;
  cf_status state;
  if((state = cf_string_clear(str)) != CF_OK) return state;
  return cf_string_append_str(str, s);
}

/* ------------------------------------------------------------------ */
/* String views                                                        */
/* ------------------------------------------------------------------ */

cf_str cf_string_as_str(cf_string str)
{
  if(!cf_string_is_valid(str)) return cf_str_empty();
  return cf_str_from(str.data, str.len);
}

/* ------------------------------------------------------------------ */
/* String truncate                                                     */
/* ------------------------------------------------------------------ */

cf_status cf_string_truncate(cf_string *str, cf_usize new_len)
{
  if(str == CF_NULL) return CF_ERR_NULL;
  if(!cf_string_is_valid(*str)) return CF_ERR_STATE;
  if(new_len > str->len) return CF_ERR_BOUNDS;
  str->len = new_len;
  str->data[str->len] = '\0';
  return CF_OK;
}