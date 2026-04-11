#include "cf_string.h"
#include <stdlib.h>
#include <string.h>

cf_str cf_str_empty(void) {return (cf_str) {CF_NULL, 0};}

cf_string cf_string_empty(void) {return (cf_string) {CF_NULL, 0, 0};}

cf_str cf_str_from(const char *data, cf_usize len) {return (cf_str) {data, len};}

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

cf_bool cf_str_is_empty(cf_str str) {return str.len == 0;}

cf_bool cf_string_is_empty(cf_string str) {return str.len == 0;}

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

cf_status cf_str_slice(cf_str src, cf_usize offset, cf_usize len, cf_str *dst)
{
  if(dst == CF_NULL) return CF_ERR_NULL;
  if(!cf_str_is_valid(src)) return CF_ERR_STATE;
  if(offset > src.len || len > src.len - offset) return CF_ERR_BOUNDS;
  dst->data = src.data + offset;
  dst->len = len;
  return CF_OK;
}

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

static cf_usize cf_next_cap(cf_usize current, cf_usize required)
{
  cf_usize cap = current == 0 ? 32 : current;
  while (cap < required && cap < 2048) cap *= 2;
  if(cap < required) cap = required;
  return cap;
}

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

cf_str cf_string_as_str(cf_string str)
{
  if(!cf_string_is_valid(str)) return cf_str_empty();
  return cf_str_from(str.data, str.len);
}

cf_status cf_string_set_str(cf_string *str, cf_str s)
{
  if(str == CF_NULL) return CF_ERR_NULL;
  if(!cf_string_is_valid(*str) || !cf_str_is_valid(s)) return CF_ERR_STATE;
  cf_status state;
  if((state = cf_string_clear(str)) != CF_OK) return state;
  return cf_string_append_str(str, s);
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

cf_status cf_string_truncate(cf_string *str, cf_usize new_len)
{
  if(str == CF_NULL) return CF_ERR_NULL;
  if(!cf_string_is_valid(*str)) return CF_ERR_STATE;
  if(new_len > str->len) return CF_ERR_BOUNDS;
  str->len = new_len;
  str->data[str->len] = '\0';
  return CF_OK;
}

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