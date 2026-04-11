#include "cf_memory.h"
#include <stdlib.h>
#include <string.h>

cf_bytes cf_bytes_empty(void) { return (cf_bytes){CF_NULL, 0}; }

cf_bytes_mut cf_bytes_mut_empty(void) { return (cf_bytes_mut){CF_NULL, 0}; }

cf_buffer cf_buffer_empty(void) { return (cf_buffer){CF_NULL, 0, 0}; }

cf_bytes cf_bytes_from(const cf_u8 *data, cf_usize len) { return (cf_bytes){data, len}; }

cf_bytes_mut cf_bytes_mut_from(cf_u8 *data, cf_usize len) { return (cf_bytes_mut){data, len}; }

cf_bool cf_bytes_is_valid(cf_bytes bytes) { return bytes.len > 0 && bytes.data == CF_NULL ? CF_FALSE : CF_TRUE; }

cf_bool cf_bytes_mut_is_valid(cf_bytes_mut bytes) { return bytes.len > 0 && bytes.data == CF_NULL ? CF_FALSE : CF_TRUE; }

cf_bool cf_buffer_is_valid(cf_buffer buffer) { return ((buffer.len > 0 || buffer.cap > 0) && buffer.data == CF_NULL) || (buffer.len > buffer.cap) ? CF_FALSE : CF_TRUE; }

cf_bool cf_bytes_is_empty(cf_bytes bytes) { return bytes.len == 0; }

cf_bool cf_bytes_mut_is_empty(cf_bytes_mut bytes) { return bytes.len == 0; }

cf_bool cf_buffer_is_empty(cf_buffer buffer) { return buffer.len == 0; }

cf_status cf_bytes_eq(cf_bytes b1, cf_bytes b2, cf_bool *out_eq)
{
  if (out_eq == CF_NULL)
    return CF_ERR_NULL;
  if (!cf_bytes_is_valid(b1) || !cf_bytes_is_valid(b2))
    return CF_ERR_STATE;
  *out_eq = CF_FALSE;
  if (b1.len != b2.len)
    return CF_OK;
  for (cf_usize i = 0; i < b1.len; i++)
    if (b1.data[i] != b2.data[i])
      return CF_OK;
  *out_eq = CF_TRUE;
  return CF_OK;
}

cf_status cf_bytes_mut_zero(cf_bytes_mut bytes)
{
  if (!cf_bytes_mut_is_valid(bytes))
    return CF_ERR_STATE;
  if (bytes.len == 0)
    return CF_OK;
  memset(bytes.data, 0, bytes.len);
  return CF_OK;
}

cf_status cf_bytes_slice(cf_bytes src, cf_usize offset, cf_usize len, cf_bytes *dst)
{
  if (dst == CF_NULL)
    return CF_ERR_NULL;
  if (!cf_bytes_is_valid(src))
    return CF_ERR_STATE;
  if (offset > src.len || len > src.len - offset)
    return CF_ERR_BOUNDS;
  dst->data = src.data + offset;
  dst->len = len;
  return CF_OK;
}

cf_status cf_bytes_mut_slice(cf_bytes_mut src, cf_usize offset, cf_usize len, cf_bytes_mut *dst)
{
  if (dst == CF_NULL)
    return CF_ERR_NULL;
  if (!cf_bytes_mut_is_valid(src))
    return CF_ERR_STATE;
  if (offset > src.len || len > src.len - offset)
    return CF_ERR_BOUNDS;
  dst->data = src.data + offset;
  dst->len = len;
  return CF_OK;
}

cf_status cf_buffer_init(cf_buffer *buffer, cf_usize cap)
{
  if (buffer == CF_NULL)
    return CF_ERR_NULL;
  *buffer = cf_buffer_empty();
  return cf_buffer_reserve(buffer, cap);
}

cf_status cf_buffer_reserve(cf_buffer *buffer, cf_usize min_cap)
{
  if (buffer == CF_NULL)
    return CF_ERR_NULL;
  if (!cf_buffer_is_valid(*buffer))
    return CF_ERR_STATE;
  if (buffer->cap >= min_cap)
    return CF_OK;
  cf_u8 *tmp;
  if (buffer->data == CF_NULL)
    tmp = malloc(min_cap * sizeof(cf_u8));
  else
    tmp = realloc(buffer->data, min_cap * sizeof(cf_u8));
  if (tmp == CF_NULL)
    return CF_ERR_OOM;
  buffer->data = tmp;
  buffer->cap = min_cap;
  return CF_OK;
}

cf_status cf_buffer_clear(cf_buffer *buffer)
{
  if (buffer == CF_NULL)
    return CF_ERR_NULL;
  if (!cf_buffer_is_valid(*buffer))
    return CF_ERR_STATE;
  buffer->len = 0;
  return CF_OK;
}

void cf_buffer_destroy(cf_buffer *buffer)
{
  if (buffer == CF_NULL)
    return;
  free(buffer->data);
  *buffer = cf_buffer_empty();
}

static cf_usize cf_next_cap(cf_usize current, cf_usize required)
{
  cf_usize cap = current == 0 ? 32 : current;
  while (cap < required && cap < 2048)
    cap *= 2;
  if (cap < required)
    cap = required;
  return cap;
}

cf_status cf_buffer_append_byte(cf_buffer *buffer, cf_u8 byte)
{
  if (buffer == CF_NULL)
    return CF_ERR_NULL;
  if (!cf_buffer_is_valid(*buffer))
    return CF_ERR_STATE;
  if (buffer->cap == buffer->len)
  {
    cf_status state;
    if ((state = cf_buffer_reserve(
             buffer,
             cf_next_cap(buffer->cap, buffer->len + 1))) != CF_OK)
      return state;
  }
  buffer->data[buffer->len] = byte;
  buffer->len++;
  return CF_OK;
}

cf_status cf_buffer_append_bytes(cf_buffer *buffer, cf_bytes bytes)
{
  if (buffer == CF_NULL)
    return CF_ERR_NULL;
  if (!cf_buffer_is_valid(*buffer) || !cf_bytes_is_valid(bytes))
    return CF_ERR_STATE;
  if (bytes.len == 0)
    return CF_OK;

  if (buffer->cap < (buffer->len + bytes.len))
  {
    cf_status state;
    if ((state = cf_buffer_reserve(
             buffer,
             cf_next_cap(buffer->cap, buffer->len + bytes.len))) != CF_OK)
      return state;
  }
  memcpy(buffer->data + buffer->len, bytes.data, bytes.len);
  buffer->len += bytes.len;
  return CF_OK;
}

cf_bytes cf_buffer_as_bytes(cf_buffer buffer)
{
  if (!cf_buffer_is_valid(buffer))
    return cf_bytes_empty();
  return cf_bytes_from(buffer.data, buffer.len);
}

cf_bytes_mut cf_buffer_as_bytes_mut(cf_buffer *buffer)
{
  if (buffer == CF_NULL)
    return cf_bytes_mut_empty();
  if (!cf_buffer_is_valid(*buffer))
    return cf_bytes_mut_empty();
  return cf_bytes_mut_from(buffer->data, buffer->len);
}

cf_status cf_buffer_set_bytes(cf_buffer *buffer, cf_bytes bytes)
{
  if (buffer == CF_NULL)
    return CF_ERR_NULL;
  if (!cf_buffer_is_valid(*buffer) || !cf_bytes_is_valid(bytes))
    return CF_ERR_STATE;
  cf_status state;
  if ((state = cf_buffer_clear(buffer)) != CF_OK)
    return state;
  return cf_buffer_append_bytes(buffer, bytes);
}

cf_status cf_bytes_mut_fill(cf_bytes_mut bytes, cf_u8 value)
{
  if (!cf_bytes_mut_is_valid(bytes))
    return CF_ERR_STATE;
  if (bytes.len == 0)
    return CF_OK;
  memset(bytes.data, value, bytes.len);
  return CF_OK;
}

cf_status cf_buffer_fill(cf_buffer *buffer, cf_u8 value)
{
  if (buffer == CF_NULL)
    return CF_ERR_NULL;
  if (!cf_buffer_is_valid(*buffer))
    return CF_ERR_STATE;
  if (buffer->len == 0)
    return CF_OK;
  memset(buffer->data, value, buffer->len);
  return CF_OK;
}

cf_status cf_buffer_truncate(cf_buffer *buffer, cf_usize new_len)
{
  if (buffer == CF_NULL)
    return CF_ERR_NULL;
  if (!cf_buffer_is_valid(*buffer))
    return CF_ERR_STATE;
  if (new_len > buffer->len)
    return CF_ERR_BOUNDS;
  buffer->len = new_len;
  return CF_OK;
}