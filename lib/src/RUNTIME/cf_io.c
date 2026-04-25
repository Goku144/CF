/*
 * CF Framework
 * Copyright (C) 2026 Orion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "RUNTIME/cf_io.h"

#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>

cf_bool cf_io_exists(const char *path)
{
  if(path == CF_NULL) return CF_FALSE;
  return access(path, F_OK) == 0;
}

cf_status cf_io_file_size(const char *path, cf_usize *size)
{
  if(path == CF_NULL || size == CF_NULL) return CF_ERR_NULL;

  struct stat st;
  if(stat(path, &st) != 0) 
    return CF_ERR_IO_METADATA;
  *size = (cf_usize) st.st_size;
  return CF_OK;
}

cf_status cf_io_read_fd(cf_buffer *dst, int fd)
{
  if(dst == CF_NULL) return CF_ERR_NULL;
  if(fd < 0) return CF_ERR_IO_READ;
  if(cf_buffer_is_valid(dst) == CF_FALSE) return CF_ERR_STATE;

  while (1)
  {
    if(dst->len >= dst->cap)
    {
      if(dst->cap > CF_USIZE_MAX - CF_IO_RESERVE_SIZE) return CF_ERR_OVERFLOW;
      cf_status state = cf_buffer_reserve(dst, dst->cap + CF_IO_RESERVE_SIZE);
      if(state != CF_OK) return state;
    }
    cf_isize n = read(fd, dst->data + dst->len, dst->cap - dst->len);
    if(n < 0)
    {
      if(errno == EINTR) continue;
      return CF_ERR_IO_READ;
    }
    if(n == 0) return CF_OK;
    dst->len += n;
  }
}

cf_status cf_io_write_fd(int fd, cf_bytes src)
{
  if(fd < 0) return CF_ERR_IO_WRITE;
  if(src.len != 0 && src.data == CF_NULL) return CF_ERR_NULL;
  if(src.elem_size != 0 && src.len > CF_USIZE_MAX / src.elem_size)
    return CF_ERR_OVERFLOW;
  cf_usize offset = 0, len = src.len * src.elem_size; 
  while (offset < len)
  {
    cf_isize n = write(fd, (cf_u8 *) src.data + offset, len - offset);
    if(n < 0)
    {
      if(errno == EINTR) continue;
      return CF_ERR_IO_WRITE;
    }
    if(n == 0) return CF_ERR_IO_WRITE;
    offset += n; 
  }
  return CF_OK;
}

cf_status cf_io_read_file(cf_buffer *dst, const char *path)
{
  if(dst == CF_NULL || path == CF_NULL) return CF_ERR_NULL;
  if(dst->data != CF_NULL && cf_buffer_is_valid(dst) == CF_FALSE) return CF_ERR_STATE;

  if(dst->data == CF_NULL)
  {
    cf_status state = cf_buffer_init(dst, CF_IO_RESERVE_SIZE);
    if(state != CF_OK) return state;
  }

  int fd = open(path, O_RDONLY);
  if(fd < 0) return CF_ERR_IO_OPEN;

  cf_status state = cf_io_read_fd(dst, fd);
  if(state != CF_OK) 
  {
    if(close(fd) != 0) 
      return CF_ERR_IO_CLOSE;
    return state;
  }

  if(close(fd) != 0) return CF_ERR_IO_CLOSE;

  return CF_OK;
}

cf_status cf_io_write_file(const char *path, cf_bytes src)
{
  if(path == CF_NULL) return CF_ERR_NULL;
  if(src.len != 0 && src.data == CF_NULL) return CF_ERR_NULL;
  if(src.elem_size != 0 && src.len > CF_USIZE_MAX / src.elem_size) return CF_ERR_OVERFLOW;

  int fd = open(path, O_WRONLY | O_TRUNC | O_CREAT, 0644);
  if(fd < 0) return CF_ERR_IO_OPEN;

  cf_status state = cf_io_write_fd(fd, src);
  if(state != CF_OK) 
  {
    if(close(fd) != 0) 
      return CF_ERR_IO_CLOSE;
    return state;
  }

  if(close(fd) != 0) return CF_ERR_IO_CLOSE;

  return CF_OK;
}

cf_status cf_io_append_file(const char *path, cf_bytes src)
{
  if(path == CF_NULL) return CF_ERR_NULL;
  if(src.len != 0 && src.data == CF_NULL) return CF_ERR_NULL;
  if(src.elem_size != 0 && src.len > CF_USIZE_MAX / src.elem_size) return CF_ERR_OVERFLOW;

  int fd = open(path, O_WRONLY | O_APPEND | O_CREAT, 0644);
  if(fd < 0) return CF_ERR_IO_OPEN;

  cf_status state = cf_io_write_fd(fd, src);
  if(state != CF_OK) 
  {
    if(close(fd) != 0) 
      return CF_ERR_IO_CLOSE;
    return state;
  }

  if(close(fd) != 0) return CF_ERR_IO_CLOSE;

  return CF_OK;
}

cf_status cf_io_read_text(cf_string *dst, const char *path)
{
  if(dst == CF_NULL || path == CF_NULL) return CF_ERR_NULL;
  if(dst->data != CF_NULL && cf_string_is_valid(dst) == CF_FALSE) return CF_ERR_STATE;

  if(dst->data == CF_NULL)
  {
    cf_status state = cf_buffer_init(dst, CF_IO_RESERVE_SIZE);
    if(state != CF_OK) return state;
  }

  int fd = open(path, O_RDONLY);
  if(fd < 0) return CF_ERR_IO_OPEN;

  cf_status state = cf_io_read_fd(dst, fd);
  if(state != CF_OK) 
  {
    if(close(fd) != 0) 
      return CF_ERR_IO_CLOSE;
    return state;
  }
  dst->data[dst->len] = '\0';

  if(close(fd) != 0) return CF_ERR_IO_CLOSE;

  return CF_OK;
}

cf_status cf_io_write_text(const char *path, cf_string *src)
{
  if(path == CF_NULL || src == CF_NULL) return CF_ERR_NULL;
  if(cf_string_is_valid(src) == CF_FALSE) return CF_ERR_STATE;

  int fd = open(path, O_WRONLY | O_TRUNC | O_CREAT, 0644);
  if(fd < 0) return CF_ERR_IO_OPEN;

  cf_status state = cf_io_write_fd(fd, (cf_bytes) {.data = src->data, .elem_size = sizeof (char), .len = src->len});
  if(state != CF_OK) 
  {
    if(close(fd) != 0) 
      return CF_ERR_IO_CLOSE;
    return state;
  }

  if(close(fd) != 0) return CF_ERR_IO_CLOSE;

  return CF_OK;
}

cf_status cf_io_append_text(const char *path, cf_string *src)
{
  if(path == CF_NULL || src == CF_NULL) return CF_ERR_NULL;
  if(cf_string_is_valid(src) == CF_FALSE) return CF_ERR_STATE;

  int fd = open(path, O_WRONLY | O_APPEND | O_CREAT, 0644);
  if(fd < 0) return CF_ERR_IO_OPEN;

  cf_status state = cf_io_write_fd(fd, (cf_bytes) {.data = src->data, .elem_size = sizeof (char), .len = src->len});
  if(state != CF_OK) 
  {
    if(close(fd) != 0) 
      return CF_ERR_IO_CLOSE;
    return state;
  }

  if(close(fd) != 0) return CF_ERR_IO_CLOSE;

  return CF_OK;
}
