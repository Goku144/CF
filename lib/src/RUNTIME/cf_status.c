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

#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

#include <stdio.h>
#include <string.h>

typedef struct cf_status_desc
{
  cf_status flag;
  const char *name;
  const char *readable;
} cf_status_desc;

static const cf_status_desc CF_STATUS_DESC[] =
{
  {CF_ERR_NULL, "CF_ERR_NULL", "A required pointer argument was NULL."},
  {CF_ERR_INVALID, "CF_ERR_INVALID", "An argument value was invalid for this operation."},
  {CF_ERR_STATE, "CF_ERR_STATE", "The object or subsystem is in the wrong state."},
  {CF_ERR_BOUNDS, "CF_ERR_BOUNDS", "A size, offset, or index was out of allowed bounds."},
  {CF_ERR_OVERFLOW, "CF_ERR_OVERFLOW", "A numeric operation overflowed the supported range."},
  {CF_ERR_OOM, "CF_ERR_OOM", "Memory allocation or reservation failed."},
  {CF_ERR_IO, "CF_ERR_IO", "An input or output operation failed."},
  {CF_ERR_PARSE, "CF_ERR_PARSE", "Input data could not be parsed or validated."},
  {CF_ERR_UNSUPPORTED, "CF_ERR_UNSUPPORTED", "The requested behavior or platform capability is not supported."},
  {CF_ERR_SECURITY, "CF_ERR_SECURITY", "A security, integrity, or authentication check failed."},
  {CF_ERR_INTERNAL, "CF_ERR_INTERNAL", "An unexpected internal failure occurred."},
  {CF_ERR_NOT_FOUND, "CF_ERR_NOT_FOUND", "A requested file or other resource was not found."},
  {CF_ERR_UNDEFINED, "CF_ERR_UNDEFINED", "The status code is undefined or unmapped."},
  {CF_ERR_IO_OPEN, "CF_ERR_IO_OPEN", "Opening a file failed."},
  {CF_ERR_IO_READ, "CF_ERR_IO_READ", "Reading from a file descriptor failed."},
  {CF_ERR_IO_WRITE, "CF_ERR_IO_WRITE", "Writing to a file descriptor failed."},
  {CF_ERR_IO_CLOSE, "CF_ERR_IO_CLOSE", "Closing a file descriptor failed."},
  {CF_ERR_IO_METADATA, "CF_ERR_IO_METADATA", "Querying filesystem metadata failed."},
  {CF_ERR_TIME_SLEEP, "CF_ERR_TIME_SLEEP", "Sleeping for the requested duration failed."},
  {CF_ERR_TIME_CLOCK, "CF_ERR_TIME_CLOCK", "Reading the system clock failed."},
  {CF_ERR_INVALID_PADDING, "CF_ERR_INVALID_PADDING", "PKCS#7 padding validation failed."},
  {CF_ERR_RANDOM, "CF_ERR_RANDOM", "Random byte generation failed."},
  {CF_ERR_CUDA, "CF_ERR_CUDA", "A CUDA or GPU operation failed."},
  {CF_ERR_CUDA_DRIVER, "CF_ERR_CUDA_DRIVER", "The CUDA driver API reported a failure."},
  {CF_ERR_CUDA_RUNTIME, "CF_ERR_CUDA_RUNTIME", "The CUDA runtime API reported a failure."},
  {CF_ERR_CUDA_DEVICE, "CF_ERR_CUDA_DEVICE", "CUDA device selection or capability validation failed."},
  {CF_ERR_CUDA_MEMORY, "CF_ERR_CUDA_MEMORY", "CUDA device memory allocation or ownership failed."},
  {CF_ERR_CUDA_COPY, "CF_ERR_CUDA_COPY", "Copying data to or from CUDA device memory failed."},
  {CF_ERR_CUDA_LAUNCH, "CF_ERR_CUDA_LAUNCH", "Launching a CUDA kernel failed."},
  {CF_ERR_CUDA_SYNC, "CF_ERR_CUDA_SYNC", "Synchronizing CUDA work failed."},
};

static const char *cf_status_compose(cf_status state, cf_bool use_readable)
{
  static char buffers[4][512];
  static cf_usize next_buffer = 0;
  char *buffer = buffers[next_buffer];
  cf_usize used = 0;
  cf_status unknown_bits = state;

  next_buffer = (next_buffer + 1) % 4;

  if(state == CF_OK)
    return use_readable ? "Operation completed successfully." : "CF_OK";

  buffer[0] = '\0';

  for(cf_usize i = 0; i < sizeof(CF_STATUS_DESC) / sizeof(CF_STATUS_DESC[0]); ++i)
  {
    if((state & CF_STATUS_DESC[i].flag) == 0) continue;

    const char *piece = use_readable ? CF_STATUS_DESC[i].readable : CF_STATUS_DESC[i].name;
    const char *separator = use_readable ? " | " : "|";
    int written = snprintf
    (
      buffer + used,
      sizeof(buffers[0]) - used,
      "%s%s",
      used == 0 ? "" : separator,
      piece
    );

    if(written < 0) return use_readable ? "The status code is unknown." : "CF_ERR_UNKNOWN";
    if((cf_usize)written >= sizeof(buffers[0]) - used) break;
    used += (cf_usize)written;
    unknown_bits &= ~CF_STATUS_DESC[i].flag;
  }

  if(unknown_bits != 0 || used == 0)
  {
    (void)snprintf
    (
      buffer + used,
      sizeof(buffers[0]) - used,
      "%s%s(0x%X)",
      used == 0 ? "" : (use_readable ? " | " : "|"),
      use_readable ? "Unknown status bits " : "CF_ERR_UNKNOWN",
      (unsigned int)unknown_bits
    );
  }

  return buffer;
}

const char *cf_status_as_char(cf_status state)
{
  return cf_status_compose(state, CF_FALSE);
}
