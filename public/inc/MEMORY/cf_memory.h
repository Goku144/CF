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

#if !defined(CF_MEMOEY_H)
#define CF_MEMOEY_H

#include "RUNTIME/cf_types.h"
#include "RUNTIME/cf_status.h"
#include "ALLOCATOR/cf_alloc.h"

/**
 * not allocated by me
 * offer view scop
 * writable
 * not freed by me
 */
typedef struct cf_bytes
{
  cf_u8 *data;
  cf_usize len;
} cf_bytes;

/**
 * allocated by me
 * can chose allocation
 * freed by me
 */
typedef struct cf_buffer
{
  cf_u8 *data;
  cf_usize len;
  cf_usize cap;
  cf_alloc allocator;
} cf_buffer;

/* ------------------------------------------------------------------ */
/* Construction                                                        */
/* ------------------------------------------------------------------ */

cf_bytes cf_bytes_create_empty(void);

cf_buffer cf_buffer_create_empty(void);

/* ------------------------------------------------------------------ */
/* Validation                                                          */
/* ------------------------------------------------------------------ */

cf_bool cf_bytes_is_valid(cf_bytes *bytes);

cf_bool cf_buffer_is_valid(cf_buffer *buffer);

/* ------------------------------------------------------------------ */
/* Emptiness                                                           */
/* ------------------------------------------------------------------ */

cf_bool cf_bytes_is_empty(cf_bytes *bytes);

cf_bool cf_buffer_is_empty(cf_buffer *buffer);

/* ------------------------------------------------------------------ */
/* Equality                                                            */
/* ------------------------------------------------------------------ */

cf_bool cf_bytes_is_eq(cf_bytes *b1, cf_bytes *b2);

cf_bool cf_buffer_is_eq(cf_buffer *buf1, cf_buffer *buf2);

/* ------------------------------------------------------------------ */
/* Slicing                                                             */
/* ------------------------------------------------------------------ */

cf_status cf_bytes_slice(cf_bytes *dst_bytes, cf_bytes *src_bytes, cf_usize index, cf_usize size);

cf_status cf_buffer_slice(cf_bytes *dst_bytes, cf_buffer *src_buffer, cf_usize index, cf_usize size);
/* ------------------------------------------------------------------ */
/* Fill / Zero                                                         */
/* ------------------------------------------------------------------ */

cf_status cf_bytes_fill(cf_bytes *bytes, cf_u8 fill, cf_usize size);

cf_status cf_buffer_fill(cf_buffer *buffer, cf_u8 fill, cf_usize size);

/* ------------------------------------------------------------------ */
/* Buffer lifecycle                                                    */
/* ------------------------------------------------------------------ */

cf_status cf_buffer_init(cf_buffer *buffer, cf_alloc *allocator, cf_usize capacity);

cf_status cf_buffer_reserve(cf_buffer *buffer, cf_usize size);

void cf_buffer_clear(cf_buffer *buffer);

void cf_buffer_destroy(cf_buffer *buffer);

#endif /* CF_MEMOEY_H */
