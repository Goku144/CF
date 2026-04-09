#if !defined(CF_MEMORY_H)
#define CF_MEMORY_H

#include "cf_types.h"

/* 
 * borrowed
 * read-only
 * not freed by me 
 * */
typedef struct cf_bytes
{
  const cf_u8 *data;
  cf_usize len;
} cf_bytes;

/* 
 * borrowed
 * writable
 * not freed by me 
 * */
typedef struct cf_bytes_mut
{
  cf_u8 *data;
  cf_usize len;
} cf_bytes_mut;

/* 
 * owned
 * readable
 * writable
 * resizable
 * freed by me
 * */
typedef struct cf_buffer
{
  cf_u8 *data;
  cf_usize len;
  cf_usize cap;
} cf_buffer;

#endif // CF_MEMORY_H