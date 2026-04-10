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

/*
* Return Initialized cf_bytes as (CF_NULL,0)
*/
cf_bytes cf_bytes_empty(void);

/*
* Return Initialized cf_bytes_mut as (CF_NULL,0)
*/
cf_bytes_mut cf_bytes_mut_empty(void);

/*
* Return Initialized cf_buffer as (CF_NULL,0,0)
*/
cf_buffer cf_buffer_empty(void);

/*
* Return Already existing memory data as cf_bytes for read-only data
*/
cf_bytes cf_bytes_from(const cf_u8 *data, cf_usize len);

/*
* Return Already existing memory data as cf_bytes_mut for writable data
*/
cf_bytes_mut cf_bytes_mut_from(cf_u8 *data, cf_usize len);

#endif // CF_MEMORY_H