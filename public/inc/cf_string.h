#if !defined(CF_STRING_H)
#define CF_STRING_H

#include "cf_types.h"

/*
 * borrowed
 * read-only
 * text
 * not freed by me
 */
typedef struct cf_str
{
    const char *data;
    cf_usize len;
} cf_str;

/*
 * owned
 * readable
 * writable
 * resizable
 * text
 * freed by me
 */
typedef struct cf_string
{
    char *data;
    cf_usize len;
    cf_usize cap;
} cf_string;

/*
* Return Initialized cf_str as (CF_NULL,0)
*/
cf_str cf_str_empty(void);

/*
* Return Initialized cf_string as (CF_NULL,0,0)
*/
cf_string cf_string_empty(void);

/*
* Return Already existing memory data as cf_str for writable data
*/
cf_str cf_str_from(const char *data, cf_usize len);

#endif // CF_STRING_H
