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

#endif // CF_STRING_H
