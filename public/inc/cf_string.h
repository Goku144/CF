#if !defined(CF_STRING_H)
#define CF_STRING_H

#include "cf_types.h"
#include "cf_status.h"

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

/** Returns an empty (null, zero-length) cf_str. */
cf_str cf_str_empty(void);

/** Returns an empty (null, zero-length, zero-capacity) cf_string. */
cf_string cf_string_empty(void);

/** Constructs a cf_str from a pointer and length. Does not copy data. */
cf_str cf_str_from(const char *data, cf_usize len);

/** Returns CF_TRUE if the cf_str is in a consistent state (non-null if len > 0). */
cf_bool cf_str_is_valid(cf_str str);

/**
 * Returns CF_TRUE if the cf_string is in a consistent state
 * (non-null if len/cap > 0, len <= cap, null terminator present).
 */
cf_bool cf_string_is_valid(cf_string str);

/** Returns CF_TRUE if the cf_str has zero length. */
cf_bool cf_str_is_empty(cf_str str);

/** Returns CF_TRUE if the cf_string has zero length. */
cf_bool cf_string_is_empty(cf_string str);

/**
 * Compares two cf_str values for equality.
 * @param lhs     Left-hand string.
 * @param rhs     Right-hand string.
 * @param out_eq  Output boolean set to CF_TRUE if equal, CF_FALSE otherwise.
 * @return CF_ERR_NULL if out_eq is null; CF_ERR_STATE if either is invalid; CF_OK otherwise.
 */
cf_status cf_str_eq(cf_str lhs, cf_str rhs, cf_bool *out_eq);

/**
 * Returns a sub-slice of a cf_str without copying.
 * @param src    Source string slice.
 * @param offset Start offset within src.
 * @param len    Length of the sub-slice.
 * @param dst    Output slice pointing into src.
 * @return CF_ERR_NULL if dst is null; CF_ERR_STATE if src is invalid; CF_ERR_BOUNDS if out of range; CF_OK otherwise.
 */
cf_status cf_str_slice(cf_str src, cf_usize offset, cf_usize len, cf_str *dst);

/**
 * Allocates a cf_string with at least the given initial capacity.
 * @param str  Output string to initialize.
 * @param cap  Desired initial capacity in characters (excluding null terminator).
 * @return CF_ERR_NULL if str is null; CF_ERR_OOM on allocation failure; CF_OK otherwise.
 */
cf_status cf_string_init(cf_string *str, cf_usize cap);

/**
 * Ensures the string has at least min_cap capacity, reallocating if needed.
 * Always maintains a null terminator at data[len].
 * @param str      String to grow.
 * @param min_cap  Minimum required capacity (excluding null terminator).
 * @return CF_ERR_NULL if str is null; CF_ERR_STATE if invalid; CF_ERR_OOM on failure; CF_OK otherwise.
 */
cf_status cf_string_reserve(cf_string *str, cf_usize min_cap);

/**
 * Resets the string length to zero and writes a null terminator, without freeing memory.
 * @param str  String to clear.
 * @return CF_ERR_NULL if str is null; CF_ERR_STATE if invalid; CF_OK otherwise.
 */
cf_status cf_string_clear(cf_string *str);

/**
 * Frees the string's memory and resets it to an empty state.
 * @param str  String to destroy. Safe to call on a null pointer.
 */
void cf_string_destroy(cf_string *str);

/**
 * Appends a single character to the string, growing it if necessary.
 * Maintains null termination.
 * @param str  Target string.
 * @param s    Character to append.
 * @return CF_ERR_NULL if str is null; CF_ERR_STATE if invalid; CF_ERR_OOM on failure; CF_OK otherwise.
 */
cf_status cf_string_append_char(cf_string *str, char s);

/**
 * Appends a cf_str slice to the string, growing it if necessary.
 * Maintains null termination.
 * @param str  Target string.
 * @param s    Source string slice to append.
 * @return CF_ERR_NULL if str is null; CF_ERR_STATE if either is invalid; CF_ERR_OOM on failure; CF_OK otherwise.
 */
cf_status cf_string_append_str(cf_string *str, cf_str s);

#endif // CF_STRING_H
