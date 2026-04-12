#if !defined(CF_STRING_H)
#define CF_STRING_H

#include "cf_types.h"
#include "cf_status.h"
#include "cf_alloc.h"

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
    const cf_allocator *allocator;
} cf_string;


/* ------------------------------------------------------------------ */
/* Construction                                                        */
/* ------------------------------------------------------------------ */

/** Returns an empty (null, zero-length) cf_str. */
cf_str cf_str_empty(void);

/** Returns an empty (null, zero-length, zero-capacity) cf_string. */
cf_string cf_string_empty(void);

/** Constructs a cf_str from a pointer and length. Does not copy data. */
cf_str cf_str_from(const char *data, cf_usize len);


/* ------------------------------------------------------------------ */
/* Validation                                                          */
/* ------------------------------------------------------------------ */

/** Returns CF_TRUE if the cf_str is in a consistent state (non-null if len > 0). */
cf_bool cf_str_is_valid(cf_str str);

/**
 * Returns CF_TRUE if the cf_string is in a consistent state
 * (non-null if len/cap > 0, len <= cap, null terminator present).
 */
cf_bool cf_string_is_valid(cf_string str);


/* ------------------------------------------------------------------ */
/* Emptiness                                                           */
/* ------------------------------------------------------------------ */

/** Returns CF_TRUE if the cf_str has zero length. */
cf_bool cf_str_is_empty(cf_str str);

/** Returns CF_TRUE if the cf_string has zero length. */
cf_bool cf_string_is_empty(cf_string str);


/* ------------------------------------------------------------------ */
/* Equality / comparison                                               */
/* ------------------------------------------------------------------ */

/**
 * Compares two cf_str values for equality (case-sensitive).
 * @param s1      Left-hand string.
 * @param s2      Right-hand string.
 * @param out_eq  Output boolean set to CF_TRUE if equal, CF_FALSE otherwise.
 * @return CF_ERR_NULL if out_eq is null; CF_ERR_STATE if either is invalid; CF_OK otherwise.
 */
cf_status cf_str_eq(cf_str s1, cf_str s2, cf_bool *out_eq);

/**
 * Compares two cf_str values for equality, ignoring ASCII letter case.
 * Non-alphabetic characters are compared as-is.
 * @param s1      Left-hand string.
 * @param s2      Right-hand string.
 * @param out_eq  Output boolean set to CF_TRUE if equal, CF_FALSE otherwise.
 * @return CF_ERR_NULL if out_eq is null; CF_ERR_STATE if either is invalid; CF_OK otherwise.
 */
cf_status cf_str_eq_ignore_case(cf_str s1, cf_str s2, cf_bool *out_eq);


/* ------------------------------------------------------------------ */
/* Prefix / suffix checks                                              */
/* ------------------------------------------------------------------ */

/**
 * Checks whether a cf_str starts with the given prefix.
 * @param s       String to check.
 * @param prefix  Prefix to look for.
 * @param out     Set to CF_TRUE if s starts with prefix, CF_FALSE otherwise.
 * @return CF_ERR_NULL if out is null; CF_ERR_STATE if either is invalid; CF_OK otherwise.
 */
cf_status cf_str_starts_with(cf_str s, cf_str prefix, cf_bool *out);

/**
 * Checks whether a cf_str ends with the given suffix.
 * @param s       String to check.
 * @param suffix  Suffix to look for.
 * @param out     Set to CF_TRUE if s ends with suffix, CF_FALSE otherwise.
 * @return CF_ERR_NULL if out is null; CF_ERR_STATE if either is invalid; CF_OK otherwise.
 */
cf_status cf_str_ends_with(cf_str s, cf_str suffix, cf_bool *out);


/* ------------------------------------------------------------------ */
/* Slicing / indexing                                                  */
/* ------------------------------------------------------------------ */

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
 * Returns the character at the given index in a cf_str.
 * @param s       Source string slice.
 * @param index   Zero-based index.
 * @param out_ch  Output character.
 * @return CF_ERR_NULL if out_ch is null; CF_ERR_STATE if slice is invalid; CF_ERR_BOUNDS if index >= len; CF_OK otherwise.
 */
cf_status cf_str_at(cf_str s, cf_usize index, char *out_ch);

/**
 * Returns the character at the given index in a cf_string.
 * @param str     Source string.
 * @param index   Zero-based index.
 * @param out_ch  Output character.
 * @return CF_ERR_NULL if out_ch is null; CF_ERR_STATE if string is invalid; CF_ERR_BOUNDS if index >= len; CF_OK otherwise.
 */
cf_status cf_string_at(cf_string str, cf_usize index, char *out_ch);


/* ------------------------------------------------------------------ */
/* Search                                                              */
/* ------------------------------------------------------------------ */

/**
 * Searches for the first occurrence of a character in a cf_str.
 * @param s          String to search.
 * @param ch         Character to find.
 * @param out_index  Set to the zero-based index of the first match if found.
 * @param out_found  Set to CF_TRUE if the character was found, CF_FALSE otherwise.
 * @return CF_ERR_NULL if out_index or out_found is null; CF_ERR_STATE if s is invalid; CF_OK otherwise.
 */
cf_status cf_str_find_char(cf_str s, char ch, cf_usize *out_index, cf_bool *out_found);

/**
 * Searches for the first occurrence of a substring in a cf_str.
 * @param s          String to search.
 * @param needle     Substring to find.
 * @param out_index  Set to the zero-based start index of the first match if found.
 * @param out_found  Set to CF_TRUE if the substring was found, CF_FALSE otherwise.
 * @return CF_ERR_NULL if out_index or out_found is null; CF_ERR_STATE if either is invalid; CF_OK otherwise.
 */
cf_status cf_str_find_str(cf_str s, cf_str needle, cf_usize *out_index, cf_bool *out_found);


/* ------------------------------------------------------------------ */
/* Trim                                                                */
/* ------------------------------------------------------------------ */

/**
 * Returns a view of s with leading whitespace removed. Does not copy data.
 * Whitespace is defined as space, tab, carriage return, and newline.
 * @param s    Source string slice.
 * @param out  Output slice with leading whitespace stripped.
 * @return CF_ERR_NULL if out is null; CF_ERR_STATE if s is invalid; CF_OK otherwise.
 */
cf_status cf_str_trim_left(cf_str s, cf_str *out);

/**
 * Returns a view of s with trailing whitespace removed. Does not copy data.
 * Whitespace is defined as space, tab, carriage return, and newline.
 * @param s    Source string slice.
 * @param out  Output slice with trailing whitespace stripped.
 * @return CF_ERR_NULL if out is null; CF_ERR_STATE if s is invalid; CF_OK otherwise.
 */
cf_status cf_str_trim_right(cf_str s, cf_str *out);

/**
 * Returns a view of s with both leading and trailing whitespace removed. Does not copy data.
 * Whitespace is defined as space, tab, carriage return, and newline.
 * @param s    Source string slice.
 * @param out  Output slice with surrounding whitespace stripped.
 * @return CF_ERR_NULL if out is null; CF_ERR_STATE if s is invalid; CF_OK otherwise.
 */
cf_status cf_str_trim(cf_str s, cf_str *out);


/* ------------------------------------------------------------------ */
/* Split                                                               */
/* ------------------------------------------------------------------ */

/**
 * Splits a cf_str on the first occurrence of a separator character.
 * If the separator is found, left receives the part before it and right
 * receives the part after it (the separator itself is not included in either).
 * If not found, left and right are both set to empty.
 * @param s          Source string slice.
 * @param sep        Separator character to split on.
 * @param left       Output slice for the portion before the separator.
 * @param right      Output slice for the portion after the separator.
 * @param out_found  Set to CF_TRUE if the separator was found, CF_FALSE otherwise.
 * @return CF_ERR_NULL if left, right, or out_found is null; CF_ERR_STATE if s is invalid; CF_OK otherwise.
 */
cf_status cf_str_split_once_char(cf_str s, char sep, cf_str *left, cf_str *right, cf_bool *out_found);


/* ------------------------------------------------------------------ */
/* String lifecycle                                                    */
/* ------------------------------------------------------------------ */

cf_status cf_string_init_ex(cf_string *str, cf_usize cap, const cf_allocator *allocator);

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


/* ------------------------------------------------------------------ */
/* String append / set                                                 */
/* ------------------------------------------------------------------ */

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

/**
 * Clears the string and replaces its content with the given cf_str slice.
 * Maintains null termination.
 * @param str  Target string.
 * @param s    Source string slice to copy in.
 * @return CF_ERR_NULL if str is null; CF_ERR_STATE if either is invalid; CF_ERR_OOM on failure; CF_OK otherwise.
 */
cf_status cf_string_set_str(cf_string *str, cf_str s);


/* ------------------------------------------------------------------ */
/* String views                                                        */
/* ------------------------------------------------------------------ */

/** Returns a cf_str view of the string's current content. Returns empty if string is invalid. */
cf_str cf_string_as_str(cf_string str);


/* ------------------------------------------------------------------ */
/* String truncate                                                     */
/* ------------------------------------------------------------------ */

/**
 * Truncates the string to new_len, which must be <= current len.
 * Maintains null termination.
 * @param str      Target string.
 * @param new_len  New length; must not exceed str->len.
 * @return CF_ERR_NULL if str is null; CF_ERR_STATE if invalid; CF_ERR_BOUNDS if new_len > len; CF_OK otherwise.
 */
cf_status cf_string_truncate(cf_string *str, cf_usize new_len);

#endif // CF_STRING_H