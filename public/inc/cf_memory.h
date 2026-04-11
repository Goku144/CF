#if !defined(CF_MEMORY_H)
#define CF_MEMORY_H

#include "cf_types.h"
#include "cf_status.h"

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


/** Returns an empty (null, zero-length) cf_bytes. */
cf_bytes cf_bytes_empty(void);

/** Returns an empty (null, zero-length) cf_bytes_mut. */
cf_bytes_mut cf_bytes_mut_empty(void);

/** Returns an empty (null, zero-length, zero-capacity) cf_buffer. */
cf_buffer cf_buffer_empty(void);

/** Constructs a cf_bytes from a pointer and length. Does not copy data. */
cf_bytes cf_bytes_from(const cf_u8 *data, cf_usize len);

/** Constructs a cf_bytes_mut from a pointer and length. Does not copy data. */
cf_bytes_mut cf_bytes_mut_from(cf_u8 *data, cf_usize len);

/** Returns CF_TRUE if the cf_bytes is in a consistent state (non-null if len > 0). */
cf_bool cf_bytes_is_valid(cf_bytes bytes);

/** Returns CF_TRUE if the cf_bytes_mut is in a consistent state (non-null if len > 0). */
cf_bool cf_bytes_mut_is_valid(cf_bytes_mut bytes);

/** Returns CF_TRUE if the cf_buffer is in a consistent state (non-null if len/cap > 0, len <= cap). */
cf_bool cf_buffer_is_valid(cf_buffer buffer);

/** Returns CF_TRUE if the cf_bytes has zero length. */
cf_bool cf_bytes_is_empty(cf_bytes bytes);

/** Returns CF_TRUE if the cf_bytes_mut has zero length. */
cf_bool cf_bytes_mut_is_empty(cf_bytes_mut bytes);

/** Returns CF_TRUE if the cf_buffer has zero length. */
cf_bool cf_buffer_is_empty(cf_buffer buffer);

/**
 * Compares two cf_bytes for equality.
 * @param b1     First byte slice.
 * @param b2     Second byte slice.
 * @param out    Output boolean set to CF_TRUE if equal, CF_FALSE otherwise.
 * @return CF_ERR_NULL if out is null; CF_ERR_STATE if either slice is invalid; CF_OK otherwise.
 */
cf_status cf_bytes_eq(cf_bytes b1, cf_bytes b2, cf_bool *out);

/**
 * Zeroes out all bytes in a mutable byte slice.
 * @param bytes  The mutable slice to zero.
 * @return CF_ERR_STATE if invalid; CF_OK otherwise.
 */
cf_status cf_bytes_mut_zero(cf_bytes_mut bytes);

/**
 * Returns a sub-slice of a cf_bytes without copying.
 * @param src    Source byte slice.
 * @param offset Start offset within src.
 * @param len    Length of the sub-slice.
 * @param dst    Output slice pointing into src.
 * @return CF_ERR_NULL if dst is null; CF_ERR_STATE if src is invalid; CF_ERR_BOUNDS if out of range; CF_OK otherwise.
 */
cf_status cf_bytes_slice(cf_bytes src, cf_usize offset, cf_usize len, cf_bytes *dst);

/**
 * Allocates a cf_buffer with at least the given initial capacity.
 * @param buffer  Output buffer to initialize.
 * @param cap     Desired initial capacity in bytes.
 * @return CF_ERR_NULL if buffer is null; CF_ERR_OOM on allocation failure; CF_OK otherwise.
 */
cf_status cf_buffer_init(cf_buffer *buffer, cf_usize cap);

/**
 * Ensures the buffer has at least min_cap capacity, reallocating if needed.
 * @param buffer   Buffer to grow.
 * @param min_cap  Minimum required capacity.
 * @return CF_ERR_NULL if buffer is null; CF_ERR_STATE if invalid; CF_ERR_OOM on failure; CF_OK otherwise.
 */
cf_status cf_buffer_reserve(cf_buffer *buffer, cf_usize min_cap);

/**
 * Resets the buffer length to zero without freeing memory.
 * @param buffer  Buffer to clear.
 * @return CF_ERR_NULL if buffer is null; CF_ERR_STATE if invalid; CF_OK otherwise.
 */
cf_status cf_buffer_clear(cf_buffer *buffer);

/**
 * Frees the buffer's memory and resets it to an empty state.
 * @param buffer  Buffer to destroy. Safe to call on a null pointer.
 */
void cf_buffer_destroy(cf_buffer *buffer);

/**
 * Appends a single byte to the buffer, growing it if necessary.
 * @param buffer  Target buffer.
 * @param byte    Byte value to append.
 * @return CF_ERR_NULL if buffer is null; CF_ERR_STATE if invalid; CF_ERR_OOM on failure; CF_OK otherwise.
 */
cf_status cf_buffer_append_byte(cf_buffer *buffer, cf_u8 byte);

/**
 * Appends a cf_bytes slice to the buffer, growing it if necessary.
 * @param buffer  Target buffer.
 * @param bytes   Source byte slice to append.
 * @return CF_ERR_NULL if buffer is null; CF_ERR_STATE if either is invalid; CF_ERR_OOM on failure; CF_OK otherwise.
 */
cf_status cf_buffer_append_bytes(cf_buffer *buffer, cf_bytes bytes);

/** Returns a cf_bytes view of the buffer's current content. Returns empty if buffer is invalid. */
cf_bytes cf_buffer_as_bytes(cf_buffer buffer);

/** Returns a cf_bytes_mut view of the buffer's current content. Returns empty if buffer or pointer is null/invalid. */
cf_bytes_mut cf_buffer_as_bytes_mut(cf_buffer *buffer);

/**
 * Clears the buffer and replaces its content with the given byte slice.
 * @param buffer  Target buffer.
 * @param bytes   Source byte slice to copy in.
 * @return CF_ERR_NULL if buffer is null; CF_ERR_STATE if either is invalid; CF_ERR_OOM on failure; CF_OK otherwise.
 */
cf_status cf_buffer_set_bytes(cf_buffer *buffer, cf_bytes bytes);

/**
 * Fills every byte in the slice with the given value.
 * @param bytes  Target mutable slice.
 * @param value  Byte value to fill with.
 * @return CF_ERR_STATE if slice is invalid; CF_OK otherwise.
 */
cf_status cf_bytes_mut_fill(cf_bytes_mut bytes, cf_u8 value);

/**
 * Fills every byte in the buffer's current content with the given value.
 * @param buffer  Target buffer.
 * @param value   Byte value to fill with.
 * @return CF_ERR_NULL if buffer is null; CF_ERR_STATE if invalid; CF_OK otherwise.
 */
cf_status cf_buffer_fill(cf_buffer *buffer, cf_u8 value);

/**
 * Truncates the buffer to new_len, which must be <= current len.
 * @param buffer   Target buffer.
 * @param new_len  New length; must not exceed buffer->len.
 * @return CF_ERR_NULL if buffer is null; CF_ERR_STATE if invalid; CF_ERR_BOUNDS if new_len > len; CF_OK otherwise.
 */
cf_status cf_buffer_truncate(cf_buffer *buffer, cf_usize new_len);

#endif // CF_MEMORY_H