# Implemented API Reference

This page documents the functions that currently have real implementations.
Placeholder modules are listed in [Project Hierarchy](project-hierarchy.md).

The framework uses:

- `cf_status` for recoverable API errors.
- `CF_OK` for success.
- `CF_NULL` for null pointers.
- `cf_bool`, `CF_TRUE`, and `CF_FALSE` for boolean APIs.
- `cf_usize` and `cf_isize` for framework sizes and signed offsets.

## Runtime

### `cf_types_type_size`

Header: `public/inc/RUNTIME/cf_types.h`

```c
cf_native_group cf_types_type_size(cf_usize type_size);
```

Classifies a byte width into one of the framework native-size groups:

```text
1, 2, 4, 8, 16 bytes, or CF_NATIVE_UNKNOWN
```

Critical points:

- It is an exact match classifier.
- It does not inspect a real C type, only a byte count.
- It is useful for diagnostics and generic container metadata.

### `cf_types_as_char`

```c
const char *cf_types_as_char(cf_usize type_size);
```

Returns a stable descriptive string for a native-size group.

Critical points:

- Returned strings are static and must not be freed.
- Unknown sizes return a readable unknown-size explanation.

### `cf_status_as_char`

Header: `public/inc/RUNTIME/cf_status.h`

```c
const char *cf_status_as_char(cf_status state);
```

Converts a status code or combined status bitset into symbolic text.

Critical points:

- Returns a pointer to a static rotating buffer.
- Do not free or modify the result.
- Combined flags are joined with `|`.

### `cf_log_level_as_char`

Header: `public/inc/RUNTIME/cf_log.h`

```c
const char *cf_log_level_as_char(cf_log_level level);
```

Returns a short symbolic log level name.

### `cf_log_set_level`

```c
void cf_log_set_level(cf_log_level level);
```

Sets the process-wide minimum log level.

Critical points:

- Invalid levels are ignored.
- `CF_LOG_LEVEL_OFF` suppresses all log output.

### `cf_log_get_level`

```c
cf_log_level cf_log_get_level(void);
```

Returns the current process-wide minimum log level.

### `cf_log_should_write`

```c
cf_bool cf_log_should_write(cf_log_level level);
```

Checks whether a log message at `level` would be emitted.

### `cf_log_write`

```c
void cf_log_write(cf_log_level level, const char *file, int line, const char *fmt, ...);
```

Writes a formatted log message to `stderr`.

Critical points:

- Used by `CF_LOG_TRACE`, `CF_LOG_DEBUG`, `CF_LOG_INFO`, `CF_LOG_WARN`,
  `CF_LOG_ERROR`, and `CF_LOG_FATAL`.
- Does nothing if `fmt == CF_NULL`.
- Does nothing if the level is filtered out.

### `cf_time_now_wall`

Header: `public/inc/RUNTIME/cf_time.h`

```c
cf_status cf_time_now_wall(cf_time_point *out);
```

Reads wall-clock time in nanoseconds.

Critical points:

- Wall time can move if the system clock changes.
- Use this for timestamps, not performance timing.

### `cf_time_now_mono`

```c
cf_status cf_time_now_mono(cf_time_point *out);
```

Reads monotonic time in nanoseconds.

Critical points:

- Use this for elapsed-time measurements.
- It is the right clock for benchmarks.

### `cf_time_from_ns`, `cf_time_from_ms`, `cf_time_from_sec`

```c
cf_time cf_time_from_ns(cf_i64 ns);
cf_time cf_time_from_ms(cf_i64 ms);
cf_time cf_time_from_sec(cf_i64 sec);
```

Create framework duration values.

Critical points:

- These are simple conversions.
- They do not check multiplication overflow.

### `cf_time_as_ns`, `cf_time_as_ms`, `cf_time_as_sec`

```c
cf_i64 cf_time_as_ns(cf_time d);
cf_i64 cf_time_as_ms(cf_time d);
cf_i64 cf_time_as_sec(cf_time d);
```

Convert a framework duration into a scalar unit.

Critical points:

- Millisecond and second conversions truncate toward zero.

### `cf_time_elapsed`

```c
cf_time cf_time_elapsed(cf_time_point start, cf_time_point end);
```

Returns `end - start`.

Critical points:

- Usually pair this with `cf_time_now_mono`.

### `cf_time_sleep_ms`, `cf_time_sleep_ns`

```c
cf_status cf_time_sleep_ms(cf_u64 ms);
cf_status cf_time_sleep_ns(cf_u64 ns);
```

Sleep for at least the requested duration.

Critical points:

- Interrupted sleeps are resumed.
- System sleep failures return `CF_ERR_TIME_SLEEP`.

### `cf_io_exists`

Header: `public/inc/RUNTIME/cf_io.h`

```c
cf_bool cf_io_exists(const char *path);
```

Checks whether a filesystem path exists.

Critical points:

- Returns `CF_FALSE` for `CF_NULL`.
- It does not distinguish “missing” from “inaccessible”.

### `cf_io_file_size`

```c
cf_status cf_io_file_size(const char *path, cf_usize *size);
```

Reads file metadata and writes the size in bytes.

Critical points:

- Returns `CF_ERR_IO_METADATA` if `stat` fails.

### `cf_io_read_fd`

```c
cf_status cf_io_read_fd(cf_buffer *dst, int fd);
```

Reads all bytes from an open file descriptor into `dst`.

Critical points:

- `dst` must already be a valid `cf_buffer`.
- Data is appended at current `dst->len`.
- The buffer grows in `CF_IO_RESERVE_SIZE` chunks.
- Retries on `EINTR`.

### `cf_io_write_fd`

```c
cf_status cf_io_write_fd(int fd, cf_bytes src);
```

Writes a full byte view to an open file descriptor.

Critical points:

- Handles partial writes.
- Retries on `EINTR`.
- Validates byte count overflow from `src.len * src.elem_size`.

### `cf_io_read_file`

```c
cf_status cf_io_read_file(cf_buffer *dst, const char *path);
```

Opens and reads a full binary file into a `cf_buffer`.

Critical points:

- If `dst->data == CF_NULL`, the buffer is initialized.
- Otherwise data is appended.
- Returns open/read/close-specific IO status values.

### `cf_io_write_file`, `cf_io_append_file`

```c
cf_status cf_io_write_file(const char *path, cf_bytes src);
cf_status cf_io_append_file(const char *path, cf_bytes src);
```

Write or append raw bytes to a file.

Critical points:

- `write_file` truncates existing content.
- `append_file` preserves existing content.
- Both create the file if missing.

### `cf_io_read_text`

```c
cf_status cf_io_read_text(cf_string *dst, const char *path);
```

Reads a file into a framework string.

Critical points:

- Initializes `dst` when empty.
- Appends when already initialized.
- Ensures null termination on success.

### `cf_io_write_text`, `cf_io_append_text`

```c
cf_status cf_io_write_text(const char *path, cf_string *src);
cf_status cf_io_append_text(const char *path, cf_string *src);
```

Write or append string content to a file.

Critical points:

- Writes exactly `src->len` bytes.
- Does not write the string terminator.

### `cf_random_bytes`

Header: `public/inc/RUNTIME/cf_random.h`

```c
cf_status cf_random_bytes(void *dst, cf_usize len);
```

Fills memory with OS-provided random bytes.

Critical points:

- `len == 0` is a successful no-op.
- Uses `getrandom`.
- Retries on `EINTR`.

### `cf_random_u32`, `cf_random_u64`

```c
cf_status cf_random_u32(cf_u32 *dst);
cf_status cf_random_u64(cf_u64 *dst);
```

Generate one random integer.

## Allocator

### `cf_alloc_new`

Header: `public/inc/ALLOCATOR/cf_alloc.h`

```c
void cf_alloc_new(cf_alloc *alloc);
```

Initializes an allocator with default heap-backed callbacks.

Critical points:

- Uses `malloc`, `realloc`, and `free`.
- `ctx` is `CF_NULL`.
- Many containers store a `cf_alloc` internally.

### `cf_alloc_debug_new`

Header: `public/inc/ALLOCATOR/cf_alloc_debug.h`

```c
void cf_alloc_debug_new(cf_alloc_debug *alloc_debug, cf_alloc *alloc, char *statement);
```

Wraps another allocator and tracks allocation activity.

Critical points:

- The debug allocator forwards actual memory operations to the wrapped
  allocator.
- It tracks live pointers with a linked list.
- Invalid realloc/free attempts are counted.
- Useful for tests and leak diagnostics.

### `cf_alloc_debug_log`

```c
void cf_alloc_debug_log(cf_alloc_debug *debug, int line);
```

Prints debug allocator counters and latest state.

## Memory

### `cf_buffer_is_valid`

Header: `public/inc/MEMORY/cf_memory.h`

```c
cf_bool cf_buffer_is_valid(cf_buffer *buffer);
```

Validates the core buffer invariant.

Critical points:

- `data == CF_NULL` requires `len == 0` and `cap == 0`.
- Live data requires `cap >= len`.

### `cf_buffer_init`

```c
cf_status cf_buffer_init(cf_buffer *buffer, cf_usize capacity);
```

Initializes a growable byte buffer.

Critical points:

- Uses the default allocator.
- `capacity == 0` creates an empty valid buffer.

### `cf_buffer_reserve`

```c
cf_status cf_buffer_reserve(cf_buffer *buffer, cf_usize capacity);
```

Ensures at least `capacity` bytes.

Critical points:

- Does not shrink.
- Requires a valid buffer and a realloc callback.

### `cf_buffer_destroy`

```c
void cf_buffer_destroy(cf_buffer *buffer);
```

Releases buffer storage and resets the object.

### `cf_buffer_append_byte`

```c
cf_status cf_buffer_append_byte(cf_buffer *buffer, cf_u8 byte);
```

Appends one byte.

### `cf_buffer_append_bytes`

```c
cf_status cf_buffer_append_bytes(cf_buffer *buffer, cf_bytes bytes);
```

Appends a byte span.

Critical points:

- `bytes.elem_size` is not used in the current append-size calculation.
- Callers should pass byte views with `elem_size == 1` when using this as raw
  byte append.

### `cf_buffer_as_bytes`

```c
cf_status cf_buffer_as_bytes(cf_buffer *buffer, cf_bytes *bytes, cf_usize start, cf_usize end);
```

Creates an inclusive byte-slice view.

Critical points:

- The returned view aliases buffer memory.
- It does not allocate.

### `cf_buffer_reset`

```c
void cf_buffer_reset(cf_buffer *buffer);
```

Sets `len` to zero and keeps capacity.

### `cf_buffer_trunc`

```c
cf_status cf_buffer_trunc(cf_buffer *buffer, cf_usize len);
```

Truncates logical length.

### `cf_buffer_is_empty`

```c
cf_bool cf_buffer_is_empty(cf_buffer *buffer);
```

Returns whether `len == 0`.

### `cf_buffer_info`

```c
void cf_buffer_info(cf_buffer *buffer);
```

Prints debug information.

### `cf_array_is_valid`

Header: `public/inc/MEMORY/cf_array.h`

```c
cf_bool cf_array_is_valid(cf_array *array);
```

Validates a growable array of `cf_array_element`.

### `cf_array_init`

```c
cf_status cf_array_init(cf_array *array, cf_usize capacity);
```

Initializes the array with optional element capacity.

### `cf_array_reserve`

```c
cf_status cf_array_reserve(cf_array *array, cf_usize capacity);
```

Ensures element capacity.

### `cf_array_destroy`

```c
void cf_array_destroy(cf_array *array);
```

Releases array storage.

Critical points:

- It does not free memory referenced by element payload pointers.
- It frees only the array storage itself.

### `cf_array_reset`

```c
void cf_array_reset(cf_array *array);
```

Clears logical contents and keeps capacity.

### `cf_array_peek`

```c
cf_status cf_array_peek(cf_array *array, cf_array_element *element);
```

Reads the last element without removing it.

Critical points:

- Empty arrays return a zeroed element and `CF_OK`.

### `cf_array_push`

```c
cf_status cf_array_push(cf_array *array, cf_array_element *element, ...);
```

Pushes one or more elements. The variadic list must end with `CF_NULL`.

Critical points:

- Elements are copied by value.
- Payload data pointed to by `element->data` is not copied.

### `cf_array_pop`

```c
cf_status cf_array_pop(cf_array *array, cf_array_element *element);
```

Reads and removes the last element.

### `cf_array_get`, `cf_array_set`

```c
cf_status cf_array_get(cf_array *array, cf_usize index, cf_array_element *element);
cf_status cf_array_set(cf_array *array, cf_usize index, cf_array_element *element);
```

Checked indexed access.

### `cf_array_is_empty`

```c
cf_bool cf_array_is_empty(cf_array *array);
```

Returns whether the array has zero logical elements.

### `cf_array_info`

```c
void cf_array_info(cf_array *array);
```

Prints debug information.

## Text

### ASCII Functions

Header: `public/inc/TEXT/cf_ascii.h`

```c
cf_bool cf_ascii_is_alpha(char c);
cf_bool cf_ascii_is_digit(char c);
cf_bool cf_ascii_is_alnum(char c);
cf_bool cf_ascii_is_space(char c);
cf_bool cf_ascii_is_upper(char c);
cf_bool cf_ascii_is_lower(char c);
char cf_ascii_to_upper(char c);
char cf_ascii_to_lower(char c);
cf_isize cf_ascii_hex_value(char c);
```

These functions are locale-free ASCII helpers.

Critical points:

- They do not use C locale.
- They are intended for predictable parsing and security encoding paths.
- `cf_ascii_hex_value` returns `-1` for invalid hex digits.

### `cf_string_is_valid`

Header: `public/inc/TEXT/cf_string.h`

```c
cf_bool cf_string_is_valid(cf_string *str);
```

Validates the string invariant.

Critical points:

- A live string must have `cap > len`.
- `data[len]` must be `'\0'`.

### `cf_string_init`

```c
cf_status cf_string_init(cf_string *str, cf_usize capacity);
```

Initializes a growable null-terminated string.

Critical points:

- Allocates `capacity + 1` bytes.
- The extra byte is for the terminator.

### `cf_string_reserve`

```c
cf_status cf_string_reserve(cf_string *str, cf_usize capacity);
```

Ensures user-visible capacity and keeps terminator space.

### `cf_string_reset`, `cf_string_destroy`

```c
void cf_string_reset(cf_string *str);
void cf_string_destroy(cf_string *str);
```

Reset clears logical contents. Destroy frees storage.

### Append And Conversion

```c
cf_status cf_string_append_char(cf_string *dst, char c);
cf_status cf_string_append_cstr(cf_string *dst, char *c);
cf_status cf_string_append_str(cf_string *dst, cf_string *src);
cf_status cf_string_from_cstr(cf_string *dst, char *src);
cf_status cf_string_as_cstr(char **cdst, cf_string *src);
```

Critical points:

- Append functions preserve existing content.
- `cf_string_from_cstr` replaces content.
- `cf_string_as_cstr` allocates a new C string with the string allocator.

### Query And Mutation

```c
cf_status cf_string_trunc(cf_string *str, cf_usize len);
cf_bool cf_string_is_empty(cf_string *str);
void cf_string_info(cf_string *str);
cf_bool cf_string_eq(cf_string *str1, cf_string *str2);
cf_bool cf_string_contains_char(cf_string *str, char c);
cf_bool cf_string_contains_cstr(cf_string *str, char *c);
cf_bool cf_string_contains_str(cf_string *str1, cf_string *str2);
cf_status cf_string_char_at(cf_string *str, cf_usize index, char *c);
cf_status cf_string_str_at(cf_string *str, cf_usize index, char **c);
cf_status cf_string_trim_left(cf_string *str);
cf_status cf_string_trim_right(cf_string *str);
cf_status cf_string_trim(cf_string *str);
cf_status cf_string_strip(cf_string *str);
cf_status cf_string_replace(cf_string *str, char targetc, char newc);
cf_status cf_string_slice(char **dst, cf_string *src, cf_usize start, cf_usize end);
cf_status cf_string_split(cf_array *dst, cf_string *src, char c);
```

Critical points:

- Trim/strip use ASCII whitespace.
- `cf_string_str_at`, `cf_string_slice`, and `cf_string_split` allocate memory.
- `cf_string_split` stores allocated C strings in a `cf_array`; callers must
  design cleanup for those payloads.

## Math

Headers: `public/inc/MATH/cf_math.h`, `public/inc/MATH/cf_math_storage.h`

The current math layer is documented in detail in
[CF Math Layer Guide](cf-math-layer.md). That page is the complete
function-by-function and struct-by-struct reference for `cf_math`.

At a high level, the public math API now includes:

- primitive AES/rotate/size helpers,
- dtype/layout/device metadata,
- non-owning `cf_math` views,
- reusable `cf_math_metadata` shape descriptions,
- shared CUDA context and workspace lifecycle,
- handler-backed `cf_math_arena` storage,
- readable `cf_math` shape printing,
- bind, unbind, and rebind lifecycle helpers.

The current implementation is focused on the runtime foundation. Operation
families will be layered on top of handlers after allocation, binding, and
metadata are stable.

### Primitive Math Helpers

```c
cf_u8 cf_math_g8_mul_mod(cf_u8 p, cf_u8 q);
cf_u8 cf_math_rotl8(cf_u8 x, cf_u8 n);
cf_u8 cf_math_rotr8(cf_u8 x, cf_u8 n);
cf_u32 cf_math_rotl32(cf_u32 x, cf_u8 n);
cf_u32 cf_math_rotr32(cf_u32 x, cf_u8 n);
cf_usize cf_math_min_usize(cf_usize a, cf_usize b);
cf_usize cf_math_max_usize(cf_usize a, cf_usize b);
```

These helpers cover AES finite-field multiplication, bit rotation, and size
min/max.

### Metadata, Handler, And Context Lifecycle

```c
cf_status cf_math_cuda_context_init(cf_math_cuda_context *ctx, cf_usize bytes, int device_id);
cf_status cf_math_cuda_context_destroy(cf_math_cuda_context *ctx);
cf_status cf_math_cuda_context_reserve(cf_math_cuda_context *ctx, cf_usize bytes);

cf_status cf_math_metadata_init(...);
cf_status cf_math_print_shape(const cf_math *x);
cf_status cf_math_handle_init(...);
cf_status cf_math_handle_reserve(cf_math_handle_t *handler, cf_usize bytes);
cf_status cf_math_handle_alloc(cf_math_handle_t *handler, cf_usize bytes, void **ptr);
void cf_math_handle_reset(cf_math_handle_t *handler);
cf_status cf_math_handle_destroy(cf_math_handle_t *handler);

cf_status cf_math_bind(cf_math *x, cf_math_handle_t *handler, cf_math_metadata *metadata);
cf_status cf_math_unbind(cf_math *x);
cf_status cf_math_rebind(cf_math *x, cf_math_handle_t *handler, cf_math_metadata *metadata);
```

These functions create CUDA runtime state, reusable metadata, and handler-owned
`cf_math_arena` storage. A handler points at a shared CUDA context instead of
copying backend handles. `cf_math` itself is a non-owning view over a handler
slice. Unbinding automatically returns a slice to the handler free-list when no
other view references it.

### Operation Families

Operation APIs are intentionally not documented as implemented until they are
rebuilt against the new handler model.

## Legacy Tensor

Historical header: `public/inc/MATH/cf_tensor.h` (not present in the active
tree after the math-layer update)

The older `cf_tensor` documentation is preserved in
[Tensor And CUDA Backend](tensor-cuda.md) for historical context, but the active
math-layer design now centers on `cf_math`.

## Security

### Hex

Header: `public/inc/SECURITY/cf_hex.h`

```c
cf_status cf_hex_encode(cf_string *dst, cf_bytes src);
cf_status cf_hex_decode(cf_buffer *dst, cf_string *src);
```

Critical points:

- Encode appends uppercase hex text.
- Decode accepts uppercase and lowercase hex.
- Decode requires even input length.

### Base64

Header: `public/inc/SECURITY/cf_base64.h`

```c
cf_status cf_base64_encode(cf_string *dst, cf_bytes src);
cf_status cf_base64_decode(cf_buffer *dst, cf_string *src);
```

Critical points:

- Standard alphabet: `A-Z`, `a-z`, `0-9`, `+`, `/`.
- Padding uses `=`.
- Decode requires length divisible by four.

### AES

Header: `public/inc/SECURITY/cf_aes.h`

```c
cf_status cf_aes_init(cf_aes *aes, const cf_u8 key[CF_AES_MAX_ROUND_KEYS * 4], cf_aes_key_size key_size);
void cf_aes_encrypt_block(cf_aes *aes, cf_u8 dst[CF_AES_BLOCK_SIZE], const cf_u8 src[CF_AES_BLOCK_SIZE]);
void cf_aes_decrypt_block(cf_aes *aes, cf_u8 dst[CF_AES_BLOCK_SIZE], const cf_u8 src[CF_AES_BLOCK_SIZE]);
cf_status cf_aes_pkcs7_pad(cf_buffer *buffer);
cf_status cf_aes_pkcs7_unpad(cf_buffer *buffer);
```

Critical points:

- Supports AES-128, AES-192, and AES-256 key sizes.
- Encrypt/decrypt operate on one 16-byte block.
- No block mode is implemented yet.
- No authentication is implemented yet.
- Use PKCS#7 helpers only around block-aligned AES workflows.
