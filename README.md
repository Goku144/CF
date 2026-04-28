# Cypher Framework

Cypher Framework is a C framework project organized around small runtime,
memory, text, security, math, allocator, configuration, and AI modules. The
current codebase is still in active development, but the implemented core has
unit coverage and is built with strict compiler flags.

## Project Layout

```text
app/            Example application entry point
context/        Project notes and diagrams
lib/src/        Framework implementation files
public/inc/     Public framework headers
tests/src/      Test suite source
public/doc/     Public documentation, generated test output, and crypto artifacts
```

The public include tree mirrors the implementation tree:

```text
ALLOCATOR/
AI/
CONFIG/
MATH/
MEMORY/
RUNTIME/
SECURITY/
TEXT/
```

## Build

The project uses `gcc`, `make`, and `nasm`. CUDA is optional.

If `nvcc` is found on `PATH`, the build defines `CF_CUDA_AVAILABLE=1`, compiles
`.cu` files with `nvcc`, and links with `nvcc`, `cuBLASLt`, and `cuBLAS`.
Otherwise the framework builds and runs with CPU tensor paths only.

If CUDA is installed but not on `PATH`, direct CUDA compilation can still work,
for example:

```sh
/usr/local/cuda-13.2/bin/nvcc -O3 -Ipublic/inc -c lib/src/MATH/cf_tensor_cuda.cu -o /tmp/cf_tensor_cuda.o
```

Build the library objects:

```sh
make lib
```

Build and run the app:

```sh
make app
```

The app is a tensor backend smoke test. In CUDA builds it runs the same
deterministic inputs through CPU and GPU paths, then prints status, result
shape, values, and maximum absolute difference for:

- elementwise add
- elementwise multiply
- scalar multiply
- matrix multiply
- batched matrix multiply

CPU-only builds print that GPU comparison was skipped.

Build and run the tests:

```sh
make test
```

Clean generated build outputs:

```sh
make clean
```

The default compiler flags are:

```text
-Wall -Wextra -Wpedantic -Werror -O3
```

## Implemented Core

### Runtime

Implemented runtime modules:

- `cf_status`: shared framework status codes and symbolic status names.
- `cf_io`: file existence, file size, binary/text read and write helpers.
- `cf_time`: wall-clock time, monotonic time, elapsed time, and sleep helpers.
- `cf_random`: cryptographically secure random bytes using the OS RNG.
- `cf_log`: log levels, log filtering, formatted logs, and `CF_LOG_STATUS`.
- `cf_types`: native-width type grouping helpers.

### Memory

Implemented memory modules:

- `cf_buffer`: owned growable byte buffer.
- `cf_array`: owned growable array of `cf_array_element` values.

These modules include structural validation, bounds checks, null checks, and
overflow checks for public API entry points.

### Text

Implemented text modules:

- `cf_ascii`: ASCII classification, conversion, and hex-value helpers.
- `cf_string`: growable null-terminated string built on top of `cf_buffer`.

### Security

Implemented security modules:

- `cf_hex`: hex encode/decode.
- `cf_base64`: Base64 encode/decode.
- `cf_aes`: AES block encryption/decryption for 128, 192, and 256-bit keys.
- `cf_aes_pkcs7_pad` and `cf_aes_pkcs7_unpad`: PKCS#7 padding helpers for AES block-sized data.

AES currently provides raw block operations and padding helpers. Higher-level
block modes such as CBC, CTR, or GCM are not implemented yet.

### Allocator

Implemented allocator modules:

- `cf_alloc`: default allocation interface wrapping `malloc`, `realloc`, and `free`.
- `cf_alloc_debug`: debug allocator wrapper that tracks allocation events and invalid operations.

### Math

Implemented math module:

- `cf_math_g8_mul_mod`: GF(2^8) multiplication helper used by AES.
- `cf_math_rotl8` and `cf_math_rotr8`: 8-bit rotate helpers.
- `cf_math_rotl32` and `cf_math_rotr32`: 32-bit rotate helpers.
- `cf_math_min_usize` and `cf_math_max_usize`: `cf_usize` min/max helpers.
- `cf_tensor`: dense tensor validation, CPU lifecycle, CPU get/set, readable
  CPU printing, CPU elementwise addition, CPU elementwise multiplication, CPU
  scalar multiplication, and CPU batched matrix multiplication.
- `cf_tensor_cuda`: CUDA tensor lifecycle, CUDA get/set, CPU/GPU transfer
  helpers, CUDA elementwise addition, CUDA elementwise multiplication, CUDA
  scalar multiplication, and cuBLASLt-backed batched float/double matrix
  multiplication.

Tensor operations are in-place: `op1` is both the left operand and destination.
Elementwise hot paths do not validate shape/type compatibility, so callers keep
that contract for performance. Shape flexibility is handled through
`cf_tensor_reserve_*`, `cf_tensor_reshape_*`, and `cf_tensor_resize_*`.

Tensor setup is now explicit:

- `cf_tensor_init_cpu` creates CPU-backed tensors with `tensor.data`.
- `cf_tensor_init_many_cpu` initializes many CPU tensors with the same shape.
- `cf_tensor_init_gpu` creates GPU-backed tensors with `tensor.device_data`.
- `cf_tensor_init_many_gpu` initializes many CUDA tensors with the same shape.
- `cf_tensor_to_gpu` uploads an existing CPU tensor to CUDA storage.
- `cf_tensor_to_cpu` downloads an existing CUDA tensor to CPU storage.
- `cf_tensor_destroy_cpu` frees CPU storage.
- `cf_tensor_destroy_gpu` frees CUDA storage and any optional CPU mirror.

The generic macros select the default backend:

- CPU-only builds map `cf_tensor_init`, `cf_tensor_get`, `cf_tensor_set`,
  `cf_tensor_destroy`, and math operations to CPU functions.
- CUDA builds map those generic names to GPU functions.

Important: in CUDA builds, code that writes directly through `tensor.data`
should call `cf_tensor_init_cpu` explicitly.

Example:

```c
cf_tensor a = {0};
cf_tensor b = {0};
cf_tensor *batch[2] = {&a, &b};

cf_tensor_init_many_cpu(
  batch,
  2,
  (cf_usize[]){4, 0, 0, 0, 0, 0, 0, 0},
  1,
  CF_TENSOR_DOUBLE
);

cf_tensor_add_cpu(&a, &b); /* a = a + b */
cf_tensor_print(&a);

cf_tensor_destroy_many_cpu(batch, 2);
```

CUDA currently supports tensor add, elementwise multiply, and scalar multiply
for:

```text
char, short, int, long, long long,
float, double,
cf_u8, cf_u16, cf_u32, cf_u64
```

CUDA does not currently support `CF_TENSOR_LD` or `CF_TENSOR_U128`.

CUDA math operations are strict GPU operations: tensors must already have
`device_data`, results mutate `op1` on the GPU, and CPU reads should happen
explicitly through `cf_tensor_to_cpu`.

Matrix multiplication supports `[..., M, K] @ [..., K, N] -> [..., M, N]`
through `cf_tensor_batch_mul_*`; `cf_tensor_matrix_mul_*` uses the
same implementation. CUDA uses cuBLASLt for `CF_TENSOR_FLOAT` and
`CF_TENSOR_DOUBLE`. Other CUDA element types currently return
`CF_ERR_UNSUPPORTED` for matrix multiplication.

## Documentation

Public documentation lives under:

```text
public/doc/project/
```

Start here:

```text
public/doc/project/index.md
```

Available pages:

- `project-hierarchy.md`: repository layout, implemented modules, placeholders,
  and dependency direction.
- `implemented-api.md`: function-by-function reference for implemented APIs.
- `tensor-cuda.md`: tensor layout, CPU/GPU setup, CUDA behavior, and
  performance notes.
- `extension-guide.md`: how to add modules, functions, tests, and CUDA tensor
  operations.

## Tests

The test suite currently covers allocator, array, ASCII, string, hex, Base64,
random, log, runtime error checks, math primitives, tensor CPU behavior, tensor
CUDA declarations when compiled with CUDA support, AES block vectors, AES
PKCS#7 padding, file roundtrips, and type helpers.

Latest local test result:

```text
Passed : 484
Failed : 0
```

The test report is written to:

```text
public/doc/test/test.result.txt
```

AES padding file artifacts are written under:

```text
public/doc/crypt/
```

## Logging And Status

`cf_status` is used for API results. `cf_log` is the main diagnostic reporting
tool.

Use status values for control flow:

```c
cf_status status = cf_io_read_file(&buffer, path);
if(status != CF_OK)
  return status;
```

Use logging for diagnostics:

```c
CF_LOG_STATUS(CF_LOG_LEVEL_ERROR, status);
CF_LOG_INFO("loaded file: %s", path);
```

`cf_status_print` has been removed so status reporting goes through the logging
system.

## Current Development Notes

The active core modules have been hardened with error checks for public API
boundaries, including null pointers, invalid state, bounds, overflow, invalid
input, and failed allocation paths.

The codebase still contains placeholder modules so the project structure can
grow without changing layout later. These placeholders compile but do not expose
real public APIs yet.

Tensor GPU lifecycle, get/set, transfer helpers, and elementwise addition are
implemented, along with elementwise multiplication, scalar multiplication, and
cuBLASLt-backed float/double matrix and batched matrix multiplication. CUDA
paths that still lack a supported output contract return `CF_ERR_UNSUPPORTED`.

## Not Implemented Yet

The following modules are still placeholders or incomplete:

- `AI/cf_graph`
- `AI/cf_model`
- `AI/cf_runtime`
- `AI/cf_tokenizer`
- `CONFIG/cf_config`
- `CONFIG/cf_json`
- `CONFIG/cf_cbor`
- `ALLOCATOR/cf_arena`
- `ALLOCATOR/cf_pool`
- `ALLOCATOR/cf_slab`
- `SECURITY/cf_hash`
- `SECURITY/cf_hmac`
- `SECURITY/cf_parse`
- `SECURITY/cf_secure_mem`

Security work still needed:

- AES multi-block modes such as CBC or CTR.
- IV/nonce handling for encryption modes.
- Message authentication or authenticated encryption.
- Hash implementations.
- HMAC implementation.
- Secure memory wiping and secret handling.

CUDA tensor work still needed:

- More efficient strided-batched cuBLASLt path for large batch counts.
- Broader CUDA matrix multiplication types when their output contract is clear.
- cuTENSOR-backed tensor contraction/reduction paths.
- CUB-backed reductions such as sum/min/max.

Infrastructure work still needed:

- Public API design for the placeholder modules.
- Tests for every new module as it becomes real.
- More focused app examples for non-tensor modules.
- Keeping `public/doc/project/implemented-api.md` updated for each public API.

## License

This project is licensed under the GNU General Public License v3.0. See the
`LICENSE` file for details.
