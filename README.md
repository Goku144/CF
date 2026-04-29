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

The project uses `gcc`, `make`, `libm`, and `nasm`. CUDA is optional.

The active math implementation is `cf_math` in `lib/src/MATH/cf_math.cu`.
When `nvcc` is available, the Makefile compiles `.cu` sources with `nvcc` and
enables CUDA headers. When `nvcc` is not available, the same file is compiled
with `gcc -x c`, so CPU-only machines can still build the library.
A physical GPU is not required for compilation; CUDA runtime calls only need a
GPU when they are executed.

Build the library objects:

```sh
make lib
```

Build and run the example app:

```sh
make app
```

The app runs CPU `cf_math` examples and only attempts the CUDA roundtrip
example when CUDA runtime headers and a usable CUDA device are both available.

Build and run the tests:

```sh
make test
```

The tests exercise the `cf_math` CPU reference surface and skip GPU checks
unless CUDA is truly available at build and runtime.

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

- `cf_math` now contains the main tensor object, dtype/layout/device metadata,
  reference-counted shared storage, CUDA context fields, descriptor caches,
  sparse sidecars, dropout sidecars, RNN sidecars, and the public operation map.
- Primitive helpers remain available: `cf_math_g8_mul_mod`, rotate helpers, and
  `cf_usize` min/max helpers.
- CPU reference paths exist for lifecycle, views, initialization, random
  filling, elementwise arithmetic, reductions, dense linear algebra,
  activations, softmax/losses, attention pieces, dropout, embedding,
  optimizers, sparse CSR conversion/ops, shape manipulation, and several
  convolution/normalization paths.
- CUDA-aware context/allocation/copy hooks are guarded by CUDA availability.
  High-performance CUDA kernels and vendor-library dispatch are intended to
  live behind the same `cf_math_*` API surface.

Basic example:

```c
cf_math a = {0};
cf_math b = {0};
cf_math out = {0};
cf_usize dim[CF_MATH_HIGHEST_RANK] = {2, 3};

cf_math_alloc(&a, dim, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL);
cf_math_alloc(&b, dim, 2, CF_DTYPE_F64, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL);

cf_math_ones(&a, CF_NULL);
cf_math_fill(&b, 2.0, CF_NULL);
cf_math_add(&out, &a, &b, CF_NULL); /* out = a + b */

cf_math_free(&out, CF_NULL);
cf_math_free(&b, CF_NULL);
cf_math_free(&a, CF_NULL);
```

The detailed math reference is in
`public/doc/project/cf-math-layer.md`. It explains every public math struct and
every public `cf_math_*` function.

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
- `cf-math-layer.md`: complete `cf_math` tensor/lifecycle/operation reference.
- `tensor-cuda.md`: legacy `cf_tensor` layout, CPU/GPU setup, CUDA behavior,
  and performance notes.
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

The `cf_math` layer has a broad public tensor operation map with CPU reference
paths and CUDA-aware context/allocation hooks. RNN/LSTM/GRU, complete MHA
backward, and multi-GPU all-reduce still need their full CUDA/library-backed
training implementations.

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
- CUDA kernels and vendor-library dispatch behind the new `cf_math_*` public
  operation map.

Infrastructure work still needed:

- Public API design for the placeholder modules.
- Tests for every new module as it becomes real.
- More focused app examples for non-tensor modules.
- Keeping `public/doc/project/implemented-api.md` updated for each public API.

## License

This project is licensed under the GNU General Public License v3.0. See the
`LICENSE` file for details.
