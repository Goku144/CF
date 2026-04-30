# Project Hierarchy

Cypher Framework is arranged as a public C API plus private implementation
files. The tree mirrors framework domains, so adding a new module should feel
predictable.

## Top-Level Layout

```text
app/
  src/                  Example/development app entry point.

lib/
  src/                  Framework implementation files.

public/
  inc/                  Public headers installed or included by users.
  doc/                  Public documentation and generated test artifacts.

tests/
  src/                  Framework test suite.

Makefile                Build rules for app, library objects, and tests.
README.md              Short project overview.
LICENSE                GPLv3 license.
```

## Public API Tree

```text
public/inc/
  AI/                   Future AI graph/model/runtime/tokenizer APIs.
  ALLOCATOR/            Allocator interfaces and allocator placeholders.
  CONFIG/               Future config format APIs.
  MATH/                 Math primitives and the cf_math tensor API.
  MEMORY/               Buffer and generic array containers.
  RUNTIME/              Status, types, time, IO, logging, random.
  SECURITY/             AES, hex, base64, future hash/HMAC/secure memory.
  TEXT/                 ASCII and string APIs.
```

## Implementation Tree

```text
lib/src/
  AI/                   Placeholder implementations.
  ALLOCATOR/            Default allocator, debug allocator, placeholders.
  ASM/                  Assembly support files.
  CONFIG/               Placeholder implementations.
  MATH/                 CUDA-source cf_math implementation with CPU fallback.
  MEMORY/               Buffer and array implementation.
  RUNTIME/              Runtime support implementation.
  SECURITY/             AES, hex, base64, placeholders.
  TEXT/                 ASCII and string implementation.
```

## Implemented Modules

### Runtime

- `cf_types`
- `cf_status`
- `cf_log`
- `cf_time`
- `cf_io`
- `cf_random`

Runtime is the foundation layer. Other modules depend on it for shared types,
status codes, timing, logging, file IO, and random bytes.

### Memory

- `cf_buffer`
- `cf_array`

Memory provides reusable dynamic containers used by text, IO, security codecs,
and tests.

### Text

- `cf_ascii`
- `cf_string`

Text builds ASCII utilities and null-terminated growable strings on top of the
memory layer.

### Security

- `cf_aes`
- `cf_hex`
- `cf_base64`

Security currently provides byte/text encoders and AES block operations with
PKCS#7 padding helpers.

### Allocator

- `cf_alloc`
- `cf_alloc_debug`

Allocator modules provide the framework allocator interface and a debug wrapper
used by tests and diagnostics.

### Math

- `cf_math`
- `cf_math.cu` as the active implementation file, compiled by `nvcc` when CUDA
  is available and by `gcc -x c` as a CPU fallback otherwise
- legacy `cf_tensor` documentation retained for historical context

Math now centers on the `cf_math` tensor layer:

- non-owning `cf_math` views,
- reusable dtype-aware shape metadata,
- handler-owned CUDA storage arenas,
- free-list reuse for unbound slices,
- automatic unbind/rebind slice lifecycle,
- CUDA context handles, workspace metadata, and descriptor caches.

The detailed math hierarchy and function reference is documented in
[CF Math Layer Guide](cf-math-layer.md).

## Placeholder Modules

These modules compile and reserve public boundaries, but do not expose real
APIs yet:

```text
AI/cf_graph
AI/cf_model
AI/cf_runtime
AI/cf_tokenizer

CONFIG/cf_config
CONFIG/cf_json
CONFIG/cf_cbor

ALLOCATOR/cf_arena
ALLOCATOR/cf_pool
ALLOCATOR/cf_slab

SECURITY/cf_hash
SECURITY/cf_hmac
SECURITY/cf_parse
SECURITY/cf_secure_mem
```

The placeholders are intentional. They let the project keep a stable directory
shape while future APIs are designed carefully.

## Dependency Direction

The intended dependency direction is:

```text
RUNTIME
  -> ALLOCATOR
  -> MEMORY
  -> TEXT
  -> SECURITY
  -> MATH
  -> CONFIG / AI / APP
```

Important rules:

- Lower layers should not depend on higher layers.
- Public headers should include only what they need.
- Source files should keep implementation-only helpers `static`.
- New modules should return `cf_status` for recoverable API failures.
- Data-owning objects should have `init`, `destroy`, validation, and clear
  ownership rules.
