# Cypher Framework Documentation

This documentation describes the implemented parts of Cypher Framework and the
module boundaries reserved for future work.

The framework is intentionally small, explicit, and extendable. Public headers
live in `public/inc`, implementation files live in `lib/src`, and generated or
hand-written public documentation lives in `public/doc`.

## Pages

- [Project Hierarchy](project-hierarchy.md)
  - How the repository is organized.
  - Which modules are implemented.
  - Which modules are placeholders.

- [Implemented API Reference](implemented-api.md)
  - Public functions that currently have real implementations.
  - What each function does.
  - Critical usage rules and failure modes.

- [CF Math Layer Guide](cf-math-layer.md)
  - Main `cf_math` tensor hierarchy.
  - Every public math enum and struct.
  - Every public `cf_math_*` function grouped by operation family.
  - CPU reference behavior, CUDA dispatch intent, and current unsupported
    training surfaces.

- [Tensor And CUDA Backend](tensor-cuda.md)
  - Legacy `cf_tensor` backend notes.
  - Older CPU/CUDA tensor behavior.
  - Historical GPU type support and smoke-test notes.

- [Extension Guide](extension-guide.md)
  - How to add modules without breaking the framework shape.
  - How to add new functions.
  - How to add CUDA-backed tensor operations.
  - Testing and documentation requirements.

## Current Build Notes

The project builds with strict C flags:

```sh
make lib
make app
make test
```

CUDA is optional. The active math implementation is `cf_math` in
`lib/src/MATH/cf_math.cu`. With `nvcc`, the Makefile compiles `.cu` sources as
CUDA sources. Without `nvcc`, it compiles the same CPU-compatible `.cu` source
with `gcc -x c`, so machines without a CUDA toolkit can still build the
library. A physical GPU is not required for compilation.

`make app` runs CPU `cf_math` examples and only attempts its CUDA roundtrip
example when CUDA runtime headers and a usable CUDA device are both available.
`make test` runs the math-focused test entry point and skips GPU checks unless
CUDA is truly available at build and runtime.

For the current math tensor design, start with
[CF Math Layer Guide](cf-math-layer.md).
