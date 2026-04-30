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
  - Non-owning `cf_math` view model.
  - Metadata, handler, storage, and CUDA context lifecycle.
  - Why handler arenas and pointer rebinding make the new direction faster.

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

CUDA is optional. The active math implementation is split across
`lib/src/MATH/cf_math.cu` and `lib/src/MATH/cf_math_storage.cu`. With `nvcc`,
the Makefile compiles `.cu` sources as CUDA sources. Without `nvcc`, it compiles
the same CPU-compatible `.cu` sources with `gcc -x c`, so machines without a
CUDA toolkit can still build the library. A physical GPU is not required for
compilation.

`app/src/app.c` contains a CUDA handler lifecycle smoke example. Do not run it
on machines without a usable CUDA device. The old broad math benchmark test has
been removed while the handler-based math layer is rebuilt.

For the current math tensor design, start with
[CF Math Layer Guide](cf-math-layer.md).
