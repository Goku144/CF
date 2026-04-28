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

- [Tensor And CUDA Backend](tensor-cuda.md)
  - CPU tensor behavior.
  - CUDA tensor behavior.
  - Supported GPU types.
  - CPU/GPU comparison app behavior.
  - Important ownership and performance notes.

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

CUDA is optional. The Makefile uses CUDA only when `nvcc` is available on
`PATH`. If CUDA is installed outside `PATH`, direct compilation can still work:

```sh
/usr/local/cuda-13.2/bin/nvcc -O3 -Ipublic/inc -c lib/src/MATH/cf_tensor_cuda.cu -o /tmp/cf_tensor_cuda.o
```

`make app` runs the tensor smoke-test application. CUDA builds compare CPU and
GPU results for add, elementwise multiply, scalar multiply, matrix multiply,
and batched matrix multiply. CPU-only builds report that GPU comparison was
skipped.
