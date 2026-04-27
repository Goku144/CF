# Extension Guide

Cypher Framework is meant to stay free, readable, and extendable. This guide
describes how to add new code without breaking the project shape.

## General Rules

1. Public declarations go in `public/inc/<MODULE>/`.
2. Implementations go in `lib/src/<MODULE>/`.
3. The public include tree should mirror the implementation tree.
4. Public functions should return `cf_status` for recoverable failures.
5. Public data-owning objects should have:
   - validation function,
   - init function,
   - destroy function,
   - clear ownership rules.
6. Implementation-only helpers should be `static`.
7. Placeholder modules should compile cleanly until implemented.
8. New APIs should be documented in `public/doc`.
9. New behavior should be tested in `tests/src/test.c`.

## Error Handling

Use existing status codes when possible:

```text
CF_ERR_NULL          required pointer is null
CF_ERR_INVALID       invalid argument value
CF_ERR_STATE         object is not in a valid state for this operation
CF_ERR_BOUNDS        index/range is outside allowed bounds
CF_ERR_OVERFLOW      size arithmetic overflow
CF_ERR_OOM           allocation failed
CF_ERR_UNSUPPORTED   valid request, but backend/feature is not implemented
```

Avoid inventing new status codes until existing ones cannot express the error.

## Validation Philosophy

The framework checks important public boundaries:

- Null pointers.
- Invalid object state.
- Bounds.
- Size overflow.
- Allocation failure.
- Unsupported platform/backend capabilities.

Performance-sensitive inner loops should not repeat heavy validation once the
public entry point has already proven the inputs are safe.

## Adding A New Module

Example module:

```text
public/inc/FOO/cf_bar.h
lib/src/FOO/cf_bar.c
```

Header shape:

```c
#if !defined(CF_BAR_H)
#define CF_BAR_H

#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_types.h"

typedef struct cf_bar
{
  /* public state */
} cf_bar;

cf_status cf_bar_init(cf_bar *bar);
void cf_bar_destroy(cf_bar *bar);

#endif /* CF_BAR_H */
```

Source shape:

```c
#include "FOO/cf_bar.h"

/*
 * Explain what this helper does and why it belongs to this module.
 */
static cf_status cf_bar_helper(cf_bar *bar)
{
  ...
}

/*
 * Initialize bar state and establish ownership.
 */
cf_status cf_bar_init(cf_bar *bar)
{
  ...
}
```

## Adding A Placeholder Module

If a module is reserved but not implemented yet, keep the header explicit:

```c
/*
 * Public <module> API placeholder.
 *
 * Explain what belongs here later.
 */
```

And the source explicit:

```c
/*
 * <Module> implementation placeholder. Explain what future work belongs here.
 */
typedef int cf_module_placeholder;
```

This keeps strict ISO C builds happy without pretending the module is ready.

## Adding Tensor CPU Operations

CPU tensor operations should:

1. Require CPU-readable storage.
2. Check rank/shape/type compatibility at the public entry point.
3. Dispatch by `cf_tensor_type`.
4. Use flat loops when the operation is elementwise.
5. Keep output caller-owned.

Elementwise operation pattern:

```c
cf_status cf_tensor_op_cpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out)
{
  /* validate once */
  /* switch on t_out->metadata.elem_type */
  /* flat loop over metadata.len */
}
```

Critical point:

> Do not allocate output tensors inside math operations unless the API clearly
> says it owns allocation. Current tensor operations expect pre-initialized
> outputs.

## Adding Tensor CUDA Operations

CUDA tensor operations should:

1. Reject unsupported types with `CF_ERR_UNSUPPORTED`.
2. Prefer CUDA-resident inputs when available.
3. Avoid unnecessary host/device copies.
4. Keep CPU-backed compatibility path when useful.
5. Check `cudaGetLastError` after kernel launch.
6. Synchronize only when the API requires completion before returning.

Elementwise CUDA pattern:

```cuda
__global__ void kernel(const T *a, const T *b, T *out, cf_usize len)
{
  cf_usize i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = a[i] + b[i];
}
```

Launch shape:

```c
int threads = 256;
int blocks = (len + threads - 1) / threads;
kernel<<<blocks, threads>>>(...);
```

## CUDA Libraries

For production performance, prefer NVIDIA libraries:

```text
CUB       reductions, scans, sort, low-level primitives
cuBLAS    matrix multiply, vectors, GEMM, batched GEMM
cuTENSOR  tensor contractions, reductions, permutations
cuDNN     neural-network layers and deep-learning primitives
CUTLASS   custom high-performance GEMM kernels
```

Recommended mapping:

```text
tensor sum/min/max        CUB or cuTENSOR
matrix multiplication     cuBLAS
general tensor contraction cuTENSOR
simple elementwise ops     custom CUDA kernel
custom fused GEMM          CUTLASS
```

## Documentation Requirement

When adding a public function:

1. Add header documentation near the declaration.
2. Add implementation comment above the function in `.c` or `.cu`.
3. Add the function to `public/doc/implemented-api.md`.
4. Add critical ownership/performance notes.

## Testing Requirement

Tests should cover:

- Success path.
- Null pointer inputs.
- Invalid state.
- Bounds errors.
- Overflow when applicable.
- Allocation failure when testable.
- Roundtrip behavior for encoders/decoders.
- Known vectors for cryptographic code.
- CPU/GPU consistency for tensor code when CUDA is available.

## Stability Rule

Before changing a public struct layout or macro dispatch behavior, check:

- `app/src/app.c`
- `tests/src/test.c`
- `README.md`
- `public/doc`

Public shape changes should be rare and documented.

