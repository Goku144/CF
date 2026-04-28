# Tensor API Rewrite

This rewrite changes tensors from an out-of-place math model to an in-place
model built for repeated AI-style operations.

## Main Rule

Tensor operations now use `op1` as both the left operand and the destination:

```c
cf_tensor_add_cpu(&op1, &op2);          /* op1 = op1 + op2 */
cf_tensor_mul_cpu(&op1, &op2);          /* op1 = op1 * op2 */
cf_tensor_scalar_mul_cpu(&op1, &value); /* op1 = op1 * value */
cf_tensor_matrix_mul_cpu(&op1, &op2);   /* op1 = op1 @ op2 */
```

The old `t_out` argument was removed from CPU and CUDA math operations.

## Hot Operation Contract

Elementwise add, elementwise multiply, and scalar multiply do not validate
shape, type, rank, or storage compatibility. They dispatch by `op1` element
type and loop over `op1->metadata.len`.

Before calling them, the caller must make sure:

- `op1` and `op2` use the same element type.
- `op2` has at least `op1->metadata.len` elements.
- The selected backend storage exists.
- For CUDA operations, both tensors already have CUDA storage.

This keeps the math path small and predictable.

## Flexible Shape And Capacity

`cf_tensor_metadata` now has:

- `len`: active element count.
- `capacity`: allocated element count.

`cf_tensor_reshape_*` changes only shape metadata and requires the new element
count to fit in the current capacity.

`cf_tensor_resize_*` changes shape and grows storage when needed.

`cf_tensor_reserve_*` grows storage capacity without changing shape.

Batched matrix multiplication stores the result in `op1` and supports:

```text
[..., M, K] @ [..., K, N] -> [..., M, N]
```

Leading batch dimensions can broadcast when one side has dimension `1`.
`cf_tensor_matrix_mul_*` now uses the same implementation as
`cf_tensor_batch_mul_*`.

## Multi Tensor Init And Destroy

You can initialize or destroy many tensors at once with pointer arrays:

```c
cf_tensor a = {0};
cf_tensor b = {0};
cf_tensor *batch[2] = {&a, &b};

cf_tensor_init_many_cpu(batch, 2, dims, rank, CF_TENSOR_DOUBLE);
cf_tensor_destroy_many_cpu(batch, 2);
```

CUDA has matching `cf_tensor_init_many_gpu` and `cf_tensor_destroy_many_gpu`
functions.

## Copy Helpers

The new copy helpers are:

```c
cf_tensor_copy_cpu(&dst, &src);
cf_tensor_copy_from_array_cpu(&tensor, array, count);
cf_tensor_copy_to_array_cpu(array, &tensor, count);
```

CUDA has matching GPU versions. `copy_from_array` makes the tensor a rank-1
vector of `count` elements. `copy` copies shape and data; if the destination is
empty, it is initialized automatically with the source type and shape.

## CPU And CUDA Backends

Both backends expose the same API shape:

- `init`
- `init_many`
- `destroy`
- `destroy_many`
- `reserve`
- `reshape`
- `resize`
- `copy`
- `copy_from_array`
- `copy_to_array`
- `get`
- `set`
- `add`
- `mul`
- `scalar_mul`
- `batch_mul`
- `matrix_mul`

Generic macros still map to CUDA when `CF_CUDA_AVAILABLE` is defined and to CPU
otherwise.

## Files Changed

- `public/inc/MATH/cf_tensor.h`
- `lib/src/MATH/cf_tensor.c`
- `lib/src/MATH/cf_tensor_cuda.cu`
- `tests/src/test.c`
- `app/src/app.c`
- `explanation.md`
