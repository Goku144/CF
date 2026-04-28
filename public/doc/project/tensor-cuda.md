# Tensor And CUDA Backend

The tensor module is the most performance-sensitive part of the current
framework. It has a CPU backend and a CUDA backend.

Header:

```text
public/inc/MATH/cf_tensor.h
```

Implementations:

```text
lib/src/MATH/cf_tensor.c
lib/src/MATH/cf_tensor_cuda.cu
```

## Tensor Object

```c
typedef struct cf_tensor
{
  void *data;
  void *device_data;
  cf_usize dim[CF_TENSOR_HIGHEST_RANK];
  cf_usize rank;
  cf_tensor_device device;
  cf_tensor_metadata metadata;
} cf_tensor;
```

Fields:

- `data`
  - CPU memory.
  - Used by CPU operations, printing, CPU get/set, and host-side code.

- `device_data`
  - CUDA device memory.
  - Used by CUDA operations.

- `dim`
  - Active tensor dimensions.
  - Only `dim[0..rank-1]` are meaningful.

- `rank`
  - Number of active dimensions.
  - Maximum is `CF_TENSOR_HIGHEST_RANK`.

- `device`
  - Current active storage device.
  - Either `CF_TENSOR_DEVICE_CPU` or `CF_TENSOR_DEVICE_CUDA`.

- `metadata.len`
  - Element count, not byte count.

- `metadata.capacity`
  - Allocated element count.
  - Resize can grow this; reshape only succeeds when the new shape fits.

- `metadata.stride`
  - Row-major stride table in elements.

- `metadata.elem_size`
  - Byte width of one element.

- `metadata.elem_type`
  - Framework tensor type enum.

## Shape And Stride

The tensor stores data as flat memory. Shape is metadata over that flat memory.

For shape:

```text
[K, 7, 8]
```

row-major strides are:

```text
stride[0] = 7 * 8 = 56
stride[1] = 8
stride[2] = 1
```

Coordinate:

```text
[k, i, j]
```

flat index:

```c
flat = k * 56 + i * 8 + j;
```

Critical point:

> A tensor is a shaped interpretation of flat storage.

## Operation Model

Math operations are in-place:

```c
cf_tensor_add_cpu(&op1, &op2);          /* op1 = op1 + op2 */
cf_tensor_mul_cpu(&op1, &op2);          /* op1 = op1 * op2 */
cf_tensor_scalar_mul_cpu(&op1, &value); /* op1 = op1 * value */
cf_tensor_matrix_mul_cpu(&op1, &op2);   /* op1 = op1 @ op2 */
```

Elementwise hot paths do not validate shape, rank, type, or storage
compatibility. Callers must make sure operands are compatible before calling
them.

## CPU API

### `cf_tensor_init_cpu`

```c
cf_status cf_tensor_init_cpu(
  cf_tensor *tensor,
  const cf_usize dim[CF_TENSOR_HIGHEST_RANK],
  cf_usize rank,
  cf_tensor_type elem_type
);
```

Creates a CPU-resident tensor.

Critical points:

- Allocates `tensor->data`.
- Zeroes CPU storage.
- Sets `tensor->device = CF_TENSOR_DEVICE_CPU`.
- Does not allocate CUDA storage.

### `cf_tensor_init_many_cpu`, `cf_tensor_destroy_many_cpu`

```c
cf_status cf_tensor_init_many_cpu(
  cf_tensor **tensors,
  cf_usize count,
  const cf_usize dim[CF_TENSOR_HIGHEST_RANK],
  cf_usize rank,
  cf_tensor_type elem_type
);

void cf_tensor_destroy_many_cpu(cf_tensor **tensors, cf_usize count);
```

Initialize or destroy many tensors from an array of tensor pointers.

### `cf_tensor_destroy_cpu`

```c
void cf_tensor_destroy_cpu(cf_tensor *tensor);
```

Frees CPU storage and resets the tensor.

Critical points:

- It only frees `data`.
- Use `cf_tensor_destroy_gpu` for tensors that may own CUDA storage.

### `cf_tensor_reserve_cpu`, `cf_tensor_reshape_cpu`, `cf_tensor_resize_cpu`

```c
cf_status cf_tensor_reserve_cpu(cf_tensor *tensor, cf_usize capacity);
cf_status cf_tensor_reshape_cpu(
  cf_tensor *tensor,
  const cf_usize dim[CF_TENSOR_HIGHEST_RANK],
  cf_usize rank
);
cf_status cf_tensor_resize_cpu(
  cf_tensor *tensor,
  const cf_usize dim[CF_TENSOR_HIGHEST_RANK],
  cf_usize rank
);
```

Capacity and shape management.

Critical points:

- `reserve` grows allocation without changing shape.
- `reshape` changes metadata only and requires enough capacity.
- `resize` changes shape and grows allocation when needed.

### `cf_tensor_get_cpu`, `cf_tensor_set_cpu`

```c
cf_status cf_tensor_get_cpu(
  void *out_value,
  const cf_tensor *tensor,
  const cf_usize indexs[CF_TENSOR_HIGHEST_RANK]
);
cf_status cf_tensor_set_cpu(
  cf_tensor *tensor,
  const cf_usize indexs[CF_TENSOR_HIGHEST_RANK],
  const void *value
);
```

Read/write one element through CPU memory.

Critical points:

- Tensor must be CPU-active.
- These functions use stride indexing.
- They return bounds errors for invalid logical indices.

### CPU Copy Helpers

```c
cf_status cf_tensor_copy_cpu(cf_tensor *dst, const cf_tensor *src);
cf_status cf_tensor_copy_from_array_cpu(
  cf_tensor *tensor,
  const void *array,
  cf_usize count
);
cf_status cf_tensor_copy_to_array_cpu(
  void *array,
  const cf_tensor *tensor,
  cf_usize count
);
```

Copy tensor data or plain array data.

Critical points:

- `copy` copies shape and data; empty destinations are initialized.
- `copy_from_array` makes the tensor a rank-1 vector of `count` elements.

### `cf_tensor_add_cpu`

```c
cf_status cf_tensor_add_cpu(cf_tensor *op1, const cf_tensor *op2);
```

In-place elementwise addition.

Critical points:

- Stores `op1[i] + op2[i]` back into `op1[i]`.
- Shape/type/storage compatibility is caller-owned.
- Operation is a flat loop over `op1->metadata.len`.

### `cf_tensor_mul_cpu`

```c
cf_status cf_tensor_mul_cpu(cf_tensor *op1, const cf_tensor *op2);
```

In-place elementwise multiplication.

Critical points:

- Stores `op1[i] * op2[i]` back into `op1[i]`.
- Same caller-owned compatibility contract as addition.
- This is not matrix multiplication.

### `cf_tensor_scalar_mul_cpu`

```c
cf_status cf_tensor_scalar_mul_cpu(cf_tensor *op1, const void *scalar);
```

Multiplies every element by one scalar.

Critical points:

- `scalar` must point to the same C type as the tensor element type.
- Result is stored back into `op1`.

### `cf_tensor_batch_mul_cpu`, `cf_tensor_matrix_mul_cpu`

```c
cf_status cf_tensor_batch_mul_cpu(cf_tensor *op1, const cf_tensor *op2);
cf_status cf_tensor_matrix_mul_cpu(cf_tensor *op1, const cf_tensor *op2);
```

In-place batched matrix multiplication.

Critical points:

- Supports `[..., M, K] @ [..., K, N] -> [..., M, N]`.
- Leading batch dimensions can broadcast when one side has dimension `1`.
- The result is stored in `op1`.
- `op1` shape becomes the broadcasted output shape.
- A temporary result buffer protects the original `op1` data while computing.
- Uses typed CPU loops, not BLAS.

### `cf_tensor_print`

```c
void cf_tensor_print(const cf_tensor *tensor);
```

Prints a readable nested CPU tensor.

Critical points:

- It reads `tensor->data`.
- CUDA tensors should be copied to CPU before printing.

## CUDA API

CUDA functions are declared only when `CF_CUDA_AVAILABLE` is enabled.

### `cf_tensor_init_gpu`

```c
cf_status cf_tensor_init_gpu(
  cf_tensor *tensor,
  const cf_usize dim[CF_TENSOR_HIGHEST_RANK],
  cf_usize rank,
  cf_tensor_type elem_type
);
```

Creates a GPU-resident tensor.

Critical points:

- Allocates `tensor->device_data`.
- Zeroes CUDA storage.
- Sets `tensor->device = CF_TENSOR_DEVICE_CUDA`.
- Does not allocate CPU `data`.

### `cf_tensor_init_many_gpu`, `cf_tensor_destroy_many_gpu`

```c
cf_status cf_tensor_init_many_gpu(
  cf_tensor **tensors,
  cf_usize count,
  const cf_usize dim[CF_TENSOR_HIGHEST_RANK],
  cf_usize rank,
  cf_tensor_type elem_type
);

void cf_tensor_destroy_many_gpu(cf_tensor **tensors, cf_usize count);
```

Initialize or destroy many CUDA tensors from an array of tensor pointers.

### `cf_tensor_destroy_gpu`

```c
void cf_tensor_destroy_gpu(cf_tensor *tensor);
```

Frees CUDA storage and any optional CPU mirror.

### `cf_tensor_reserve_gpu`, `cf_tensor_reshape_gpu`, `cf_tensor_resize_gpu`

```c
cf_status cf_tensor_reserve_gpu(cf_tensor *tensor, cf_usize capacity);
cf_status cf_tensor_reshape_gpu(
  cf_tensor *tensor,
  const cf_usize dim[CF_TENSOR_HIGHEST_RANK],
  cf_usize rank
);
cf_status cf_tensor_resize_gpu(
  cf_tensor *tensor,
  const cf_usize dim[CF_TENSOR_HIGHEST_RANK],
  cf_usize rank
);
```

CUDA capacity and shape management.

Critical points:

- `reserve` grows CUDA allocation without changing shape.
- `reshape` changes metadata only and requires enough capacity.
- `resize` changes shape and grows device allocation when needed.
- GPU resize invalidates stale CPU mirrors.

### `cf_tensor_get_gpu`, `cf_tensor_set_gpu`

```c
cf_status cf_tensor_get_gpu(
  void *out_value,
  const cf_tensor *tensor,
  const cf_usize indexs[CF_TENSOR_HIGHEST_RANK]
);
cf_status cf_tensor_set_gpu(
  cf_tensor *tensor,
  const cf_usize indexs[CF_TENSOR_HIGHEST_RANK],
  const void *value
);
```

Read/write one element by copying between host memory and device memory.

Critical points:

- These are convenient, not high throughput.
- Do not use them in tight loops for many elements.
- Bulk transfers or kernels are better for performance.

### CUDA Copy Helpers

```c
cf_status cf_tensor_copy_gpu(cf_tensor *dst, const cf_tensor *src);
cf_status cf_tensor_copy_from_array_gpu(
  cf_tensor *tensor,
  const void *array,
  cf_usize count
);
cf_status cf_tensor_copy_to_array_gpu(
  void *array,
  const cf_tensor *tensor,
  cf_usize count
);
```

Copy tensor data or host array data into/out of CUDA storage.

### `cf_tensor_to_gpu`

```c
cf_status cf_tensor_to_gpu(cf_tensor *tensor);
```

Uploads CPU storage into CUDA storage.

Critical points:

- Requires `tensor->data`.
- Allocates `device_data` if needed.
- Marks CUDA active.

### `cf_tensor_to_cpu`

```c
cf_status cf_tensor_to_cpu(cf_tensor *tensor);
```

Downloads CUDA storage into CPU storage.

Critical points:

- Allocates `data` if needed.
- Marks CPU active.

### `cf_tensor_free_gpu`

```c
cf_status cf_tensor_free_gpu(cf_tensor *tensor);
```

Frees only CUDA storage.

Critical points:

- If CPU data exists, the tensor remains CPU-backed.
- If the tensor was GPU-only, it is reset.

### `cf_tensor_add_gpu`

```c
cf_status cf_tensor_add_gpu(cf_tensor *op1, const cf_tensor *op2);
```

CUDA elementwise addition.

Critical points:

- `op1` and `op2` must already have CUDA storage.
- Type and element count compatibility is caller-owned.
- Result is stored back into `op1`.
- The launch is asynchronous after `cudaGetLastError` validates the launch.
- Use `cf_tensor_to_cpu` after the operation when host-side data is needed.

### `cf_tensor_mul_gpu`

```c
cf_status cf_tensor_mul_gpu(cf_tensor *op1, const cf_tensor *op2);
```

CUDA elementwise multiplication.

Critical points:

- `op1` and `op2` must already have CUDA storage.
- Type and element count compatibility is caller-owned.
- Result is stored back into `op1`.
- The launch is asynchronous after `cudaGetLastError` validates the launch.
- Use `cf_tensor_to_cpu` after the operation when host-side data is needed.

### `cf_tensor_scalar_mul_gpu`

```c
cf_status cf_tensor_scalar_mul_gpu(cf_tensor *op1, const void *scalar);
```

CUDA scalar multiplication.

Critical points:

- `scalar` is a host pointer to one value of the tensor element type.
- The scalar is passed into the kernel by value.
- `op1` must already have CUDA storage.
- Result is stored back into `op1`.
- The launch is asynchronous after `cudaGetLastError` validates the launch.
- Use `cf_tensor_to_cpu` after the operation when host-side data is needed.

### `cf_tensor_batch_mul_gpu`, `cf_tensor_matrix_mul_gpu`

```c
cf_status cf_tensor_batch_mul_gpu(cf_tensor *op1, const cf_tensor *op2);
cf_status cf_tensor_matrix_mul_gpu(cf_tensor *op1, const cf_tensor *op2);
```

CUDA batched matrix multiplication through cuBLASLt.

Critical points:

- Supports `CF_TENSOR_FLOAT` and `CF_TENSOR_DOUBLE`.
- `op1` and `op2` must already have CUDA storage.
- Supports `[..., M, K] @ [..., K, N] -> [..., M, N]`.
- Leading batch dimensions can broadcast when one side has dimension `1`.
- Result is stored back into `op1`.
- `op1` shape becomes the broadcasted output shape.
- Storage is row-major, matching the framework tensor layout.
- The float path tries TF32-enabled cuBLASLt compute first for throughput, then
  falls back to normal FP32 compute if the device/path does not support it.
- Non-float/non-double tensor types return `CF_ERR_UNSUPPORTED` for matrix
  multiplication.

## Supported GPU Types

CUDA backend currently supports:

```text
CF_TENSOR_CHAR
CF_TENSOR_SHORT
CF_TENSOR_INT
CF_TENSOR_LONG
CF_TENSOR_LL
CF_TENSOR_FLOAT
CF_TENSOR_DOUBLE
CF_TENSOR_U8
CF_TENSOR_U16
CF_TENSOR_U32
CF_TENSOR_U64
```

Matrix multiplication is narrower than tensor allocation and elementwise math:

```text
CF_TENSOR_FLOAT
CF_TENSOR_DOUBLE
```

CUDA backend does not support:

```text
CF_TENSOR_LD
CF_TENSOR_U128
```

Unsupported GPU types return:

```text
CF_ERR_UNSUPPORTED
```

## Macro Dispatch

When CUDA is available:

```c
#define cf_tensor_init(tensor, dim, rank, elem_type) \
  cf_tensor_init_gpu((tensor), (dim), (rank), (elem_type))
#define cf_tensor_resize(tensor, dim, rank) \
  cf_tensor_resize_gpu((tensor), (dim), (rank))
#define cf_tensor_add(op1, op2) cf_tensor_add_gpu((op1), (op2))
#define cf_tensor_batch_mul(op1, op2) \
  cf_tensor_batch_mul_gpu((op1), (op2))
#define cf_tensor_matrix_mul(op1, op2) cf_tensor_matrix_mul_gpu((op1), (op2))
```

When CUDA is not available:

```c
#define cf_tensor_init(tensor, dim, rank, elem_type) \
  cf_tensor_init_cpu((tensor), (dim), (rank), (elem_type))
#define cf_tensor_resize(tensor, dim, rank) \
  cf_tensor_resize_cpu((tensor), (dim), (rank))
#define cf_tensor_add(op1, op2) cf_tensor_add_cpu((op1), (op2))
#define cf_tensor_batch_mul(op1, op2) \
  cf_tensor_batch_mul_cpu((op1), (op2))
#define cf_tensor_matrix_mul(op1, op2) cf_tensor_matrix_mul_cpu((op1), (op2))
```

Critical point:

> In CUDA builds, code that wants host-writeable `tensor.data` should call
> `cf_tensor_init_cpu` explicitly.

## App Smoke Test

`app/src/app.c` is a small CPU/GPU tensor comparison program.

When the framework is built with CUDA support, `make app` runs these
operations on deterministic double-precision inputs:

```text
tensor add
tensor elementwise mul
tensor scalar mul
tensor matrix mul
tensor batched matrix mul
```

For each operation, the app prints:

- CPU status.
- GPU status.
- Maximum absolute difference between CPU and GPU results.
- Output shape and values for both backends.

The comparison expects exact or near-exact agreement. The current threshold is:

```text
1e-9
```

CPU-only builds still compile the same app entry point, but print that GPU tests
were skipped because `CF_CUDA_AVAILABLE` is not enabled.

This app is meant as a quick backend sanity check, not a replacement for
`tests/src/test.c`.

## Performance Rules

For elementwise operations:

```text
one thread = one flat tensor element
```

For reductions:

```text
use CUB or multi-pass reduction
avoid one atomic per element
avoid many blocks fighting over one global atomic
```

For matrix multiplication:

```text
use cuBLASLt for real performance
do not hand-roll GEMM unless the goal is learning
```

For general tensor contractions:

```text
use cuTENSOR when available
```
