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

## CPU API

### `cf_tensor_init_cpu`

```c
cf_status cf_tensor_init_cpu(
  cf_tensor *tensor,
  cf_usize dim[CF_TENSOR_HIGHEST_RANK],
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

### `cf_tensor_destroy_cpu`

```c
void cf_tensor_destroy_cpu(cf_tensor *tensor);
```

Frees CPU storage and resets the tensor.

Critical points:

- It only frees `data`.
- Use `cf_tensor_destroy_gpu` for tensors that may own CUDA storage.

### `cf_tensor_get_cpu`, `cf_tensor_set_cpu`

```c
cf_status cf_tensor_get_cpu(void *out_value, cf_tensor *tensor, cf_usize indexs[CF_TENSOR_HIGHEST_RANK]);
cf_status cf_tensor_set_cpu(cf_tensor *tensor, cf_usize indexs[CF_TENSOR_HIGHEST_RANK], void *value);
```

Read/write one element through CPU memory.

Critical points:

- Tensor must be CPU-active.
- These functions use stride indexing.
- They return bounds errors for invalid logical indices.

### `cf_tensor_add_cpu`

```c
cf_status cf_tensor_add_cpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);
```

Elementwise addition.

Critical points:

- All tensors must have same rank, dimensions, type, and element size.
- Output tensor must already be initialized.
- Operation is a flat loop over `metadata.len`.

### `cf_tensor_mul_cpu`

```c
cf_status cf_tensor_mul_cpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);
```

Elementwise multiplication.

Critical points:

- Same shape/type requirements as addition.
- This is not matrix multiplication.

### `cf_tensor_scalar_mul_cpu`

```c
cf_status cf_tensor_scalar_mul_cpu(cf_tensor *t1, void *scalar, cf_tensor *t_out);
```

Multiplies every element by one scalar.

Critical points:

- `scalar` must point to the same C type as the tensor element type.
- Output tensor must already be initialized.

### `cf_tensor_matrix_mul_cpu`

```c
cf_status cf_tensor_matrix_mul_cpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);
```

Matrix multiplication with scalar/vector/matrix/batched behavior.

Critical points:

- Rank 0 falls back to scalar multiplication.
- Rank 1 vectors are normalized into matrix form internally.
- Leading dimensions support broadcasting when one side has dimension `1`.
- Output shape must match the normalized/broadcasted result.
- Uses typed CPU loops, not BLAS.

### `cf_tensor_print`

```c
void cf_tensor_print(cf_tensor *tensor);
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
  cf_usize dim[CF_TENSOR_HIGHEST_RANK],
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

### `cf_tensor_destroy_gpu`

```c
void cf_tensor_destroy_gpu(cf_tensor *tensor);
```

Frees CUDA storage and any optional CPU mirror.

### `cf_tensor_get_gpu`, `cf_tensor_set_gpu`

```c
cf_status cf_tensor_get_gpu(void *out_value, cf_tensor *tensor, cf_usize indexs[CF_TENSOR_HIGHEST_RANK]);
cf_status cf_tensor_set_gpu(cf_tensor *tensor, cf_usize indexs[CF_TENSOR_HIGHEST_RANK], void *value);
```

Read/write one element by copying between host memory and device memory.

Critical points:

- These are convenient, not high throughput.
- Do not use them in tight loops for many elements.
- Bulk transfers or kernels are better for performance.

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
cf_status cf_tensor_add_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);
```

CUDA elementwise addition.

Critical points:

- Same shape/type requirements as CPU addition.
- CPU-backed inputs are copied temporarily.
- CUDA-backed inputs avoid temporary input copies.
- CUDA-backed output keeps result on device.
- CPU-backed output receives a device-to-host copy after the kernel.

### GPU Placeholders

```c
cf_status cf_tensor_scalar_mul_gpu(cf_tensor *t1, void *scalar, cf_tensor *t_out);
cf_status cf_tensor_mul_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);
cf_status cf_tensor_matrix_mul_gpu(cf_tensor *t1, cf_tensor *t2, cf_tensor *t_out);
```

Currently return `CF_ERR_UNSUPPORTED`.

Recommended future implementation:

- Elementwise multiplication:
  - Custom CUDA kernel.

- Scalar multiplication:
  - Custom CUDA kernel.

- Matrix multiplication:
  - cuBLAS for float/double GEMM.
  - cuTENSOR for general tensor contractions.

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
#define cf_tensor_init(...)    cf_tensor_init_gpu(...)
#define cf_tensor_destroy(...) cf_tensor_destroy_gpu(...)
#define cf_tensor_get(...)     cf_tensor_get_gpu(...)
#define cf_tensor_set(...)     cf_tensor_set_gpu(...)
#define cf_tensor_add(...)     cf_tensor_add_gpu(...)
```

When CUDA is not available:

```c
#define cf_tensor_init(...)    cf_tensor_init_cpu(...)
#define cf_tensor_destroy(...) cf_tensor_destroy_cpu(...)
#define cf_tensor_get(...)     cf_tensor_get_cpu(...)
#define cf_tensor_set(...)     cf_tensor_set_cpu(...)
#define cf_tensor_add(...)     cf_tensor_add_cpu(...)
```

Critical point:

> In CUDA builds, code that wants host-writeable `tensor.data` should call
> `cf_tensor_init_cpu` explicitly.

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
use cuBLAS for real performance
do not hand-roll GEMM unless the goal is learning
```

For general tensor contractions:

```text
use cuTENSOR when available
```

