# CF Math Layer Guide

The active math layer is built around descriptor-backed views over handler-owned
storage:

```text
cf_math_desc       shape, strides, dtype, cuBLASLt and cuDNN descriptors
cf_math           byte slice plus descriptor pointer
cf_math_workspace scratchpad plus CUDA stream
cf_math_handle    storage arena plus context/workspace references
cf_math_context   CUDA or CPU backend handles
```

`cf_math` does not allocate by itself. The caller creates descriptors, binds
views into a handle, and then calls hot math functions that assume shapes and
storage were validated during setup.

## Core Objects

```c
struct cf_math_desc
{
  int rank;
  int dim[CF_MATH_MAX_RANK];
  int strides[CF_MATH_MAX_RANK];
  cf_math_cublaslt_desc cublastlt;
  cf_math_cudnn_desc cudnn;
  cf_math_dtype dtype;
};

struct cf_math
{
  cf_usize byte_offset;
  cf_usize elem_len;
  cf_math_desc *desc;
  cf_math_grad_node *grad_fn;
};
```

`cf_math_desc_create` fills row-major strides and creates backend descriptors
when the rank and dtype make them useful. `cf_math_handle_add` binds storage for
a view by reserving a byte slice from `handle->storage.backend`.

## Execution Rules

- Public f16 math functions are hot-path `void` calls.
- Inputs, outputs, optimizer state, and temporary cache tensors must already be
  bound by the caller.
- CUDA kernels use `handle->workspace->stream`.
- Operations do not perform hidden host/device copies.
- Functions with reductions use the caller-provided workspace scratchpad when a
  temporary buffer is needed.

## Implemented F16 Families

Elementwise and unary kernels use packed `uint4` loads for groups of eight
halves and scalar tail kernels for leftovers:

- binary: `add`, `sub`, `mul`, `div`
- unary: `neg`, `sqrt`, `exp`, `log`, `tanh`, `relu`, `sigmoid`, `gelu`
- reductions: `reduce_sum`, `reduce_mean`

Matrix and training helpers:

- `cf_math_matmul_f16`
- `cf_math_matmul_trans_b_f16`
- `cf_math_linear_bias_f16`
- `cf_math_layer_norm_f16`
- `cf_math_layer_norm_stats_f16`
- `cf_math_layer_norm_backward_f16`
- `cf_math_softmax_f16`
- `cf_math_cross_entropy_f16`
- `cf_math_cross_entropy_backward_f16`
- `cf_math_adamw_update_f16`
- `cf_math_zero_grad_f16`

## Training Kernel Contracts

Layer norm normalizes over the last dimension. The stats variant also writes
one mean and one variance value per row; backward consumes those cached row
stats. Softmax is last-dimension only. Cross entropy expects one f16 class index
per row and writes a one-element mean loss. Cross entropy backward writes
`(softmax(logits) - one_hot(target)) / rows`.

The row-wise training kernels use one CUDA block per row, CUB shared-memory
block reductions, and float accumulation for row statistics. Rows whose width is
divisible by eight use packed half loads; non-divisible row widths stay inside
the same kernel but use scalar loads for the row so `uint4` alignment remains
safe and row statistics remain correct.

AdamW is flat and in-place: it updates `Weight`, `M`, and `V` using vectorized
packed half loads where possible and a scalar tail for leftovers. The update is
decoupled AdamW with bias correction from the one-based `step`.

## Benchmark App

`app/src/app.cu` benchmarks the f16 elementwise, unary, reduction, and training
functions. Training benchmarks use row-major matrices with `cols = 1024` for
large runs and `cols = element_count` for small runs, which exercises scalar
tails with commands such as:

```sh
./app/build/app
./app/build/app 1031 10 2
```
