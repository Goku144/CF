# CF Math Layer Guide

This page is the detailed guide for the new `cf_math` layer.

It explains the math library from the top down:

- What problem the math layer solves.
- How the files are arranged.
- What every public enum and struct means.
- What every public `cf_math_*` function is for.
- Which functions are CPU reference paths today.
- Which functions are CUDA/library dispatch surfaces for the GPU backend.

The goal is to make the math layer understandable even if you have never built
a tensor library before.

## File Hierarchy

The math layer lives in the public include tree and implementation tree:

```text
public/inc/MATH/
  cf_math.h       Public math tensor API, public structs, function map.

lib/src/MATH/
  cf_math.cu      CUDA-source implementation file. It contains the CPU reference
                 implementation today and is the home for CUDA kernels/library
                 dispatch as the backend grows.
```

Older docs may mention `cf_tensor.h`, `cf_tensor.c`, or `cf_tensor_cuda.cu`.
Those names describe the previous tensor backend and are not active files after
the math-layer update.

The main public API is:

```text
public/inc/MATH/cf_math.h
```

The main implementation currently lives in:

```text
lib/src/MATH/cf_math.cu
```

Build behavior:

- With `nvcc`, `cf_math.cu` is compiled as a CUDA source and
  `CF_CUDA_AVAILABLE` is defined.
- Without `nvcc`, `cf_math.cu` is compiled by `gcc -x c`, which keeps the CPU
  reference implementation available on machines without CUDA.
- A physical GPU is not required to compile the project. GPU hardware is only
  required when CUDA runtime paths are executed.

## Big Picture

The math layer is built around a tensor object named `cf_math`.

A tensor is a block of memory plus metadata that explains how to interpret that
memory as a scalar, vector, matrix, image batch, sequence batch, or higher-rank
array.

Example:

```text
Flat memory:
  [1, 2, 3, 4, 5, 6]

Shape:
  [2, 3]

Meaning:
  2 rows, 3 columns

View:
  row 0: 1 2 3
  row 1: 4 5 6
```

The same memory can sometimes be viewed with a different shape without copying.
That is why the tensor stores both a storage pointer and metadata such as rank,
dimensions, strides, layout, dtype, and device.

## Dispatch Model

Most math functions have this shape:

```c
cf_status cf_math_operation(cf_math *out, const cf_math *x, ..., cf_math_cuda_context *ctx);
```

The pattern means:

- `out` is the output tensor.
- `x` and other tensor pointers are inputs.
- `ctx` is the optional CUDA context used by GPU paths.
- The return value is a `cf_status`, usually `CF_OK` or an error code.

The intended dispatch rule is:

```text
dtype x layout x device -> CPU loop, CUDA kernel, cuBLAS, cuDNN, cuSPARSE, etc.
```

For example:

- F32 matrix multiplication on CUDA should go to cuBLASLt or cuBLAS.
- Convolution on CUDA should go to cuDNN.
- Sparse matrix multiplication on CUDA should go to cuSPARSE.
- CPU fallback paths use clear reference loops.

## Important Words

### Tensor

A tensor is a multi-dimensional array. A scalar has rank 0, a vector usually has
rank 1, a matrix has rank 2, and image batches often have rank 4.

### Rank

Rank is the number of dimensions.

Examples:

```text
scalar       rank 0
[10]         rank 1
[3, 4]       rank 2
[N, C, H, W] rank 4
```

### Dimension

A dimension is the size of one axis. For `[N, C, H, W]`, `N` is batch size,
`C` is channels, `H` is height, and `W` is width.

### Stride

Stride says how far to move in flat memory when one coordinate changes.

For row-major shape `[2, 3, 4]`:

```text
stride[0] = 12
stride[1] = 4
stride[2] = 1
```

The logical coordinate `[a, b, c]` maps to:

```text
flat_index = a * 12 + b * 4 + c
```

### Dtype

Dtype means data type. It tells the math layer whether the memory contains
`double`, `float`, `int32_t`, byte values, and so on.

The dtype matters because a raw `void *` pointer is not enough to know how to
read or compute values.

### Layout

Layout describes the order of data in memory. For example:

- `ROW_MAJOR`: C-style arrays.
- `COL_MAJOR`: Fortran-style arrays and native cuBLAS convention.
- `NCHW`: image tensors ordered as batch, channels, height, width.
- `NHWC`: image tensors ordered as batch, height, width, channels.
- `STRIDED`: arbitrary layout, usually from views, slices, or broadcasting.

### Device

Device says where the newest tensor values live:

- `CF_DEVICE_CPU`: host memory.
- `CF_DEVICE_CUDA`: CUDA device memory.

## Public Enums

### `cf_math_shape`

```c
typedef enum cf_math_shape
{
  CF_SHAPE_SCALAR = 0,
  CF_SHAPE_MATRIX,
  CF_SHAPE_TENSOR,
} cf_math_shape;
```

This is a coarse label for the tensor shape:

- `CF_SHAPE_SCALAR`: rank 0, one value.
- `CF_SHAPE_MATRIX`: rank 2, rows and columns.
- `CF_SHAPE_TENSOR`: any other shaped array.

It is metadata for dispatch and debugging. The exact shape still comes from
`rank` and `dim[]`.

### `cf_math_device`

```c
typedef enum cf_math_device
{
  CF_DEVICE_CPU = 0,
  CF_DEVICE_CUDA,
} cf_math_device;
```

This tells operations where memory lives.

CPU tensors can be processed with CPU loops. CUDA tensors require CUDA kernels
or CUDA library calls. CPU code must not blindly read CUDA pointers.

### `cf_math_dtype`

```c
typedef enum cf_math_dtype
{
  CF_DTYPE_F64 = 0,
  CF_DTYPE_F32 = 1,
  CF_DTYPE_F16 = 2,
  CF_DTYPE_BF16 = 3,
  CF_DTYPE_FP8E4M3 = 4,
  CF_DTYPE_FP8E5M2 = 5,
  CF_DTYPE_I32 = 6,
  CF_DTYPE_I8 = 7,
  CF_DTYPE_U8 = 8,
  CF_DTYPE_BOOL = 9,
} cf_math_dtype;
```

This tells the math layer how to interpret bytes:

- `F64`: 64-bit floating point, C `double`.
- `F32`: 32-bit floating point, C `float`.
- `F16`: IEEE half precision, common tensor-core input type.
- `BF16`: bfloat16, shorter mantissa than F32 but wider range than F16.
- `FP8E4M3`: FP8 format often used for Hopper inference.
- `FP8E5M2`: FP8 format often used for gradients/training paths.
- `I32`: 32-bit signed integer, useful for labels and indices.
- `I8`: signed byte, useful for quantized inference.
- `U8`: unsigned byte, useful for masks and byte data.
- `BOOL`: byte-backed boolean, `0` or `1`.

The current C reference implementation directly reads and writes F64, F32,
I32, I8, U8, and BOOL. The smaller GPU-focused floating types are represented
in metadata so the CUDA side can dispatch correctly.

### `cf_math_layout`

```c
typedef enum cf_math_layout
{
  CF_LAYOUT_ROW_MAJOR = 0,
  CF_LAYOUT_COL_MAJOR = 1,
  CF_LAYOUT_NHWC = 2,
  CF_LAYOUT_NCHW = 3,
  CF_LAYOUT_STRIDED = 4,
} cf_math_layout;
```

This explains how coordinates map to memory.

Why this matters:

- cuBLAS is naturally column-major.
- C code is naturally row-major.
- cuDNN convolution can be faster in NHWC on tensor cores.
- Views and broadcasts may have arbitrary strides.

The layer stores layout so operations can dispatch without guessing.

### `cf_math_mem_flags`

```c
typedef enum cf_math_mem_flags
{
  CF_MEM_DEFAULT = 0,
  CF_MEM_PINNED = 1 << 0,
  CF_MEM_MANAGED = 1 << 1,
  CF_MEM_POOLED = 1 << 2,
  CF_MEM_ALIGNED_128 = 1 << 3,
  CF_MEM_READ_ONLY = 1 << 4,
  CF_MEM_PEER_MAPPED = 1 << 5,
} cf_math_mem_flags;
```

These flags describe allocation and memory-use policy:

- `CF_MEM_DEFAULT`: ordinary CPU allocation or ordinary device allocation.
- `CF_MEM_PINNED`: host memory allocated for faster host/device transfers.
- `CF_MEM_MANAGED`: CUDA managed memory when CUDA is available.
- `CF_MEM_POOLED`: memory intended to come from a CUDA memory pool.
- `CF_MEM_ALIGNED_128`: storage is aligned for 128-byte vector/tensor-core use.
- `CF_MEM_READ_ONLY`: memory can be treated as read-mostly, useful for weights.
- `CF_MEM_PEER_MAPPED`: memory can be accessed by another GPU.

Flags do not replace the actual pointer. They are metadata used by allocators
and dispatch decisions.

### `cf_math_grad_state`

```c
typedef enum cf_math_grad_state
{
  CF_GRAD_NONE = 0,
  CF_GRAD_LEAF = 1,
  CF_GRAD_INTERIOR = 2,
  CF_GRAD_DETACHED = 3,
} cf_math_grad_state;
```

This describes how a tensor participates in automatic differentiation:

- `CF_GRAD_NONE`: no gradient is needed.
- `CF_GRAD_LEAF`: a parameter or user-created tensor that stores gradients.
- `CF_GRAD_INTERIOR`: an intermediate value in a graph.
- `CF_GRAD_DETACHED`: explicitly removed from gradient flow.

The current implementation stores the metadata. Full autograd graph execution
is not implemented yet.

### `cf_math_softmax_mode`

```c
typedef enum cf_math_softmax_mode
{
  CF_SOFTMAX_CHANNEL = 0,
  CF_SOFTMAX_INSTANCE = 1,
} cf_math_softmax_mode;
```

This tells softmax what conceptual axis is being normalized:

- `CHANNEL`: classification logits over classes/channels.
- `INSTANCE`: per-instance normalization, useful in attention scores.

The CPU reference path takes an explicit axis and treats the mode as dispatch
metadata.

### `cf_math_rnn_mode`

```c
typedef enum cf_math_rnn_mode
{
  CF_RNN_RELU = 0,
  CF_RNN_TANH = 1,
  CF_RNN_LSTM = 2,
  CF_RNN_GRU = 3,
} cf_math_rnn_mode;
```

This describes which recurrent cell a `cf_math_rnn_state` belongs to:

- ReLU RNN.
- tanh RNN.
- LSTM.
- GRU.

The RNN surface exists for cuDNN-backed future work.

## Public Structs

### `cf_math_workspace`

```c
typedef struct cf_math_workspace
{
  void *ptr;
  cf_usize size;
  cf_usize high_water;
} cf_math_workspace;
```

This is reusable scratch memory.

Many GPU library calls need temporary storage. Allocating that temporary memory
inside every operation is slow, so a context owns a workspace that can grow and
then be reused.

Fields:

- `ptr`: pointer to the scratch buffer.
- `size`: current buffer capacity in bytes.
- `high_water`: largest requested workspace size so far.

Use:

```c
cf_math_workspace_reserve(&ctx, bytes);
```

### `cf_math_cuda_context`

```c
typedef struct cf_math_cuda_context
{
  int device_id;
  cudaStream_t stream;
  cudaStream_t h2d_stream;
  cudaStream_t d2h_stream;
  cublasHandle_t cublas;
  cublasLtHandle_t cublasLt;
  cudnnHandle_t cudnn;
  cusparseHandle_t cusparse;
  cusolverDnHandle_t cusolver;
  curandGenerator_t curand;
  ncclComm_t nccl;
  cf_math_workspace workspace;
} cf_math_cuda_context;
```

This groups CUDA state for one device.

Fields:

- `device_id`: CUDA device index.
- `stream`: main asynchronous compute stream.
- `h2d_stream`: stream for host-to-device copies.
- `d2h_stream`: stream for device-to-host copies.
- `cublas`: cuBLAS handle for BLAS operations.
- `cublasLt`: cuBLASLt handle for high-performance/fused GEMM.
- `cudnn`: cuDNN handle for neural-network primitives.
- `cusparse`: cuSPARSE handle for sparse operations.
- `cusolver`: cuSOLVER handle for decompositions such as SVD.
- `curand`: cuRAND generator for random/init paths.
- `nccl`: NCCL communicator for multi-GPU synchronization.
- `workspace`: reusable scratch buffer.

The context lets operations reuse expensive handles instead of recreating them
every call.

### `cf_math_desc_cache`

```c
typedef struct cf_math_desc_cache
{
  cudnnTensorDescriptor_t tensor_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnRNNDataDescriptor_t rnn_data_desc;
  cublasLtMatrixLayout_t lt_layout;
  cf_u8 valid;
} cf_math_desc_cache;
```

This stores expensive CUDA descriptors inside a tensor.

Fields:

- `tensor_desc`: cuDNN tensor descriptor for activations, batch norm, softmax,
  and dropout.
- `filter_desc`: cuDNN filter descriptor for convolution weights.
- `rnn_data_desc`: cuDNN descriptor for recurrent sequence data.
- `lt_layout`: cuBLASLt matrix layout descriptor.
- `valid`: nonzero when the cached descriptors match current shape/dtype.

Why it exists:

Descriptor creation can be expensive. Caching lets the framework build them
once and reuse them until the tensor shape or dtype changes.

### `cf_math_storage`

```c
typedef struct cf_math_storage
{
  void *data_ptr;
  cf_usize capacity;
  cf_u32 refcount;
  cf_math_mem_flags mem_flags;
  int device_id;
  cf_math_device device;
} cf_math_storage;
```

This owns the actual memory buffer.

More than one tensor can point into the same storage. For example:

- A slice shares the parent tensor's memory.
- A reshape shares the same memory with a new shape.
- A view starts at an offset inside the same buffer.

Fields:

- `data_ptr`: beginning of the owned memory allocation.
- `capacity`: byte capacity of the allocation.
- `refcount`: how many `cf_math` views share this storage.
- `mem_flags`: allocation flags such as pinned or managed memory.
- `device_id`: GPU that owns the memory, when CUDA is used.
- `device`: CPU or CUDA memory.

Storage is freed only when `refcount` reaches zero.

### `cf_math_metadata`

```c
typedef struct cf_math_metadata
{
  cf_usize len;
  cf_usize batch;
  cf_usize strides[CF_MATH_HIGHEST_RANK];
  cf_math_shape shape;
  cf_math_layout layout;
  cf_math_device device;
  cf_math_dtype dtype;
  cf_math_mem_flags mem_flags;
  cf_math_cuda_context ctx;
} cf_math_metadata;
```

This is the descriptive part of a tensor.

Fields:

- `len`: number of logical elements.
- `batch`: usually `dim[0]`; used by batch-aware operations.
- `strides`: element strides for each dimension.
- `shape`: scalar, matrix, or general tensor label.
- `layout`: memory layout such as row-major, NCHW, NHWC, or strided.
- `device`: where the active data lives.
- `dtype`: element type.
- `mem_flags`: allocation behavior and memory hints.
- `ctx`: CUDA context snapshot associated with this tensor.

Metadata is what lets the framework dispatch operations without re-inspecting
raw memory.

### `cf_math`

```c
typedef struct cf_math
{
  cf_math_storage *storage;
  void *data;
  cf_usize byte_offset;
  cf_usize rank;
  cf_usize dim[CF_MATH_HIGHEST_RANK];
  cf_math_grad_state grad_state;
  struct cf_math *grad;
  cf_math_node *grad_fn;
  cf_math_desc_cache desc_cache;
  cf_math_metadata metadata;
} cf_math;
```

This is the main tensor object.

Fields:

- `storage`: shared memory owner.
- `data`: pointer to this tensor's first visible byte. It is usually
  `storage->data_ptr + byte_offset`.
- `byte_offset`: byte position into shared storage.
- `rank`: number of active dimensions.
- `dim`: shape dimensions.
- `grad_state`: autograd participation state.
- `grad`: optional gradient tensor.
- `grad_fn`: optional autograd graph node.
- `desc_cache`: cached CUDA descriptors.
- `metadata`: dtype, layout, strides, device, and context metadata.

Important idea:

`cf_math` does not just store a pointer. It stores enough information to know
what the pointer means.

### `cf_math_conv2d_params`

```c
typedef struct cf_math_conv2d_params
{
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int groups;
} cf_math_conv2d_params;
```

This describes how 2D convolution walks over an input image.

Fields:

- `pad_h`, `pad_w`: zeros logically added around height and width.
- `stride_h`, `stride_w`: how far the filter moves each step.
- `dilation_h`, `dilation_w`: spacing between filter elements.
- `groups`: number of channel groups. `groups == input_channels` means
  depthwise convolution.

### `cf_math_dropout_state`

```c
typedef struct cf_math_dropout_state
{
  void *descriptor;
  void *reserve;
  cf_usize reserve_size;
  float probability;
  cf_u64 seed;
} cf_math_dropout_state;
```

This stores dropout layer state.

Fields:

- `descriptor`: backend-owned descriptor, used by cuDNN paths.
- `reserve`: mask/reserve memory that must survive from forward to backward.
- `reserve_size`: bytes or elements reserved for the mask.
- `probability`: dropout probability.
- `seed`: random seed used for deterministic masks.

Forward dropout creates a mask. Backward dropout must reuse the same mask.

### `cf_math_rnn_state`

```c
typedef struct cf_math_rnn_state
{
  void *descriptor;
  void *weights;
  void *workspace;
  void *reserve;
  cf_usize weights_size;
  cf_usize workspace_size;
  cf_usize reserve_size;
  cf_math_rnn_mode mode;
} cf_math_rnn_state;
```

This is the sidecar state needed by RNN/LSTM/GRU operations.

Fields:

- `descriptor`: cuDNN RNN descriptor.
- `weights`: packed recurrent weights.
- `workspace`: scratch memory for forward/backward.
- `reserve`: saved forward data required for backward.
- `weights_size`: weight buffer size.
- `workspace_size`: workspace size.
- `reserve_size`: reserve-space size.
- `mode`: RNN, LSTM, or GRU mode.

RNN libraries usually require persistent descriptors and packed weights, so the
state must live outside a single tensor.

### `cf_math_sparse`

```c
typedef struct cf_math_sparse
{
  void *values;
  cf_i32 *row_offsets;
  cf_i32 *col_indices;
  cf_usize rows;
  cf_usize cols;
  cf_usize nnz;
  cf_math_dtype dtype;
  cf_math_device device;
} cf_math_sparse;
```

This stores a sparse matrix in CSR format.

CSR means Compressed Sparse Row.

Fields:

- `values`: nonzero values.
- `row_offsets`: where each row begins in `values` and `col_indices`.
- `col_indices`: column index for each nonzero value.
- `rows`: number of matrix rows.
- `cols`: number of matrix columns.
- `nnz`: number of nonzero values.
- `dtype`: element type for values.
- `device`: CPU or CUDA storage.

Example:

```text
dense matrix:
  [10  0  3]
  [ 0  2  0]

values:      [10, 3, 2]
col_indices: [0,  2, 1]
row_offsets: [0,  2, 3]
```

## Function Reference

This section documents every public function in `cf_math.h`.

### Primitive Helpers

#### `cf_math_g8_mul_mod`

```c
cf_u8 cf_math_g8_mul_mod(cf_u8 p, cf_u8 q);
```

Multiplies two bytes in the AES finite field GF(2^8).

This is not normal integer multiplication. AES uses polynomial arithmetic with
the `0x11B` reduction polynomial. The AES module uses this helper for
MixColumns and inverse MixColumns.

#### `cf_math_rotl8`

```c
cf_u8 cf_math_rotl8(cf_u8 x, cf_u8 n);
```

Rotates an 8-bit value left by `n` bits. The shift count wraps modulo 8.

#### `cf_math_rotr8`

```c
cf_u8 cf_math_rotr8(cf_u8 x, cf_u8 n);
```

Rotates an 8-bit value right by `n` bits. The shift count wraps modulo 8.

#### `cf_math_rotl32`

```c
cf_u32 cf_math_rotl32(cf_u32 x, cf_u8 n);
```

Rotates a 32-bit value left by `n` bits. Useful in hash and crypto code.

#### `cf_math_rotr32`

```c
cf_u32 cf_math_rotr32(cf_u32 x, cf_u8 n);
```

Rotates a 32-bit value right by `n` bits.

#### `cf_math_min_usize`

```c
cf_usize cf_math_min_usize(cf_usize a, cf_usize b);
```

Returns the smaller of two framework size values.

#### `cf_math_max_usize`

```c
cf_usize cf_math_max_usize(cf_usize a, cf_usize b);
```

Returns the larger of two framework size values.

#### `cf_math_dtype_size`

```c
cf_usize cf_math_dtype_size(cf_math_dtype dtype);
```

Returns the byte width of one element for a dtype.

Examples:

- `CF_DTYPE_F64` returns `sizeof(double)`.
- `CF_DTYPE_F32` returns `sizeof(float)`.
- `CF_DTYPE_I32` returns `sizeof(cf_i32)`.
- FP16/BF16 return 2.
- FP8 types return 1.

Returns 0 for an unknown dtype.

### CUDA Context And Workspace

#### `cf_math_context_init`

```c
cf_status cf_math_context_init(cf_math_cuda_context *ctx, int device_id);
```

Initializes a CUDA context object for one GPU device.

It creates streams and library handles when CUDA headers and libraries are
available. In a CPU-only build it returns `CF_ERR_UNSUPPORTED`.

Use this before CUDA math calls that need shared handles.

#### `cf_math_context_destroy`

```c
cf_status cf_math_context_destroy(cf_math_cuda_context *ctx);
```

Destroys handles, streams, and workspace owned by a CUDA context, then resets
the struct.

#### `cf_math_workspace_reserve`

```c
cf_status cf_math_workspace_reserve(cf_math_cuda_context *ctx, cf_usize bytes);
```

Ensures `ctx->workspace` has at least `bytes` bytes.

This is for operations that need temporary memory. The workspace grows but does
not shrink automatically.

### Memory And Lifecycle

#### `cf_math_alloc`

```c
cf_status cf_math_alloc(
  cf_math *out,
  const cf_usize dim[CF_MATH_HIGHEST_RANK],
  cf_usize rank,
  cf_math_dtype dtype,
  cf_math_device device,
  cf_math_mem_flags flags,
  cf_math_cuda_context *ctx
);
```

Allocates a tensor.

What it does:

- Computes element count from `dim` and `rank`.
- Computes byte count from dtype.
- Allocates CPU or CUDA storage.
- Sets shape, strides, dtype, layout, device, and storage metadata.
- Clears descriptor cache.

For rank 0, pass `dim == CF_NULL`; the tensor is a scalar with one value.

#### `cf_math_free`

```c
cf_status cf_math_free(cf_math *x, cf_math_cuda_context *ctx);
```

Releases a tensor's storage reference and resets the tensor object.

If other views share the same storage, the memory is kept alive until the last
reference is freed.

#### `cf_math_alloc_pinned`

```c
cf_status cf_math_alloc_pinned(...);
```

Allocates CPU memory with the pinned-memory flag. Pinned memory can be faster
for host/device transfers because the OS will not page it out.

#### `cf_math_alloc_managed`

```c
cf_status cf_math_alloc_managed(...);
```

Allocates CUDA managed memory when CUDA is available. Managed memory can be
accessed from CPU and GPU, with migration handled by CUDA.

#### `cf_math_view`

```c
cf_status cf_math_view(
  cf_math *out,
  const cf_math *x,
  cf_usize offset_elems,
  const cf_usize dim[CF_MATH_HIGHEST_RANK],
  cf_usize rank
);
```

Creates a zero-copy view into another tensor's storage.

It does not allocate a new data buffer. It increments the storage reference
count and moves `out->data` by `offset_elems`.

#### `cf_math_contiguous`

```c
cf_status cf_math_contiguous(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
```

Creates a contiguous copy of a tensor.

This is needed when a strided view must be passed to an operation that expects
normal dense memory.

#### `cf_math_to_device`

```c
cf_status cf_math_to_device(cf_math *out, const cf_math *x, int device_id, cf_math_cuda_context *ctx);
```

Copies a tensor to CUDA device memory.

In CPU-only builds this returns `CF_ERR_UNSUPPORTED`.

#### `cf_math_to_host`

```c
cf_status cf_math_to_host(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
```

Copies a tensor to CPU host memory.

If the source is already CPU-backed, this is a normal memory copy.

#### `cf_math_clone`

```c
cf_status cf_math_clone(cf_math *out, const cf_math *x, cf_math_cuda_context *ctx);
```

Creates a new tensor with the same shape, dtype, device metadata, and data
values as `x`.

Unlike a view, a clone owns separate storage.

### Initialization And Random

#### `cf_math_fill`

```c
cf_status cf_math_fill(cf_math *out, double value, cf_math_cuda_context *ctx);
```

Writes the same scalar value into every element.

#### `cf_math_zeros`

```c
cf_status cf_math_zeros(cf_math *out, cf_math_cuda_context *ctx);
```

Sets all bytes/elements to zero. This is the fastest clear path for buffers.

#### `cf_math_ones`

```c
cf_status cf_math_ones(cf_math *out, cf_math_cuda_context *ctx);
```

Writes `1` into every logical element.

#### `cf_math_rand_uniform`

```c
cf_status cf_math_rand_uniform(cf_math *out, double lo, double hi, cf_u64 seed, cf_math_cuda_context *ctx);
```

Fills a tensor with uniform random values from `[lo, hi)`.

The CPU reference path uses a deterministic local generator. The intended CUDA
path uses cuRAND PHILOX.

#### `cf_math_rand_normal`

```c
cf_status cf_math_rand_normal(cf_math *out, double mean, double stddev, cf_u64 seed, cf_math_cuda_context *ctx);
```

Fills a tensor with normally distributed random values.

The CPU reference path uses Box-Muller sampling.

#### `cf_math_rand_bernoulli`

```c
cf_status cf_math_rand_bernoulli(cf_math *out, double p, cf_u64 seed, cf_math_cuda_context *ctx);
```

Fills a tensor with `0` or `1`, where `1` appears with probability `p`.

#### `cf_math_init_xavier_uniform`

Initializes a weight tensor with Xavier/Glorot uniform values.

Use for layers where fan-in and fan-out both matter.

Formula:

```text
limit = sqrt(6 / (fan_in + fan_out))
W ~ U(-limit, limit)
```

#### `cf_math_init_xavier_normal`

Initializes a weight tensor with Xavier/Glorot normal values.

Formula:

```text
std = sqrt(2 / (fan_in + fan_out))
W ~ N(0, std)
```

#### `cf_math_init_kaiming_normal`

Initializes a tensor for ReLU-style networks with Kaiming normal values.

Formula:

```text
std = sqrt(2 / fan_in)
```

#### `cf_math_init_kaiming_uniform`

Initializes a tensor for ReLU-style networks with Kaiming uniform values.

Formula:

```text
limit = sqrt(6 / fan_in)
```

#### `cf_math_init_orthogonal`

Initializes a rank-2 tensor with approximately orthogonal rows or columns.

The CPU reference implementation uses Gram-Schmidt style orthogonalization.
The intended high-performance CUDA path is cuSOLVER SVD or QR.

#### `cf_math_init_eye`

Creates an identity-like matrix:

```text
out[i, i] = 1
all other entries = 0
```

### Elementwise Arithmetic

Elementwise operations apply one formula independently to every element.

For binary operations, `x`, `y`, and `out` are expected to have compatible
shapes and dtypes. Hot paths avoid heavy validation for performance.

#### `cf_math_add`

Computes:

```text
out = x + y
```

#### `cf_math_add_scalar`

Computes:

```text
out = x + c
```

#### `cf_math_sub`

Computes:

```text
out = x - y
```

#### `cf_math_mul`

Computes elementwise multiplication:

```text
out = x * y
```

This is not matrix multiplication.

#### `cf_math_mul_scalar`

Computes:

```text
out = x * c
```

#### `cf_math_div`

Computes:

```text
out = x / y
```

#### `cf_math_div_scalar`

Computes:

```text
out = x / c
```

#### `cf_math_pow`

Computes:

```text
out[i] = pow(x[i], n)
```

#### `cf_math_sqrt`

Computes square root elementwise.

#### `cf_math_rsqrt`

Computes reciprocal square root:

```text
out[i] = 1 / sqrt(x[i])
```

#### `cf_math_exp`

Computes exponential:

```text
out[i] = e ^ x[i]
```

#### `cf_math_log`

Computes natural logarithm.

#### `cf_math_abs`

Computes absolute value.

#### `cf_math_neg`

Computes arithmetic negation:

```text
out = -x
```

#### `cf_math_clamp`

Clamps every element to `[lo, hi]`.

#### `cf_math_sign`

Writes:

```text
-1 for negative values
 0 for zero
 1 for positive values
```

### Reductions

Reduction operations turn many values into fewer values.

#### `cf_math_sum`

Sums every element into a scalar tensor.

#### `cf_math_sum_axis`

Sums along one axis and removes that axis from the output shape.

Example:

```text
x shape:   [2, 3]
axis:      0
out shape: [3]
```

#### `cf_math_mean`

Computes average of all elements.

#### `cf_math_mean_axis`

Computes average along one axis.

#### `cf_math_var`

Computes population variance:

```text
var = mean((x - mean(x))^2)
```

#### `cf_math_std`

Computes standard deviation:

```text
std = sqrt(var)
```

#### `cf_math_norm2`

Computes L2 norm:

```text
sqrt(sum(x[i]^2))
```

#### `cf_math_norm1`

Computes L1 norm:

```text
sum(abs(x[i]))
```

#### `cf_math_max`

Returns maximum value as a scalar tensor.

#### `cf_math_min`

Returns minimum value as a scalar tensor.

#### `cf_math_argmax`

Returns index of the largest value as an I32 scalar tensor.

#### `cf_math_argmin`

Returns index of the smallest value as an I32 scalar tensor.

#### `cf_math_dot`

Computes vector dot product:

```text
sum(x[i] * y[i])
```

#### `cf_math_cumsum`

Computes prefix sums:

```text
out[i] = x[0] + x[1] + ... + x[i]
```

### Dense Linear Algebra

#### `cf_math_matmul`

Computes matrix multiplication:

```text
out = a @ b
```

For shape:

```text
a:   [M, K]
b:   [K, N]
out: [M, N]
```

The CPU path is a reference triple loop. The intended CUDA path is cuBLASLt or
cuBLAS GEMM.

#### `cf_math_matmul_t`

Matrix multiplication with optional transposition flags:

```text
out = maybe_transpose(a) @ maybe_transpose(b)
```

Use this to avoid physically transposing memory before GEMM.

#### `cf_math_matmul_batched`

Computes many matrix multiplications:

```text
out[i] = a[i] @ b[i]
```

Used for attention heads and batch processing.

#### `cf_math_linear`

Fully connected layer:

```text
out = x @ W^T + b
```

The implementation treats `W` as `[out_features, in_features]`.

#### `cf_math_linear_fused_relu`

Fully connected layer followed by ReLU:

```text
out = relu(x @ W^T + b)
```

On CUDA this should map to cuBLASLt GEMM epilogues.

#### `cf_math_linear_fused_gelu`

Fully connected layer followed by GELU:

```text
out = gelu(x @ W^T + b)
```

#### `cf_math_linear_backward_W`

Computes gradient of the linear weights.

Conceptually:

```text
dW = dL^T @ x
```

#### `cf_math_linear_backward_x`

Computes gradient of the linear input.

Conceptually:

```text
dx = dL @ W
```

#### `cf_math_linear_backward_b`

Computes gradient of the bias by summing `dL` across batch.

#### `cf_math_outer`

Computes outer product:

```text
out[i, j] = x[i] * y[j]
```

#### `cf_math_matvec`

Matrix-vector multiplication:

```text
out = A @ x
```

#### `cf_math_transpose`

Creates a transposed matrix copy:

```text
out[j, i] = a[i, j]
```

#### `cf_math_scale`

Scales every element:

```text
out = alpha * a
```

### Convolution

Convolution functions expect image tensors in a 4D shape, usually NCHW:

```text
[batch, channels, height, width]
```

#### `cf_math_conv2d_fwd`

2D convolution forward pass:

```text
out = conv2d(x, W) + b
```

CPU path is a reference loop over batch, output channel, output height, output
width, input channels, and kernel coordinates.

CUDA target: cuDNN convolution forward.

#### `cf_math_conv2d_bwd_data`

Computes convolution gradient with respect to input data:

```text
dx = d(conv) / dx
```

CUDA target: cuDNN backward data.

#### `cf_math_conv2d_bwd_filter`

Computes convolution gradient with respect to filters:

```text
dW = d(conv) / dW
```

CUDA target: cuDNN backward filter.

#### `cf_math_conv2d_bwd_bias`

Computes bias gradient by summing `dL` over batch and spatial dimensions.

#### `cf_math_conv2d_depthwise_fwd`

Depthwise convolution forward pass.

Depthwise means each input channel has its own filter group. This sets
`groups = input_channels`.

#### `cf_math_conv2d_dilated_fwd`

Dilated convolution forward pass.

Dilation spaces out kernel points, increasing receptive field without
increasing kernel parameter count.

#### `cf_math_conv2d_transpose_fwd`

Transposed convolution forward pass.

The CPU reference path uses the backward-data convolution formulation.

#### `cf_math_conv1d_fwd`

1D convolution API entry point.

Currently routes through the 2D convolution surface.

#### `cf_math_conv3d_fwd`

3D convolution forward pass for tensors shaped like:

```text
[batch, channels, depth, height, width]
```

CPU path is a reference implementation. CUDA target: cuDNN 3D convolution.

### Normalization

Normalization rescales data to make optimization more stable.

#### `cf_math_bn_fwd_train`

Batch normalization training forward pass.

It computes per-channel mean and inverse variance, saves them when requested,
then normalizes:

```text
y = (x - mean) * inv_std * gamma + beta
```

#### `cf_math_bn_fwd_infer`

Batch normalization inference forward pass.

Uses provided running mean and variance instead of computing batch statistics.

#### `cf_math_bn_bwd`

Batch normalization backward surface.

Current CPU reference returns input gradient by cloning `dL`; full BN gradient
math is reserved for the complete training backend.

#### `cf_math_ln_fwd`

Layer normalization forward pass.

Normalizes across the last dimension of each row/instance.

#### `cf_math_ln_bwd`

Layer normalization backward surface.

Current CPU reference clones `dL` into `dx`; full derivative handling is future
work.

#### `cf_math_in_fwd`

Instance normalization forward surface.

Currently routes through layer norm reference behavior.

#### `cf_math_gn_fwd`

Group normalization forward surface.

Currently routes through layer norm reference behavior.

#### `cf_math_rms_norm_fwd`

RMS normalization forward pass:

```text
rms = sqrt(mean(x^2) + eps)
y = x / rms * gamma
```

Used by LLaMA-style transformer blocks.

#### `cf_math_rms_norm_bwd`

RMS normalization backward surface.

Current CPU reference clones `dL` into `dx`; full derivative handling is future
work.

### Activations

Activation functions transform tensor values element by element.

#### `cf_math_relu`

Rectified linear unit:

```text
max(0, x)
```

#### `cf_math_relu_bwd`

ReLU backward pass using the forward output `y`.

If `y > 0`, gradient passes through. Otherwise it becomes zero.

#### `cf_math_leaky_relu`

Leaky ReLU:

```text
x       if x > 0
alpha*x otherwise
```

#### `cf_math_elu`

ELU activation:

```text
x                 if x >= 0
alpha*(exp(x)-1)  otherwise
```

#### `cf_math_sigmoid`

Sigmoid activation:

```text
1 / (1 + exp(-x))
```

#### `cf_math_sigmoid_bwd`

Sigmoid backward pass using output `y`:

```text
dy * y * (1 - y)
```

#### `cf_math_tanh`

Hyperbolic tangent activation.

#### `cf_math_tanh_bwd`

Tanh backward pass using output `y`:

```text
dy * (1 - y^2)
```

#### `cf_math_gelu`

Exact GELU:

```text
0.5 * x * (1 + erf(x / sqrt(2)))
```

#### `cf_math_gelu_approx`

Approximate GELU using tanh. This is common in transformer models because it is
fast and accurate enough for many networks.

#### `cf_math_gelu_bwd`

GELU backward pass.

#### `cf_math_swish`

Swish activation:

```text
x * sigmoid(beta * x)
```

#### `cf_math_silu`

SiLU activation. It is Swish with `beta = 1`.

#### `cf_math_softplus`

Smooth approximation of ReLU:

```text
log(1 + exp(x))
```

#### `cf_math_mish`

Mish activation:

```text
x * tanh(softplus(x))
```

### Softmax And Loss

#### `cf_math_softmax_fwd`

Computes stable softmax along an axis:

```text
exp(x - max(x)) / sum(exp(x - max(x)))
```

Subtracting the maximum avoids large exponentials.

#### `cf_math_softmax_bwd`

Softmax backward pass:

```text
dx = y * (dy - dot(y, dy))
```

#### `cf_math_log_softmax_fwd`

Computes log softmax.

This is more numerically stable than applying `log` after a naive softmax.

#### `cf_math_log_softmax_bwd`

Log-softmax backward pass:

```text
dx = dy - exp(y) * sum(dy)
```

#### `cf_math_cross_entropy`

Computes cross entropy from logits and target labels/probabilities.

It can also write `dx` as:

```text
softmax(logits) - target
```

#### `cf_math_cross_entropy_bwd`

Computes the fused cross-entropy gradient:

```text
dx = probability - target
```

#### `cf_math_nll_loss`

Negative log likelihood loss.

It gathers the log probability for each class label and averages the negative
values.

#### `cf_math_mse_loss`

Mean squared error:

```text
mean((y - target)^2)
```

#### `cf_math_mse_loss_bwd`

MSE backward:

```text
2 * (y - target) / n
```

#### `cf_math_bce_loss`

Binary cross entropy:

```text
-(target*log(p) + (1-target)*log(1-p))
```

#### `cf_math_huber_loss`

Huber loss, also called smooth L1 loss.

Small errors use squared loss. Large errors use linear loss.

#### `cf_math_focal_loss`

Focal loss.

This downweights easy examples and is often used in class-imbalanced detection
tasks.

### Attention

Attention functions are the building blocks of transformer attention.

#### `cf_math_attn_scores`

Computes attention scores:

```text
scores = Q @ K^T * scale
```

Usually:

```text
scale = 1 / sqrt(head_dim)
```

#### `cf_math_attn_mask_add`

Adds an attention mask to the scores.

Masks usually contain `0` for allowed positions and a very negative value for
blocked positions.

#### `cf_math_attn_softmax`

Applies softmax to attention scores.

#### `cf_math_attn_context`

Computes context:

```text
context = attention_probabilities @ V
```

#### `cf_math_attn_proj`

Applies the output projection:

```text
out = context @ W_o
```

#### `cf_math_mha_fwd`

Multi-head attention forward surface.

The CPU reference path runs scores, softmax, context, and projection.

#### `cf_math_mha_bwd`

Multi-head attention backward surface.

This currently returns `CF_ERR_UNSUPPORTED` because complete backward requires
saved forward state and a larger layer-state contract.

#### `cf_math_attn_dropout_fwd`

Dropout applied to attention probabilities.

This is a wrapper around the general dropout forward function.

#### `cf_math_rope_fwd`

Rotary position embedding forward pass.

It rotates pairs of features using cosine and sine tables:

```text
[a, b] -> [a*cos - b*sin, a*sin + b*cos]
```

#### `cf_math_rope_bwd`

Backward pass for rotary position embedding.

It applies the inverse rotation to gradients.

#### `cf_math_causal_mask`

Creates a causal attention mask.

For position `i`, future positions `j > i` receive `-infinity`.

### Dropout

#### `cf_math_dropout_fwd`

Dropout forward pass:

```text
out = x * mask / (1 - p)
```

It stores the mask in `cf_math_dropout_state.reserve` so backward can reuse it.

#### `cf_math_dropout_bwd`

Dropout backward pass:

```text
dx = dy * mask / (1 - p)
```

#### `cf_math_dropout_train_set`

Updates dropout probability for training or inference.

When inference is requested, probability becomes `0`.

### Embedding

#### `cf_math_embed_fwd`

Embedding lookup:

```text
out[token, :] = W[index[token], :]
```

Used for token embeddings in language models.

#### `cf_math_embed_bwd`

Embedding backward pass.

It accumulates gradients into rows of `dW` touched by the input indices.

#### `cf_math_embed_bwd_atomic`

Atomic-style embedding backward surface.

The CPU path is the same as `cf_math_embed_bwd`. CUDA should use `atomicAdd`
or a sort/segment-reduce path to handle repeated token indices.

### RNN, LSTM, And GRU

These functions are public surfaces for recurrent neural-network operations.
They currently return `CF_ERR_UNSUPPORTED` until the cuDNN RNN layer-state
implementation is completed.

#### `cf_math_rnn_fwd_train`

Training forward pass for a basic RNN.

Needs saved reserve space for backward.

#### `cf_math_rnn_fwd_infer`

Inference forward pass for a basic RNN.

#### `cf_math_rnn_bwd_data`

RNN backward pass with respect to input data and hidden state.

#### `cf_math_rnn_bwd_weights`

RNN backward pass with respect to recurrent weights.

#### `cf_math_lstm_fwd_train`

Training forward pass for LSTM.

LSTM has both hidden state and cell state.

#### `cf_math_lstm_bwd_data`

LSTM backward pass with respect to data, hidden state, and cell state.

#### `cf_math_gru_fwd_train`

Training forward pass for GRU.

### Sparse Operations

#### `cf_math_spmv`

Sparse matrix-vector multiplication:

```text
out = A_sparse @ x
```

Uses CSR row offsets and column indices.

#### `cf_math_spmm`

Sparse matrix-dense matrix multiplication:

```text
out = A_sparse @ B_dense
```

#### `cf_math_spgemm`

Sparse matrix-sparse matrix multiplication.

The CPU reference path computes through a dense temporary and converts back to
CSR.

#### `cf_math_dense_to_csr`

Converts a dense rank-2 tensor into CSR sparse format.

Values whose absolute value is less than or equal to `threshold` are skipped.

#### `cf_math_csr_to_dense`

Converts CSR sparse format back to a dense tensor.

#### `cf_math_sparse_attn`

Sparse attention helper:

```text
out = A_sparse @ V
```

It currently routes through sparse-dense matrix multiplication.

### Optimizer Math

Optimizer functions update model parameters or gradients.

#### `cf_math_sgd_step`

Stochastic gradient descent:

```text
W = W - lr * g
```

#### `cf_math_sgd_momentum`

SGD with momentum:

```text
v = momentum * v + g
W = W - lr * v
```

#### `cf_math_adam_step`

Adam optimizer update:

```text
m = beta1*m + (1-beta1)*g
v = beta2*v + (1-beta2)*g^2
W = W - lr * corrected_m / (sqrt(corrected_v) + eps)
```

#### `cf_math_adamw_step`

AdamW optimizer update.

This is Adam plus decoupled weight decay.

#### `cf_math_rmsprop_step`

RMSProp optimizer:

```text
v = beta*v + (1-beta)*g^2
W = W - lr * g / (sqrt(v) + eps)
```

#### `cf_math_grad_clip_norm`

Clips a gradient tensor by total L2 norm.

If the norm is already under the limit, the tensor is unchanged.

#### `cf_math_grad_clip_value`

Clips every gradient value into:

```text
[-clip, clip]
```

#### `cf_math_weight_decay`

Adds weight decay to a gradient:

```text
g = g + decay * W
```

#### `cf_math_lr_scale`

Scales a gradient tensor by a scalar.

Useful for learning-rate schedules or gradient accumulation.

#### `cf_math_grad_allreduce`

Multi-GPU gradient all-reduce surface.

For `world_size <= 1`, it succeeds as a no-op. Multi-GPU reduction requires
NCCL integration.

#### `cf_math_grad_zero`

Zeros a gradient tensor.

### Shape Manipulation

Shape operations usually change metadata, not data.

#### `cf_math_reshape`

Creates a zero-copy view with a new shape.

The element count must remain the same.

#### `cf_math_permute`

Reorders axes.

The CPU reference path creates a copied output. A future optimized path may use
metadata-only views for some cases.

#### `cf_math_squeeze`

Removes dimensions with size 1.

Example:

```text
[1, 3, 1, 4] -> [3, 4]
```

#### `cf_math_unsqueeze`

Adds a dimension of size 1.

Example:

```text
[3, 4] at axis 0 -> [1, 3, 4]
```

#### `cf_math_expand`

Creates a broadcast view.

Broadcast axes use stride 0, which means multiple logical positions read the
same memory element.

#### `cf_math_concat`

Concatenates tensors along an axis.

The CPU path copies values into the output.

#### `cf_math_split`

Splits a tensor into equal zero-copy views along an axis.

#### `cf_math_slice`

Creates a zero-copy slice view.

It adjusts the byte offset and shape while sharing storage.

#### `cf_math_pad`

Creates a padded copy of a tensor.

New positions are zero-filled.

#### `cf_math_flatten`

Combines a range of axes into one axis.

Example:

```text
[2, 3, 4] flatten axes 1..2 -> [2, 12]
```

## Current Implementation Notes

The public function map is intentionally broad. Not every high-level training
operation has its final high-performance backend yet.

Currently:

- Dense CPU reference behavior exists for most arithmetic, reduction, linalg,
  activation, softmax, loss, dropout, embedding, optimizer, sparse, shape, and
  several convolution/normalization paths.
- CUDA allocation/copy/context hooks are guarded by CUDA availability.
- RNN/LSTM/GRU functions return `CF_ERR_UNSUPPORTED` until the cuDNN RNN state
  implementation is completed.
- MHA backward returns `CF_ERR_UNSUPPORTED` until saved forward state and weight
  gradient contracts are added.
- Multi-GPU all-reduce returns `CF_OK` only for `world_size <= 1` until NCCL is
  wired into training loops.

This is deliberate. The public surface is shaped for the final math layer, and
the implementation can become more optimized without changing user-facing
function names.

## How To Think About Performance

The layer separates operation meaning from backend execution.

The meaning is stable:

```text
cf_math_matmul means matrix multiplication.
cf_math_relu means ReLU.
cf_math_dropout_fwd means dropout forward.
```

The backend can improve over time:

```text
CPU loop today
custom CUDA kernel tomorrow
cuBLASLt/cuDNN/cuSPARSE path where appropriate
fused path when it saves memory traffic
```

Good performance rules:

- Avoid allocating inside training loops.
- Reuse `cf_math_cuda_context`.
- Reuse workspace.
- Use views for reshape/slice/split when possible.
- Keep dtype/layout/device metadata correct.
- Use fused operations such as linear+activation when the backend supports it.
- Prefer vendor libraries for GEMM, convolution, normalization, sparse ops, and
  recurrent layers.
