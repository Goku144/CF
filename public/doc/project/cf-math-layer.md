# CF Math Layer Guide

The current `cf_math` layer is built around a small split:

```text
cf_math          non-owning math view
cf_math_metadata reusable shape/stride/layout description
cf_math_handle   runtime/storage handler
cf_math_arena    handler storage slice tracker
cf_math_node     optional graph lineage node
```

`cf_math` does not own memory. It is a redirectable view over data. A math view
can point to different metadata, a different handler, a gradient view, or a
graph node without moving the underlying storage.

## Why This Is Faster

The older tensor shape put too much ownership into each tensor. The new shape is
faster because the expensive work is moved into reusable objects:

- metadata can be prebuilt once for known model shapes;
- handlers can preallocate CUDA storage for known training stages;
- binding a tensor usually only consumes a slice from a handler arena;
- unbinding returns unused slices to a small free-list;
- rebinding changes pointers instead of reconstructing tensors;
- CUDA handles, streams, workspace, and descriptors stay in the handler/context;
- handlers can be tagged with `optimized_for` so later scheduling can choose a
  handler prepared for matmul, attention, reductions, transfers, and so on.

For a static or mostly static model, the framework can build the expected
metadata and handlers before the hot training loop. During the loop, most tensor
setup becomes pointer binding and offset arithmetic.

## Files

```text
public/inc/MATH/cf_math.h          public API and struct layout
public/inc/MATH/cf_math_storage.h  storage helper API
lib/src/MATH/cf_math.cu            metadata, printing, and binding helpers
lib/src/MATH/cf_math_storage.cu    CUDA context lifecycle, allocation, and arena tracking
```

## Core Objects

### `cf_math`

```c
struct cf_math
{
  void *data;
  cf_usize byte_offset;
  cf_usize byte_size;

  cf_math_metadata *metadata;
  cf_math_handle_t *handler;

  cf_math *grad;
  cf_math_node *grad_fn;
  cf_math_grad_state grad_state;
};
```

This is a non-owning math view. It stores the current visible data pointer and
points to the objects that explain and manage that data.

- `data`: visible pointer for this view.
- `byte_offset`, `byte_size`: slice inside the handler storage.
- `metadata`: shape/stride/layout view.
- `handler`: runtime and storage manager.
- `grad`: optional gradient view.
- `grad_fn`: optional operation node that produced this view.
- `grad_state`: local autograd participation flag.

### `cf_math_metadata`

```c
struct cf_math_metadata
{
  cf_usize rank;
  cf_usize dim[CF_MATH_MAX_RANK];
  cf_usize strides[CF_MATH_MAX_RANK];
  cf_usize len;
  cf_math_shape shape;
  cf_math_layout layout;
};
```

Metadata describes how to read the data. It does not own memory. Reusing
metadata is cheap and lets many math views share the same shape description.

### `cf_math_handle`

```c
struct cf_math_handle
{
  cf_math_handle_opt optimized_for;
  cf_math_desc_cache desc_cache;
  cf_math_cuda_context *cuda_ctx;
  cf_math_storage storage;
};
```

The handler owns descriptor cache and storage arena state. It does not own the
CUDA context; `cuda_ctx` points to shared runtime state that must outlive the
handler. `optimized_for` marks which operation classes the handler is prepared
for.

### `cf_math_arena`

```c
struct cf_math_arena
{
  cf_usize offset;
  cf_usize capacity;
  cf_math_memory_block free_blocks[CF_MATH_MAX_FREE_BLOCKS];
  cf_usize free_count;
  cf_math_memory_block active_blocks[CF_MATH_MAX_ACTIVE_BLOCKS];
  cf_usize active_count;
};
```

The math arena tracks byte slices inside the base allocation stored in
`storage.allocator.backend`:

- `offset`: next arena byte offset.
- `capacity`: allocated bytes.
- `free_blocks`: reusable released slices.
- `active_blocks`: slices currently used by one or more views.

### `cf_math_storage`

```c
struct cf_math_storage
{
  cf_math_arena arena;
  cf_math_dtype dtype;
  cf_math_device device;
  cf_math_allocator allocator;
};
```

Storage combines the arena with its allocation interpretation:

- `allocator.backend`: base allocation pointer owned by the handler.
- `allocator.mem_flag`: allocation policy.
- `dtype`, `device`: element and backend interpretation.

### `cf_math_memory_block`

```c
struct cf_math_memory_block
{
  cf_usize offset;
  cf_usize size;
  cf_usize ref_count;
};
```

Free blocks use `offset` and `size`. Active blocks also use `ref_count`, so
unbind can release memory only after the last view stops using a slice.

## CUDA Context

```c
struct cf_math_cuda_context
{
  int device_id;
  cudaStream_t stream;
  cublasHandle_t cublas;
  cublasLtHandle_t cublasLt;
  cudnnHandle_t cudnn;
  cusparseHandle_t cusparse;
  cusolverDnHandle_t cusolverDn;
  curandGenerator_t curand;
  cf_math_cuda_workspace cuda_workspace;
};
```

The CUDA context owns reusable backend handles and operation scratch workspace.
It is initialized once and reused by handlers.

## Memory Flags

`cf_math_mem_flags` controls storage allocation:

- `CF_MATH_MEM_DEFAULT`: device allocation through `cudaMalloc`.
- `CF_MATH_MEM_PINNED`: pinned host allocation through `cudaHostAlloc`.
- `CF_MATH_MEM_MANAGED`: managed allocation through `cudaMallocManaged`.
- `CF_MATH_MEM_POOLED`: async stream allocation through `cudaMallocAsync`.
- `CF_MATH_MEM_ALIGNED128`: requires base and slice offsets to be 128-byte aligned.
- `CF_MATH_MEM_READ_ONLY`: marks managed memory read-mostly when possible.
- `CF_MATH_MEM_PEER_MAPPED`: reserved for future multi-GPU peer mapping.

Unsupported or invalid flag combinations return a `cf_status` error instead of
being silently ignored.

## Lifecycle API

```c
cf_status cf_math_cuda_context_init(cf_math_cuda_context *ctx, cf_usize bytes, int device_id);
cf_status cf_math_cuda_context_destroy(cf_math_cuda_context *ctx);
cf_status cf_math_cuda_context_reserve(cf_math_cuda_context *ctx, cf_usize bytes);

cf_status cf_math_metadata_init(
  cf_math_metadata *metadata,
  cf_usize dim[CF_MATH_MAX_RANK],
  cf_usize rank,
  cf_math_shape shape,
  cf_math_layout layout
);

cf_status cf_math_print_shape(const cf_math *x);

cf_status cf_math_handle_init(
  cf_math_handle_t *handler,
  cf_math_cuda_context *ctx,
  cf_math_dtype dtype,
  cf_math_device device,
  cf_math_mem_flags flags,
  cf_math_handle_opt optimized_for,
  cf_usize capacity
);

cf_status cf_math_handle_reserve(cf_math_handle_t *handler, cf_usize bytes);
cf_status cf_math_handle_alloc(cf_math_handle_t *handler, cf_usize bytes, void **ptr);
void cf_math_handle_reset(cf_math_handle_t *handler);
cf_status cf_math_handle_destroy(cf_math_handle_t *handler);

cf_status cf_math_bind(cf_math *x, cf_math_handle_t *handler, cf_math_metadata *metadata);
cf_status cf_math_unbind(cf_math *x);
cf_status cf_math_rebind(cf_math *x, cf_math_handle_t *handler, cf_math_metadata *metadata);
```

## Binding Rules

`cf_math_bind` computes the byte size from `metadata->len` and the handler dtype,
reserves storage if necessary, allocates a slice, records that slice as active,
and points the math view at it.

`cf_math_unbind` clears the view. If no other active view uses the same slice,
that slice goes into the handler free-list. If another view still uses it, the
active block reference count is only decremented.

`cf_math_rebind` unbinds automatically and then binds to the new handler and
metadata. No manual `free_current` flag is needed.

`cf_math_print_shape` prints a readable summary of a math view: shape kind,
rank, dimensions, strides, length, layout, dtype, device, and byte slice.

## Example

```c
cf_math_cuda_context ctx = {0};
cf_math_handle_t handler = {0};
cf_math_metadata meta = {0};
cf_math x = {0};
cf_usize dim[CF_MATH_MAX_RANK] = {2, 2};

cf_math_cuda_context_init(&ctx, 0, 0);
cf_math_metadata_init(&meta, dim, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR);
cf_math_handle_init(
  &handler,
  &ctx,
  CF_MATH_DTYPE_F32,
  CF_MATH_DEVICE_CUDA,
  CF_MATH_MEM_MANAGED | CF_MATH_MEM_ALIGNED128,
  CF_MATH_HANDLE_OPT_MATMUL,
  0
);

cf_math_bind(&x, &handler, &meta);
cf_math_print_shape(&x);
cf_math_unbind(&x);

cf_math_handle_destroy(&handler);
cf_math_cuda_context_destroy(&ctx);
```

This example allocates no operation result itself. It only demonstrates the
runtime binding layer.
