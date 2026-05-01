# CF CPU Backend And Memory Upgrade

This document explains the upgraded math and allocator paths. The production
build expects OpenBLAS, mimalloc, and OpenMP. `CF_ENABLE_CPU_FALLBACK=1` keeps a
small reference CPU path for development machines that do not have those
packages installed.

## Build Dispatch

- `CF_MATH_USE_OPENBLAS=1`: CPU BLAS-shaped operations use the CBLAS interface.
- `CF_ALLOC_USE_MIMALLOC=1`: the default `cf_alloc` routes to mimalloc.
- `CF_MATH_USE_OPENMP=1`: CPU loops use OpenMP SIMD/parallel pragmas.
- `CF_ENABLE_CPU_FALLBACK=1`: compiles without OpenBLAS/mimalloc and uses the
  internal reference loops. This mode is for testing portability, not the fast
  production path.

## Allocator Functions

`cf_alloc_new` initializes the framework default allocator. In production it
uses `mi_malloc`, `mi_realloc`, and `mi_free`, so arrays, strings, arenas, pools,
and math storage that use `cf_alloc` inherit mimalloc behavior.

`cf_alloc_aligned` allocates an aligned block. If the allocator has an aligned
callback, it is used directly; the default production callback calls
`mi_malloc_aligned`. If a custom allocator only provides basic `alloc/free`,
the helper over-allocates and stores the original pointer immediately before the
aligned address.

`cf_alloc_aligned_free` releases memory returned by `cf_alloc_aligned`. It uses
the allocator's aligned free callback when available, otherwise it recovers the
stored original pointer and calls the allocator's normal free callback.

## Arena And Pool Functions

`cf_arena_init` creates a 64-byte aligned owning bump arena. It preserves the old
API and keeps fixed-capacity behavior.

`cf_arena_init_ex` creates an owning arena with explicit alignment and optional
chunk growth. Growth appends chunks; existing allocations are not moved. This is
safe for temporary arenas because previously returned pointers remain valid.

`cf_arena_init_with_buffer` wraps caller-owned storage. It never frees the
buffer and never grows because the caller controls the backing memory.

`cf_arena_alloc` is a bump allocation. It aligns the actual returned address,
not just the byte offset, updates `offset`, and tracks `high_water`. Growable
owning arenas allocate a new chunk when the current chunk cannot satisfy a
request.

`cf_arena_reset` resets chunk offsets to zero without freeing memory. This is
the fast reuse path for frame/workspace memory.

`cf_pool_init` creates a 64-byte aligned fixed-size block pool. Each block stride
is rounded up to hold the free-list link and satisfy alignment.

`cf_pool_init_ex` is the explicit-alignment variant. It is useful for small
objects repeatedly allocated in hot paths.

`cf_pool_alloc`, `cf_pool_free`, and `cf_pool_reset` operate through the internal
free list and do not call malloc/free after initialization.

## Math Dispatch Functions

`cf_math_op` performs in-place add/sub/mul/div. CPU uses typed SIMD/OpenMP loops
for `F32`, `F64`, and `I32`. CUDA launches the existing elementwise kernels on
the handler stream and returns after enqueue/error-check.

`cf_math_unary` performs neg/relu/gelu/exp/log/sqrt/sigmoid/tanh. CPU uses
single-pass SIMD/OpenMP loops without temporary heap buffers. CUDA keeps the
existing kernels.

`cf_math_scalar` applies add/sub/mul/div with a scalar. CPU supports `F32`,
`F64`, and `I32` through SIMD/OpenMP loops. CUDA keeps the existing scalar
kernels.

`cf_math_reduce_sum` and `cf_math_reduce_mean` reduce into a caller-bound
one-element view. CPU uses OpenMP reductions. CUDA uses CUB `DeviceReduce::Sum`
and a tiny mean finalization kernel.

`cf_math_dot` computes a vector dot product into a one-element output. CPU uses
`cblas_sdot` or `cblas_ddot` in production. CUDA uses cuBLAS dot with device
pointer mode so the result stays in the output tensor.

`cf_math_matmul` multiplies `[M, K] @ [K, N] -> [M, N]`. CPU production uses
`cblas_sgemm` or `cblas_dgemm` with `CblasRowMajor`. CUDA keeps the cuBLAS
row-major mapping. All operands must be compact row-major; unsupported strided
layouts return `CF_ERR_UNSUPPORTED` instead of copying.

`cf_math_matvec` computes `[M, N] @ [N] -> [M]`. CPU production uses
`cblas_sgemv` or `cblas_dgemv`. CUDA uses cuBLAS GEMV with the same row-major
mapping strategy as matmul.

`cf_math_batched_matmul` computes `[B, M, K] @ [B, K, N] -> [B, M, N]`. CPU
production performs zero-copy per-batch CBLAS GEMM calls. CUDA uses cuBLAS
strided-batched GEMM. Compact row-major layout is required.

`cf_math_cpy_h2d` queues a host-to-view copy. For CUDA this is asynchronous, so
the caller must keep the host buffer alive until a later sync or host-visible
copy.

`cf_math_cpy_d2h` copies a view to host memory and synchronizes before returning
so the host buffer is valid immediately.

`cf_math_handle_sync` is the explicit completion point. CPU handlers return
immediately; CUDA handlers synchronize the shared context stream. Benchmarks
should sync immediately before stopping timers.

## AI Helpers

`cf_ai_dense_forward` still calls `cf_math_matmul`, then adds bias, then applies
activation. CPU bias add uses a flat SIMD/OpenMP loop. CUDA bias add remains a
small stream kernel and no longer synchronizes internally.

`cf_ai_loss_forward` computes MSE or binary cross entropy into a caller-bound
scalar. CPU uses OpenMP reductions. CUDA uses CUB transform reductions and only
synchronizes when the caller explicitly asks or copies the scalar back to host.

## Performance Notes

- Large CPU matrix operations now enter vendor-tuned OpenBLAS kernels.
- Elementwise and reduction paths avoid heap temporaries and are vectorizable.
- mimalloc-backed aligned allocations improve cache-line and SIMD friendliness.
- Handler arenas and pools reuse memory without malloc/free in hot paths.
- CUDA math work remains asynchronous by default, which allows batching multiple
  operations on the stream before one explicit synchronization.
