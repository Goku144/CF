# Cypher Math Layer Mental Model

## Core View

`cf_math` is a non-owning view. It does not allocate memory by itself. It points
at a byte slice owned by a `cf_math_handle_t`, and it borrows shape information
from `cf_math_metadata`.

Think of the layer as:

```text
cf_math_cuda_context  owns CUDA/session handles
cf_math_handle_t      owns storage and allocation policy
cf_math_metadata      owns shape/stride/layout description
cf_math               points at one slice inside a handler
```

Binding a view reserves or reuses a slice from the handler. Unbinding returns
that slice to the handler free-list when no other view references it.

## Operation Rule

Math operations do not allocate output storage. Every input and output must
already be bound by the caller. Operations also do not perform hidden host/GPU
copies. Transfers stay explicit through `cf_math_cpy_h2d` and
`cf_math_cpy_d2h`.

The hot execution path is separate from the checked setup path:

```c
cf_math_op_check(CF_MATH_OP_ADD, &a, &b); /* setup/debug */
cf_math_op(CF_MATH_OP_ADD, &a, &b);       /* hot path: a += b */
```

`cf_math_op` is intentionally lean. AI graph/layer code should validate once
while building the layer, then call hot operations during training/inference.

## Public Math Functions

| Function | Purpose |
| --- | --- |
| `cf_math_metadata_init` | Build reusable rank/dim/stride/layout metadata. |
| `cf_math_bind` | Bind a view to handler-owned storage using metadata. |
| `cf_math_unbind` | Release a view's active slice back to handler tracking. |
| `cf_math_rebind` | Unbind and bind a view to new metadata/storage. |
| `cf_math_cpy_h2d` | Copy CPU data into a bound view. |
| `cf_math_cpy_d2h` | Copy a bound view into CPU data. |
| `cf_math_op_check` | Validate binary operation compatibility for setup/debug. |
| `cf_math_op` | Hot in-place binary add/sub/mul/div: `op1 = op1 op op2`. |
| `cf_math_op_out` | Out-of-place binary op: copy `a` into `out`, then operate. |
| `cf_math_unary` | In-place unary neg/relu/gelu/exp/log/sqrt/sigmoid/tanh. |
| `cf_math_unary_out` | Out-of-place unary op. |
| `cf_math_scalar` | In-place scalar add/sub/mul/div. |
| `cf_math_scalar_out` | Out-of-place scalar op. |
| `cf_math_reduce_sum` | Reduce all elements into a one-element sum output. |
| `cf_math_reduce_mean` | Reduce all elements into a one-element mean output. |
| `cf_math_matmul` | 2D row-major matrix multiplication. |
| `cf_math_print_shape` | Print shape, dtype, device, and byte-slice summary. |
| `cf_math_print_tensor` | Print tensor values through explicit host copy when needed. |

Primitive helpers such as `cf_math_rotl8`, `cf_math_rotr32`, and
`cf_math_g8_mul_mod` remain small standalone utilities.

## Operation Families

Binary operations use `CF_MATH_OP_ADD`, `CF_MATH_OP_SUB`, `CF_MATH_OP_MUL`, and
`CF_MATH_OP_DIV`. They support `F32`, `F64`, and `I32`.

Unary operations use `CF_MATH_OP_NEG`, `CF_MATH_OP_RELU`, `CF_MATH_OP_GELU`,
`CF_MATH_OP_EXP`, `CF_MATH_OP_LOG`, `CF_MATH_OP_SQRT`,
`CF_MATH_OP_SIGMOID`, and `CF_MATH_OP_TANH`. They support `F32` and `F64`.

Reductions write to an already-bound one-element view. `I32` mean uses integer
division. Matmul V1 supports 2D row-major `F32` and `F64`.

## CPU And CUDA Dispatch

CPU handlers execute flat typed loops. This keeps reference behavior simple and
easy to test.

CUDA handlers dispatch to custom kernels for elementwise, unary, and scalar
operations. Reductions use CUB `DeviceReduce::Sum`, with a tiny finalization
kernel for mean. Matmul uses cuBLAS with the row-major contract mapped onto
cuBLAS' column-major interface. CUDA code is compiled only when
`CF_CUDA_AVAILABLE` is set.

## Gradients

Core V1 is forward-only. The `grad` and `grad_fn` fields stay available on
`cf_math`, but automatic tape traversal is not implemented here.

The intended next step is manual backward helpers. That keeps Layer 0 fast and
predictable while giving the future AI layer a clean place to validate graph
shape, cache operation choices, and call the hot math functions repeatedly.

# Cypher AI Core Mental Model

The AI layer sits directly above `cf_math`. It does not own raw memory. Handlers
own memory, dense layers own bound `cf_math` views, and models own ordered layer
arrays supplied by the caller.

```text
cf_math_handle_t  owns parameter or activation storage
cf_ai_dense       owns weights/bias/output views plus metadata
cf_ai_model       owns the forward order over caller-provided layers
cf_ai_loss_*      writes one scalar loss into a caller-bound output
```

Dense V1 is a forward-only layer:

```text
input  [batch, in_features]
weight [in_features, out_features]
bias   [out_features]
output [batch, out_features]
```

Forward execution computes `output = input @ weights`, adds vector bias across
each batch row, then applies the optional activation. The bias add is the only
small AI-owned operation here because the math layer intentionally does not
pretend vector broadcasting is scalar math.

## Public AI Functions

| Function | Purpose |
| --- | --- |
| `cf_ai_dense_init` | Bind a dense layer's weights, bias, and output views from existing handlers. |
| `cf_ai_dense_forward` | Run dense matmul, row-wise bias add, and optional activation. |
| `cf_ai_dense_destroy` | Unbind the layer views; it does not destroy handlers. |
| `cf_ai_model_init` | Attach a caller-owned dense layer array to a model descriptor. |
| `cf_ai_model_forward` | Run layers sequentially and return the final output view. |
| `cf_ai_model_destroy` | Destroy all layer view bindings and clear the model descriptor. |
| `cf_ai_loss_forward` | Compute one-element MSE or binary cross entropy loss. |
| `cf_ai_dense_backward` | Manual-backward API boundary; returns `CF_ERR_UNSUPPORTED` in this batch. |
| `cf_ai_loss_backward` | Manual loss-backward API boundary; returns `CF_ERR_UNSUPPORTED` in this batch. |

## AI Execution Rule

The no-hidden-allocation/no-hidden-copy rule still applies. `cf_ai_dense_init`
binds views into caller-provided handlers, but no forward or loss call allocates
temporary tensors. Host/device transfers remain explicit through the math copy
APIs.

CPU execution uses the `cf_math` CPU kernels plus small typed loops for bias and
loss. CUDA execution uses `cf_math` CUDA matmul/unary kernels, a small
compile-gated bias kernel, and CUB-backed reductions for loss forward.

Gradients are now a named boundary, not an automatic tape. Manual backward will
come next so layer code can own its cached forward values and call Layer 0 math
directly during backward passes.
