# Cypher F16 Math Mental Model

## Core View

`cf_math` is a non-owning view into storage owned by `cf_math_handle`.
Shape and backend descriptor information lives in `cf_math_desc`.

```text
cf_math_context    owns CUDA/cuBLAS/cuDNN backend handles
cf_math_workspace  owns stream and scratchpad
cf_math_handle     owns arena storage and points at context/workspace
cf_math_desc       owns rank, dims, strides, dtype, and backend descriptors
cf_math            points at one byte slice plus one descriptor
```

Hot math calls do not allocate output tensors and do not copy between host and
device. The caller binds every input, output, gradient, optimizer state, and
cache tensor before launching work.

## Implemented F16 Functions

The f16 backend now covers:

- packed binary ops: `add`, `sub`, `mul`, `div`
- packed unary ops: `neg`, `sqrt`, `exp`, `log`, `tanh`, `relu`, `sigmoid`, `gelu`
- reductions: `reduce_sum`, `reduce_mean`
- cuBLASLt paths: `matmul`, `matmul_trans_b`, `linear_bias`
- training kernels: `layer_norm`, `layer_norm_stats`, `layer_norm_backward`
- row losses: `softmax`, `cross_entropy`, `cross_entropy_backward`
- optimizer utilities: `adamw_update`, `zero_grad`

## Optimization Shape

Flat elementwise kernels use `uint4`, where one packed load carries eight f16
values. If the element count is not divisible by eight, a scalar tail kernel
finishes the leftovers.

Row-wise kernels such as layer norm, softmax, and cross entropy cannot split the
row into separate fast/tail launches because the row statistics must see every
column. If the row width is divisible by eight, they use packed chunks; otherwise
they use scalar loads inside the same CUDA block to keep row starts safely
aligned. Either way, each row reduces once with CUB shared storage.

Layer norm, softmax, cross entropy, and layer norm backward accumulate row
statistics in float. Outputs are stored back as f16. This keeps the half I/O
bandwidth while avoiding the worst mean/variance and log-sum-exp errors.

## Training Contracts

Layer norm normalizes the last dimension. `cf_math_layer_norm_stats_f16` writes
row-wise `Mean` and `Var`; `cf_math_layer_norm_backward_f16` consumes those
cached tensors.

Softmax is last-dimension only. Cross entropy expects one f16 class index per
row and writes one scalar mean loss. Cross entropy backward computes:

```text
dLogits = (softmax(Logits) - one_hot(Targets)) / rows
```

AdamW updates `Weight`, `M`, and `V` in place with bias correction from the
one-based `step`. Zero grad is an async device memset over the gradient tensor.

## Benchmark App

`app/src/app.cu` initializes device-resident f16 tensors and times both the
basic math kernels and the training kernels. Large runs use `cols = 1024`; small
runs use `cols = element_count`, so a command like this exercises non-divisible
row tails:

```sh
./app/build/app 1031 10 2
```
