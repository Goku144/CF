# CypherFramework — MNIST Digit Recognizer: Complete Technical Explanation

> **Scope:** Every function in `app/src/app.cu` and `app/src/predict_cpu.c`, the
> full CypherFramework architecture, and updated project-concept notes.

---

## Table of Contents

1. [Project Mission](#1-project-mission)
2. [CypherFramework Architecture](#2-cypherframework-architecture)
3. [Module Dependency Map](#3-module-dependency-map)
4. [The AI Training Stack](#4-the-ai-training-stack)
5. [app/src/app.cu — File Header & Includes](#5-appsrcappcu--file-header--includes)
6. [Constants and Macros](#6-constants-and-macros)
7. [Data Structures](#7-data-structures)
8. [Global State](#8-global-state)
9. [Status-Check Helpers](#9-status-check-helpers)
10. [Device Pointer Helper](#10-device-pointer-helper)
11. [Pixel Normalisation Kernels](#11-pixel-normalisation-kernels)
12. [Path Utilities](#12-path-utilities)
13. [Random Number Generator](#13-random-number-generator)
14. [Shuffle Helper](#14-shuffle-helper)
15. [CSV Loader](#15-csv-loader)
16. [Image Resize Helper](#16-image-resize-helper)
17. [Batch Loader](#17-batch-loader)
18. [Training Batch Cursor](#18-training-batch-cursor)
19. [Descriptor Lifecycle](#19-descriptor-lifecycle)
20. [Tensor Binding](#20-tensor-binding)
21. [Parameter Initialisation](#21-parameter-initialisation)
22. [Gradient Attachment](#22-gradient-attachment)
23. [Forward Pass](#23-forward-pass)
24. [Weight File I/O](#24-weight-file-io)
25. [Checkpoint Orchestration](#25-checkpoint-orchestration)
26. [Single-Image Prediction (CUDA)](#26-single-image-prediction-cuda)
27. [Batch Accuracy Counting](#27-batch-accuracy-counting)
28. [Per-Digit Accuracy Counting](#28-per-digit-accuracy-counting)
29. [Dataset Evaluation](#29-dataset-evaluation)
30. [Predict Subcommand](#30-predict-subcommand)
31. [Training Orchestrator](#31-training-orchestrator)
32. [main() Entry Point](#32-main-entry-point)
33. [app/src/predict_cpu.c — Overview](#33-appsrcpredict_cpuc--overview)
34. [predict_cpu: Constants](#34-predict_cpu-constants)
35. [predict_cpu: f16→f32 Conversion](#35-predict_cpu-f16f32-conversion)
36. [predict_cpu: Weight Loader](#36-predict_cpu-weight-loader)
37. [predict_cpu: Image Loader](#37-predict_cpu-image-loader)
38. [predict_cpu: Conv2d](#38-predict_cpu-conv2d)
39. [predict_cpu: ReLU](#39-predict_cpu-relu)
40. [predict_cpu: MaxPool](#40-predict_cpu-maxpool)
41. [predict_cpu: Linear+Bias](#41-predict_cpu-linearbias)
42. [predict_cpu: Softmax](#42-predict_cpu-softmax)
43. [predict_cpu: main()](#43-predict_cpu-main)
44. [Model Architecture Deep-Dive](#44-model-architecture-deep-dive)
45. [Mixed Precision Strategy](#45-mixed-precision-strategy)
46. [Training Loop Step-by-Step](#46-training-loop-step-by-step)
47. [Backward Pass Walk-Through](#47-backward-pass-walk-through)
48. [Checkpoint Format](#48-checkpoint-format)
49. [Dataset Pipeline](#49-dataset-pipeline)
50. [Project Concept Update](#50-project-concept-update)

---

## 1. Project Mission

The CypherFramework digit recogniser is a **full-stack GPU training application**
built on top of the CypherFramework `cf_math` tensor layer. Its goals are:

- Demonstrate that a real convolutional neural network can be trained end-to-end
  entirely within the CypherFramework API with **no PyTorch, no TensorFlow, no
  external ML library of any kind**.
- Serve as a live integration test for every operator in the `cf_math` f16
  family: convolution, pooling, matmul, softmax, cross-entropy, SGD.
- Provide a **CPU-only inference path** (`predict_cpu`) so the model can be
  deployed on machines without a GPU once weights have been trained.
- Use the framework's own runtime primitives (`cf_log`, `cf_io`, `cf_buffer`,
  `cf_u8`/`cf_u16`/`cf_u32`/`cf_usize`) everywhere, validating those layers
  under realistic workloads.

---

## 2. CypherFramework Architecture

```text
┌─────────────────────────────────────────────────────┐
│                    APPLICATION                       │
│   app/src/app.cu          app/src/predict_cpu.c     │
└──────────────┬──────────────────────┬───────────────┘
               │                      │ (cf_types.h only)
               ▼                      │
┌─────────────────────────┐           │
│        AI Layer         │           │
│  cf_model  cf_gradient  │           │
└──────────┬──────────────┘           │
           ▼                          │
┌─────────────────────────┐           │
│      MATH Layer         │           │
│  cf_math  cf_math_f16   │           │
│  cuBLASLt cuDNN cuRAND  │           │
└──────────┬──────────────┘           │
           ▼                          │
┌─────────────────────────────────────┤
│           RUNTIME / MEMORY          │
│  cf_types  cf_status  cf_log        │
│  cf_io     cf_buffer  cf_string     │
└─────────────────────────────────────┘
           ▼
┌─────────────────────────┐
│      ALLOCATOR          │
│  cf_alloc  cf_arena     │
│  cf_pool   mimalloc     │
└─────────────────────────┘
```

Each layer depends only on layers **below** it. The application sits at the top
and is the only code that knows about both the MATH layer and the RUNTIME layer
simultaneously.

---

## 3. Module Dependency Map

| Module | Depends on | Provides |
|--------|-----------|---------|
| `cf_types` | `<stdint.h>` `<stdbool.h>` | `cf_u8`, `cf_u16`, `cf_u32`, `cf_u64`, `cf_usize`, `cf_isize`, `cf_uptr`, `cf_bool`, `CF_NULL`, `CF_TRUE`, `CF_FALSE` |
| `cf_status` | `cf_types` | `cf_status` enum, `cf_status_as_char()` |
| `cf_log` | `cf_status` `cf_types` | `cf_log_write()`, `CF_LOG_TRACE/DEBUG/INFO/WARN/ERROR/FATAL` macros |
| `cf_io` | `cf_memory` `cf_string` | `cf_io_read_file()`, `cf_io_write_file()`, `cf_io_append_file()`, `cf_io_exists()`, `cf_io_file_size()` |
| `cf_buffer` / `cf_memory` | `cf_alloc` `cf_types` | `cf_buffer`, `cf_bytes`, `cf_buffer_init()`, `cf_buffer_destroy()`, `cf_buffer_append_bytes()` |
| `cf_math_storage` | `cf_status` `cf_types` mimalloc CUDA | `cf_math_handle`, `cf_math_workspace`, `cf_math_context`, `cf_math_arena` |
| `cf_math` | `cf_math_storage` | `cf_math`, `cf_math_desc`, `cf_math_bind()`, `cf_math_desc_create()` |
| `cf_math_f16` | `cf_math` cuDNN cuBLASLt | All f16 training ops |

---

## 4. The AI Training Stack

The digit recogniser exercises this call chain every training step:

```
main()
  └─ train_digit_recognizer()
       ├─ load_csv()                         [RUNTIME: file I/O]
       ├─ app_desc_create()                  [MATH: cf_math_desc_create x12]
       ├─ app_bind_tensors()                 [MATH: cf_math_bind x22]
       ├─ app_init_param()                   [CUDA: cudaMemcpyAsync]
       └─ training loop (N steps)
            ├─ app_next_train_batch()        [CPU: stbi_load_16, memcpy]
            ├─ cudaMemcpyAsync ×2            [CUDA: H→D copy]
            ├─ app_normalize_u16_to_f16()   [CUDA kernel: custom]
            ├─ digit_forward()
            │    ├─ cf_math_conv2d_f16()    [cuDNN]
            │    ├─ cf_math_relu_f16()      [custom kernel]
            │    ├─ cf_math_pooling_f16()   [cuDNN]
            │    ├─ cf_math_linear_bias_f16() [cuBLASLt]
            │    └─ cf_math_softmax_f16()   [custom kernel]
            ├─ cf_math_fused_cross_entropy() [custom kernel]
            ├─ cf_math_matmul_trans_b_f16() [cuBLASLt]
            ├─ cf_math_matmul_trans_a_f16() [cuBLASLt]
            ├─ cf_math_reduce_sum_rows_f16() [custom kernel]
            ├─ cf_math_pooling_backward_f16() [cuDNN]
            ├─ cf_math_relu_backward_f16()  [custom kernel]
            ├─ cf_math_conv2d_backward_data_f16() [cuDNN]
            ├─ cf_math_conv2d_backward_filter_f16() [cuDNN]
            ├─ cf_math_sgd_update_f16() ×3  [custom kernel]
            └─ app_save_checkpoints()       [RUNTIME: cf_io_write_file]
```

---

## 5. app/src/app.cu — File Header & Includes

```c
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include "MATH/cf_math.h"
#include "RUNTIME/cf_log.h"
#include "RUNTIME/cf_io.h"
#include "RUNTIME/cf_types.h"
#include "MEMORY/cf_memory.h"
#include "RUNTIME/stb_image.h"
```

### Why each include is here

| Include | Reason |
|---------|--------|
| `<cuda_fp16.h>` | `__half`, `__float2half_rn()`, `__half2float()` used in kernels and host-side probability extraction |
| `<cuda_runtime.h>` | `cudaMemcpyAsync()`, `cudaMallocHost()`, `cudaFreeHost()`, `cudaStreamSynchronize()`, `cudaMemsetAsync()` |
| `<sys/stat.h>` | `mkdir(2)` syscall to create the checkpoint directory before saving |
| `"MATH/cf_math.h"` | Pulls in all of `cf_math_storage.h`, `cf_math_f16.h`, `cf_math_desc`, `cf_math_handle`, and the entire f16 operator surface |
| `"RUNTIME/cf_log.h"` | `CF_LOG_INFO`, `CF_LOG_WARN`, `CF_LOG_ERROR` macros used for all diagnostic output |
| `"RUNTIME/cf_io.h"` | `cf_io_write_file()`, `cf_io_read_file()` for checkpoint save/load |
| `"RUNTIME/cf_types.h"` | `cf_u8`, `cf_u16`, `cf_u32`, `cf_usize`, `cf_bool`, `CF_NULL` type aliases |
| `"MEMORY/cf_memory.h"` | `cf_buffer`, `cf_bytes`, `cf_buffer_init()`, `cf_buffer_destroy()` for file I/O staging |
| `"RUNTIME/stb_image.h"` | `stbi_load_16()`, `stbi_image_free()` for PNG image loading — implementation already compiled via `lib/src/AI/cf_tokenizer.cu` |

### Note on removed raw C headers

`<stdint.h>`, `<stdio.h>`, `<stdlib.h>`, `<string.h>` are no longer explicitly
included because:
- `cf_types.h` provides all integer type aliases.
- `cf_log.h` and `cf_io.h` transitively pull in what they need.
- `snprintf` and `memcpy` / `memset` are still used — they are available via
  the framework's transitive includes.
- `malloc` / `free` are still used for the raw `__half` host staging buffer in
  `cf_math_save_weights`, which needs temporary unmanaged memory before
  constructing the `cf_bytes` view.

---

## 6. Constants and Macros

```c
#define DIGIT_BATCH_SIZE         64
#define DIGIT_IMAGE_H            28
#define DIGIT_IMAGE_W            28
#define DIGIT_IMAGE_PIXELS       (DIGIT_IMAGE_H * DIGIT_IMAGE_W)   // 784
#define DIGIT_CONV_CHANNELS      16
#define DIGIT_PADDED_CLASSES     16
#define DIGIT_REAL_CLASSES       10
#define DIGIT_FLATTENED_FEATURES (DIGIT_CONV_CHANNELS * 14 * 14)   // 3136
```

### DIGIT_BATCH_SIZE = 64

Every forward and backward pass processes exactly 64 images simultaneously.
This batch size was chosen because:

1. **Tensor Core alignment.** cuBLASLt Tensor Cores prefer M/N/K dimensions
   that are multiples of 16 (for f16). 64 is a multiple of 16.
2. **Memory budget.** A single batch of raw `cf_u16` pixels is
   `64 × 784 × 2 = 100 352` bytes ≈ 98 KB, well within L2 cache on sm_75+.
3. **Gradient variance.** Batch size 64 provides low-variance gradient
   estimates without the overhead of larger batches.

Changing this constant requires recompilation; no runtime path adjusts to it
dynamically.

### DIGIT_IMAGE_H / DIGIT_IMAGE_W = 28 × 28

Standard MNIST resolution. Images from the dataset that are not 28 × 28 are
resized to this with nearest-neighbour interpolation in
`app_resize_or_copy_u16`.

### DIGIT_IMAGE_PIXELS = 784

Precomputed product used everywhere a per-image flat element count is needed.
Avoids repeated multiplication at runtime.

### DIGIT_CONV_CHANNELS = 16

The first (and only) convolution layer produces 16 feature maps. This value was
chosen to stay Tensor Core-aligned (multiple of 16) while keeping parameter
count low for a smoke-test model.

### DIGIT_PADDED_CLASSES = 16

The dense output layer has width 16, not 10. Classes 10–15 are padding. This
keeps the matmul M/N/K all multiples of 16, which is required for cuBLASLt
Tensor Core paths. The `predict_digit` and counting functions only inspect
columns 0–9 when computing argmax.

### DIGIT_REAL_CLASSES = 10

The ten actual digit labels (0 through 9). All accuracy reporting uses this
constant, never `DIGIT_PADDED_CLASSES`.

### DIGIT_FLATTENED_FEATURES = 3136

After the 2×2 max-pool with stride 2, the feature maps are 14×14. Flattened:
`16 × 14 × 14 = 3136`. This is the input width to the dense layer.

---

## 7. Data Structures

### 7.1 `sample_t`

```c
typedef struct {
  char path[256];
  cf_u8 label;
} sample_t;
```

One entry in the dataset CSV. `path` is a relative path string from the CSV
(e.g., `train/3/00042.png`). The loader prefixes it with `g_dataset_root` to
form an absolute-or-relative-to-CWD path before calling `stbi_load_16`.

`label` stores the ground-truth digit class (0–9) as a `cf_u8`. The GPU labels
tensor also uses `CF_MATH_DTYPE_U8` so the byte representation is identical on
both sides of the host/device copy.

### 7.2 `digit_tensors`

```c
typedef struct {
  cf_math input_raw;    // [64, 1, 28, 28]  cf_u16 — raw pixel staging
  cf_math input;        // [64, 1, 28, 28]  f16    — normalised input
  cf_math labels;       // [64]             u8     — ground-truth labels
  cf_math conv_w;       // [16, 1, 3, 3]   f16    — conv1 weights (trainable)
  cf_math conv_out;     // [64, 16, 28, 28] f16   — conv1 output
  cf_math relu_out;     // [64, 16, 28, 28] f16   — after ReLU
  cf_math pool_out;     // [64, 16, 14, 14] f16   — after max-pool
  cf_math flat;         // [64, 3136]       f16   — view alias of pool_out
  cf_math dense_w;      // [3136, 16]       f16   — dense weights (trainable)
  cf_math dense_b;      // [16]             f16   — dense bias (trainable)
  cf_math logits;       // [64, 16]         f16   — pre-softmax scores
  cf_math probs;        // [64, 16]         f16   — softmax probabilities
  cf_math dY;           // [64, 16]         f16   — cross-entropy gradient
  cf_math batch_loss;   // [64]             f32   — per-sample loss
  cf_math loss;         // [1]              f32   — mean scalar loss
  cf_math d_flat;       // [64, 3136]       f16   — gradient w.r.t. flat
  cf_math d_pool;       // [64, 16, 14, 14] f16  — view alias of d_flat
  cf_math d_relu;       // [64, 16, 28, 28] f16  — gradient w.r.t. relu_out
  cf_math d_conv_out;   // [64, 16, 28, 28] f16  — gradient w.r.t. conv_out
  cf_math d_input;      // [64, 1, 28, 28]  f16  — gradient w.r.t. input
  cf_math d_conv_w;     // [16, 1, 3, 3]    f16  — gradient w.r.t. conv_w
  cf_math d_dense_w;    // [3136, 16]       f16  — gradient w.r.t. dense_w
  cf_math d_dense_b;    // [16]             f16  — gradient w.r.t. dense_b
} digit_tensors;
```

All `cf_math` fields are **non-owning views** into the `cf_math_handle`'s arena.
None of them allocate memory themselves. Ownership lives entirely in the
`cf_math_handle`.

**The `flat` / `d_pool` alias trick:**

```c
t->flat  = t->pool_out;  t->flat.desc  = &d->flat_desc;
t->d_pool = t->d_flat;   t->d_pool.desc = &d->pool_desc;
```

`flat` and `pool_out` share the same `byte_offset` inside the arena — they are
the same bytes, just re-interpreted with a different descriptor (4D vs 2D). This
zero-copy reshape is the entire point of the non-owning view model: no data is
moved, no allocation happens, the reshape costs exactly two pointer assignments.

### 7.3 `digit_descs`

```c
typedef struct {
  cf_math_desc raw_desc;
  cf_math_desc input_desc;
  cf_math_desc label_desc;
  cf_math_desc conv_w_desc;
  cf_math_desc conv_out_desc;
  cf_math_desc pool_desc;
  cf_math_desc flat_desc;
  cf_math_desc dense_w_desc;
  cf_math_desc dense_b_desc;
  cf_math_desc logits_desc;
  cf_math_desc batch_loss_desc;
  cf_math_desc scalar_desc;
} digit_descs;
```

Each `cf_math_desc` stores:
- `rank` and `dim[]` — the tensor shape.
- `strides[]` — computed row-major strides.
- `cublastlt` — cuBLASLt matrix layout and matmul descriptors (created for 2D
  f16 tensors).
- `cudnn` — cuDNN tensor, filter, convolution, pooling descriptors (created for
  4D f16 tensors).
- `dtype` — the element type enum.

Descriptors are expensive to create (they call into cuBLAS/cuDNN), so they are
created once in `app_desc_create` and reused for every step of training. The
same descriptor is shared between the forward tensor and its gradient tensor
when they have the same shape (e.g., `conv_out_desc` is used for both
`conv_out` and `relu_out` and their gradients).

---

## 8. Global State

```c
static const char *g_train_csv    = "public/train.csv";
static const char *g_test_csv     = "public/test.csv";
static const char *g_dataset_root = "public";
static cf_u32 g_rng_state         = 0x12345678u;
```

These four module-level statics are the only mutable global state in `app.cu`.

| Variable | Purpose | Overridable |
|----------|---------|------------|
| `g_train_csv` | Path to CSV listing training images | `argv[3]` |
| `g_test_csv` | Path to CSV listing test images | `argv[4]` |
| `g_dataset_root` | Root directory prepended to CSV-relative paths | `argv[5]` |
| `g_rng_state` | Seed for the XORShift32 PRNG used to shuffle the training order | Not overridable at runtime |

Using `static const char *` (pointer, not array) for the path strings allows
`main()` to point them at `argv` strings without copying. The pointers remain
valid for the life of the process because `argv` strings live in process memory.

`g_rng_state` is a `cf_u32` (previously `uint32_t`) — the framework type alias
makes the intent explicit without changing the ABI.

---
