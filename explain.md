# Full Explanation of `app.cu` and `predict_cpu.c`

This document explains the digit recognizer in this project from the point of
view of a strong C programmer who does not yet know AI.

The short version:

```text
app/src/app.cu         trains the model on the GPU and can also predict on GPU
app/src/predict_cpu.c  loads saved weights and predicts on CPU only
```

Both files describe the same neural network idea:

```text
image
  -> normalize pixels
  -> convolution
  -> ReLU
  -> max pooling
  -> flatten
  -> dense/linear layer
  -> softmax probabilities
  -> pick digit 0..9
```

The biggest rule in this project is:

```text
The image size controls the pooled size.
The pooled size controls FLAT.
FLAT controls dense_weights size.
The saved dense_weights file only works with the exact FLAT size it was trained with.
```

So if you change image size, you are changing the model shape. That usually
means you must retrain and save new checkpoint files.

---

## 1. Mental Model: This Is Just Arrays

An AI model can look scary because of words like "tensor", "convolution",
"logits", and "gradient". In C terms, most of it is just arrays with agreed
shapes.

For example, a grayscale image of `28 x 28` pixels is:

```c
float image[28 * 28];
```

The pixel at row `y`, column `x` is:

```c
image[y * 28 + x]
```

A batch of 64 grayscale images is:

```c
float batch[64 * 1 * 28 * 28];
```

The `1` means "one channel", because grayscale has one channel. RGB would have
three channels.

The project uses the common AI layout:

```text
[N, C, H, W]
```

Where:

```text
N = batch count
C = channel count
H = height
W = width
```

For one batch of 64 grayscale 28x28 images:

```text
[64, 1, 28, 28]
```

In memory, this is still one flat array. The shape just tells functions how to
interpret the flat bytes.

---

## 2. What the Model Does

The model is a small digit classifier. It tries to read an image and output a
digit from `0` to `9`.

The architecture is:

```text
input image
  [1, H, W]

conv 3x3, 16 output channels, padding 1
  [16, H, W]

ReLU
  [16, H, W]

max pool 2x2 stride 2
  [16, H / 2, W / 2]

flatten
  [16 * (H / 2) * (W / 2)]

dense layer
  [16]

softmax
  [16]

argmax over classes 0..9
```

Only classes `0..9` are real digits. Classes `10..15` are padding. The output
is 16 values because 16 is convenient for GPU math alignment.

---

## 3. The Important Shape Formula

The current CPU predictor has:

```c
#define IMAGE_H    800
#define IMAGE_W    800
#define IMAGE_PIX  (IMAGE_H * IMAGE_W)
#define CONV_CH    16
#define POOL_H     14
#define POOL_W     14
#define FLAT       (CONV_CH * POOL_H * POOL_W)
```

But the `POOL_H` and `POOL_W` values are only correct when:

```text
IMAGE_H = 28
IMAGE_W = 28
```

Because max pooling uses `2x2` with stride `2`, it cuts height and width in
half:

```text
POOL_H = IMAGE_H / 2
POOL_W = IMAGE_W / 2
```

So the safer version is:

```c
#define IMAGE_H    800
#define IMAGE_W    800
#define IMAGE_PIX  (IMAGE_H * IMAGE_W)
#define CONV_CH    16
#define POOL_H     (IMAGE_H / 2)
#define POOL_W     (IMAGE_W / 2)
#define FLAT       (CONV_CH * POOL_H * POOL_W)
```

For `28x28`:

```text
POOL_H = 14
POOL_W = 14
FLAT   = 16 * 14 * 14 = 3136
```

For `800x800`:

```text
POOL_H = 400
POOL_W = 400
FLAT   = 16 * 400 * 400 = 2,560,000
```

Then dense weights are:

```text
dense_weights shape = [FLAT, 16]
```

For `28x28`:

```text
3136 * 16 = 50,176 half-floats
```

For `800x800`:

```text
2,560,000 * 16 = 40,960,000 half-floats
```

That is why old checkpoint files stop working when image size changes.

---

## 4. What `app.cu` Is For

`app/src/app.cu` is the CUDA trainer and GPU predictor.

It does three big jobs:

```text
1. Load dataset images and labels.
2. Train the neural network on GPU.
3. Save checkpoint files containing trained weights.
```

It also has a `predict` mode that loads checkpoint files and runs prediction on
GPU.

Training means:

```text
start with random weights
repeat many times:
  load images
  run forward pass
  measure error
  compute gradients
  update weights
save weights to .bin files
```

The saved `.bin` files are later used by `predict_cpu.c`.

---

## 5. What `predict_cpu.c` Is For

`app/src/predict_cpu.c` is a standalone CPU-only predictor.

It does not train. It only:

```text
1. Loads one image.
2. Resizes/normalizes it.
3. Loads saved checkpoint weights.
4. Runs the same forward pass using normal C loops.
5. Prints digit probabilities.
```

This file exists so you can run prediction without CUDA.

It does not link the whole framework. It only includes:

```c
#include "RUNTIME/stb_image.h"
#include "RUNTIME/cf_types.h"
```

So `predict_cpu.c` is intentionally self-contained.

---

## 6. AI Words Translated to C Words

### Tensor

A tensor is just a typed array plus a shape.

```text
AI word: tensor
C idea:  flat memory + dimensions + strides
```

Example:

```text
shape [16, 28, 28]
means 16 planes, each plane has 28*28 floats
```

### Weight

A weight is a number the model learns.

In C terms, weights are just arrays of floats or half-floats. Training changes
these arrays until the output becomes useful.

### Bias

A bias is an extra learned value added after multiplication.

Dense layer:

```c
output[j] = bias[j] + sum(input[i] * weight[i][j])
```

### Forward Pass

Forward pass means computing prediction from input to output.

```text
image -> layers -> probabilities
```

### Loss

Loss is a number that says how wrong the model was.

Small loss is good. Big loss is bad.

### Backward Pass

Backward pass computes how each weight should change to reduce loss.

The result is called a gradient.

### Gradient

A gradient is a "change direction" for a weight.

The trainer updates weights like:

```c
weight = weight - learning_rate * gradient;
```

### Checkpoint

A checkpoint is just saved model arrays.

In this project:

```text
layer1_weights_stepN.bin
dense_weights_stepN.bin
dense_bias_stepN.bin
```

They are raw binary half-float data with no header.

---

## 7. The Model Layers

### 7.1 Input

Input is a grayscale image.

The raw image pixels are loaded as 16-bit grayscale values:

```text
0..65535
```

Then they are normalized to:

```text
0.0..1.0
```

Formula:

```c
normalized = pixel * (1.0f / 65535.0f);
```

### 7.2 Convolution

The convolution layer uses 16 filters. Each filter is `3x3`.

Because the input has one channel, each filter has:

```text
1 * 3 * 3 = 9 weights
```

There are 16 filters:

```text
16 * 9 = 144 weights
```

Each filter scans over the image and creates one output image. So 16 filters
create 16 output images:

```text
input:  [1, H, W]
output: [16, H, W]
```

Padding is 1, so the output stays the same height and width.

### 7.3 ReLU

ReLU means:

```c
if(x < 0) x = 0;
```

It keeps positive values and kills negative values.

Why use it? It adds non-linearity. Without non-linear steps, multiple layers
would collapse into one big linear equation.

### 7.4 Max Pooling

Max pooling uses a `2x2` window and keeps only the largest value:

```text
a b
c d

output = max(a, b, c, d)
```

With stride 2, it jumps two pixels at a time. So width and height are divided
by 2.

```text
[16, H, W] -> [16, H/2, W/2]
```

This is exactly why `POOL_H` and `POOL_W` must change when `IMAGE_H` and
`IMAGE_W` change.

### 7.5 Flatten

Flatten does not compute anything interesting. It just treats a multi-dimensional
array as a single vector.

Before:

```text
[16, 14, 14]
```

After:

```text
[3136]
```

Because:

```text
16 * 14 * 14 = 3136
```

In `app.cu`, the flatten tensor is actually a view over the same memory as
`pool_out`.

### 7.6 Dense / Linear Layer

The dense layer is matrix multiplication plus bias.

For one image:

```text
flat:      [FLAT]
weights:   [FLAT, 16]
bias:      [16]
logits:    [16]
```

Formula:

```c
for(int j = 0; j < 16; ++j) {
  logits[j] = bias[j];
  for(int i = 0; i < FLAT; ++i) {
    logits[j] += flat[i] * weight[i * 16 + j];
  }
}
```

This is exactly what `predict_cpu.c` does in `linear_bias`.

### 7.7 Logits

Logits are raw scores before converting to probabilities.

Example:

```text
digit 0 score =  1.2
digit 1 score = -0.7
digit 2 score =  3.9
...
```

The biggest logit usually becomes the predicted class, but softmax converts
them into percentages.

### 7.8 Softmax

Softmax converts scores into probabilities.

The output sums to 1:

```text
p0 + p1 + ... + p15 = 1
```

In prediction, this project ignores padded classes `10..15` and displays only
digits `0..9`.

---

## 8. `predict_cpu.c` Detailed Walkthrough

This file is pure C prediction. It does not train and it does not use CUDA.

### 8.1 File Header Comment

The big comment at the top describes the expected model:

```text
input  [1, 28, 28]
conv   [16, 1, 3x3]
pool   [16, 14, 14]
flat   [3136]
linear [3136, 16]
```

If the macros below no longer use 28x28, this comment should be updated.

### 8.2 Includes

```c
#define STB_IMAGE_IMPLEMENTATION
#include "RUNTIME/stb_image.h"
```

This pulls in `stb_image`, a single-header image loader. The
`STB_IMAGE_IMPLEMENTATION` define tells the header to include its function
definitions in this C file.

```c
#include "RUNTIME/cf_types.h"
```

This gives aliases like:

```text
cf_u16
cf_u32
cf_usize
cf_bool
CF_NULL
CF_TRUE
CF_FALSE
```

### 8.3 Constants

```c
#define IMAGE_H    800
#define IMAGE_W    800
#define IMAGE_PIX  (IMAGE_H * IMAGE_W)
#define CONV_CH    16
#define POOL_H     14
#define POOL_W     14
#define FLAT       (CONV_CH * POOL_H * POOL_W)
#define PAD_CLS    16
#define REAL_CLS   10
```

Meaning:

```text
IMAGE_H / IMAGE_W = input image size after resize
IMAGE_PIX         = total pixels
CONV_CH           = number of convolution output channels
POOL_H / POOL_W   = image size after max pool
FLAT              = flattened feature count
PAD_CLS           = output width, padded to 16
REAL_CLS          = real digit classes, 0..9
```

Important correction:

```c
#define POOL_H (IMAGE_H / 2)
#define POOL_W (IMAGE_W / 2)
```

is better than hardcoding `14`, unless your input is always `28x28`.

### 8.4 `f16_to_f32`

```c
static float f16_to_f32(cf_u16 h)
```

Checkpoint files store weights as IEEE 754 half precision. That is 16-bit
floating point.

Normal C `float` is 32-bit. The CPU code wants to compute in `float`, so it
must convert every saved half-float into a normal float.

This function manually decodes:

```text
sign
exponent
mantissa
```

from the 16-bit half and rebuilds the equivalent 32-bit float bit pattern.

In C terms:

```text
input:  raw 16-bit floating point bits
output: normal float
```

It handles:

```text
zero
subnormal numbers
normal numbers
infinity / NaN
```

The final `memcpy` is used to move raw bits into a `float` without strict aliasing
problems:

```c
float result;
memcpy(&result, &f, sizeof(f));
return result;
```

### 8.5 `load_f16_weights`

```c
static float *load_f16_weights(const char *path, cf_usize n_elems)
```

This function loads one checkpoint file.

Steps:

```text
1. Open file with fopen.
2. Allocate raw cf_u16 buffer.
3. Read exactly n_elems * sizeof(cf_u16) bytes.
4. Allocate float output buffer.
5. Convert each half-float to float.
6. Return float pointer.
```

The caller must pass the expected element count.

For conv weights:

```c
CONV_CH * 9
```

For dense weights:

```c
FLAT * PAD_CLS
```

For dense bias:

```c
PAD_CLS
```

If `FLAT` is wrong, this function expects the wrong file size. That is one of
the ways image size changes break old checkpoints.

### 8.6 `load_image`

```c
static cf_bool load_image(const char *path, float *out_pixels)
```

This loads the input image for prediction.

```c
cf_u16 *raw = stbi_load_16(path, &w, &h, &c, 1);
```

The final `1` means:

```text
force image to one channel grayscale
```

Then it resizes with nearest neighbor:

```c
for(int y = 0; y < IMAGE_H; ++y) {
  int sy = (y * h) / IMAGE_H;
  for(int x = 0; x < IMAGE_W; ++x) {
    int sx = (x * w) / IMAGE_W;
    out_pixels[y * IMAGE_W + x] =
      (float)raw[sy * w + sx] * (1.0f / 65535.0f);
  }
}
```

This maps each output pixel `(x, y)` back to a source pixel `(sx, sy)`.

It also normalizes from `0..65535` to `0.0..1.0`.

The comment says "resize to 28x28", but if `IMAGE_H` and `IMAGE_W` are 800,
the function actually resizes to 800x800. The comment should be updated.

### 8.7 `conv2d_3x3_pad1`

```c
static void conv2d_3x3_pad1(
  const float *input,
  const float *weight,
  float *output,
  int out_ch,
  int H,
  int W)
```

This is the convolution layer in plain C.

Input:

```text
input  [H, W]
weight [out_ch, 3, 3]
output [out_ch, H, W]
```

For every output channel, for every pixel, it applies one 3x3 filter:

```c
acc += krow[ky * 3 + kx] * input[sy * W + sx];
```

Padding is handled by skipping coordinates outside the image:

```c
if(sy < 0 || sy >= H) continue;
if(sx < 0 || sx >= W) continue;
```

Because padding is 1 and kernel is 3x3, the output has the same `H` and `W`.

### 8.8 `relu_inplace`

```c
static void relu_inplace(float *buf, int n)
```

This loops over a flat array and replaces negative values with zero:

```c
for(int i = 0; i < n; ++i)
  if(buf[i] < 0.0f) buf[i] = 0.0f;
```

It is "inplace" because it modifies the same buffer.

### 8.9 `max_pool_2x2`

```c
static void max_pool_2x2(
  const float *input,
  float *output,
  int ch,
  int in_H,
  int in_W)
```

This reduces each channel from:

```text
[in_H, in_W]
```

to:

```text
[in_H / 2, in_W / 2]
```

It calculates:

```c
int out_H = in_H / 2;
int out_W = in_W / 2;
```

Then for each output pixel it reads a 2x2 area from the input:

```c
float m = iplane[sy * in_W + sx];
if(iplane[sy * in_W + sx + 1] > m) m = ...
if(iplane[(sy+1) * in_W + sx] > m) m = ...
if(iplane[(sy+1) * in_W + sx + 1] > m) m = ...
```

That is why the destination buffer must have:

```text
ch * (in_H / 2) * (in_W / 2)
```

elements.

This function assumes even input dimensions. If `IMAGE_H` or `IMAGE_W` is odd,
the last row/column is ignored because integer division truncates.

### 8.10 `linear_bias`

```c
static void linear_bias(
  const float *flat,
  const float *weight,
  const float *bias,
  float *logits,
  int in_features,
  int out_features)
```

This is dense layer matrix multiplication for one sample.

It computes:

```text
logits = flat * weight + bias
```

Weight layout is:

```text
[in_features][out_features]
```

So element `(i, j)` is:

```c
weight[i * out_features + j]
```

This function produces 16 raw output scores.

### 8.11 `softmax_inplace`

```c
static void softmax_inplace(float *v, int n)
```

This converts raw scores to probabilities.

It first finds the max:

```c
float mx = v[0];
```

Then it uses:

```c
expf(v[i] - mx)
```

Subtracting the max is a standard numeric stability trick. It prevents large
positive scores from overflowing `expf`.

Then it divides everything by the sum:

```c
v[i] *= 1.0f / sum;
```

After this:

```text
v[0] + v[1] + ... + v[n-1] = 1
```

### 8.12 `main` in `predict_cpu.c`

The CPU predictor expects:

```text
predict_cpu <image> <conv_weights.bin> <dense_weights.bin> <dense_bias.bin>
```

It does:

```text
1. Parse paths.
2. Load conv weights.
3. Load dense weights.
4. Load dense bias.
5. Allocate buffers.
6. Load image into input buffer.
7. Run forward pass.
8. Pick best real digit.
9. Print probabilities.
10. Free memory.
```

Important allocations:

```c
float *input    = calloc(IMAGE_PIX, sizeof(float));
float *conv_out = calloc(CONV_CH * IMAGE_PIX, sizeof(float));
float *pool_out = calloc(FLAT, sizeof(float));
float  logits[PAD_CLS];
```

Forward pass:

```c
conv2d_3x3_pad1(input, conv_w, conv_out, CONV_CH, IMAGE_H, IMAGE_W);
relu_inplace(conv_out, CONV_CH * IMAGE_PIX);
max_pool_2x2(conv_out, pool_out, CONV_CH, IMAGE_H, IMAGE_W);
linear_bias(pool_out, dense_w, dense_b, logits, FLAT, PAD_CLS);
softmax_inplace(logits, PAD_CLS);
```

Then it only searches classes `0..9`:

```c
for(int i = 1; i < REAL_CLS; ++i)
```

It ignores padded classes `10..15` for prediction.

---

## 9. `app.cu` Detailed Walkthrough

`app.cu` is larger because it trains the model. Training needs more objects:

```text
inputs
outputs
weights
biases
gradients
descriptors
CUDA memory
workspace memory
dataset loading
checkpoint saving
```

---

## 10. Important Constants in `app.cu`

```c
#define DIGIT_BATCH_SIZE 64
#define DIGIT_IMAGE_H 28
#define DIGIT_IMAGE_W 28
#define DIGIT_IMAGE_PIXELS (DIGIT_IMAGE_H * DIGIT_IMAGE_W)
#define DIGIT_CONV_CHANNELS 16
#define DIGIT_PADDED_CLASSES 16
#define DIGIT_REAL_CLASSES 10
#define DIGIT_FLATTENED_FEATURES (DIGIT_CONV_CHANNELS * 14 * 14)
```

Meaning:

```text
DIGIT_BATCH_SIZE        train 64 images at a time
DIGIT_IMAGE_H/W         resized image size
DIGIT_IMAGE_PIXELS      pixels per image
DIGIT_CONV_CHANNELS     conv output channels
DIGIT_PADDED_CLASSES    output width, 16
DIGIT_REAL_CLASSES      actual digits, 10
DIGIT_FLATTENED_FEATURES number of values after pool+flatten
```

Again, this line:

```c
#define DIGIT_FLATTENED_FEATURES (DIGIT_CONV_CHANNELS * 14 * 14)
```

only matches `28x28` input. Safer:

```c
#define DIGIT_POOL_H (DIGIT_IMAGE_H / 2)
#define DIGIT_POOL_W (DIGIT_IMAGE_W / 2)
#define DIGIT_FLATTENED_FEATURES \
  (DIGIT_CONV_CHANNELS * DIGIT_POOL_H * DIGIT_POOL_W)
```

Then use `DIGIT_POOL_H` and `DIGIT_POOL_W` in the pool descriptor too.

---

## 11. Structures in `app.cu`

### 11.1 `sample_t`

```c
typedef struct {
  char path[256];
  cf_u8 label;
} sample_t;
```

This represents one dataset row.

Example CSV row:

```text
train/7/image.png,7
```

Stored as:

```text
path  = "train/7/image.png"
label = 7
```

### 11.2 `digit_tensors`

```c
typedef struct {
  cf_math input_raw;
  cf_math input;
  cf_math labels;
  cf_math conv_w;
  cf_math conv_out;
  cf_math relu_out;
  cf_math pool_out;
  cf_math flat;
  cf_math dense_w;
  cf_math dense_b;
  cf_math logits;
  cf_math probs;
  cf_math dY;
  cf_math batch_loss;
  cf_math loss;
  cf_math d_flat;
  cf_math d_pool;
  cf_math d_relu;
  cf_math d_conv_out;
  cf_math d_input;
  cf_math d_conv_w;
  cf_math d_dense_w;
  cf_math d_dense_b;
} digit_tensors;
```

This is a collection of all tensor handles used by the model.

`cf_math` does not itself contain the array data. It contains:

```text
byte_offset  where the data begins inside a big storage arena
elem_len     element count
desc         pointer to shape/type descriptor
grad_fn      optional gradient metadata
```

Forward tensors:

```text
input_raw   raw 16-bit image pixels staged on GPU
input       normalized f16 image pixels
labels      correct digit labels
conv_w      convolution weights
conv_out    convolution output
relu_out    ReLU output
pool_out    max pooling output
flat        flattened view of pool_out
dense_w     dense layer weights
dense_b     dense layer bias
logits      raw class scores
probs       softmax probabilities
batch_loss  per-image loss
loss        total/average loss
```

Backward/gradient tensors:

```text
dY          gradient at softmax/logits output
d_flat      gradient flowing back into flat vector
d_pool      same memory view idea as d_flat, but pool shape
d_relu      gradient through pooling into ReLU output
d_conv_out  gradient through ReLU into conv output
d_input     gradient through conv into input
d_conv_w    gradient for conv weights
d_dense_w   gradient for dense weights
d_dense_b   gradient for dense bias
```

Prefix `d_` means derivative/gradient. It answers:

```text
How should this value change to reduce loss?
```

### 11.3 `digit_descs`

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

This stores tensor descriptions: shape, strides, dtype, and backend descriptors.

A `cf_math_desc` contains:

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
```

For example:

```text
rank = 4
dim = [64, 1, 28, 28]
strides = [784, 784, 28, 1]
dtype = CF_MATH_DTYPE_F16
```

Strides tell how far to move in the flat array when an index changes.

For `[N, C, H, W] = [64, 1, 28, 28]`:

```text
index = n*784 + c*784 + y*28 + x
```

The descriptor also owns cuDNN/cuBLASLt descriptors so CUDA libraries know the
shape too.

---

## 12. Framework Structures Used by `app.cu`

These are declared in `public/inc/MATH/cf_math.h` and
`public/inc/MATH/cf_math_storage.h`.

### 12.1 `cf_math`

```c
struct cf_math
{
  cf_usize byte_offset;
  cf_usize elem_len;
  cf_math_desc *desc;
  cf_math_grad_node *grad_fn;
};
```

Think of this like a typed view into a big arena.

It is not:

```c
float *data;
```

Instead, the real pointer is computed like:

```c
void *ptr = handle->storage.backend + math->byte_offset;
```

That is exactly what `app_device_ptr` does.

### 12.2 `cf_math_desc`

Describes shape and dtype.

Important fields:

```text
rank     number of dimensions
dim      dimensions
strides  row-major strides
dtype    f16, f32, u8, etc.
```

It also has CUDA library descriptors:

```text
cublasLt layout/op/preference/algo
cudnn tensor/filter/conv/activation/pooling/reduce/opTensor
```

### 12.3 `cf_math_workspace`

```c
struct cf_math_workspace
{
  void *scratchpad;
  cudaStream_t stream;
  cf_usize scratchpad_size;
  cf_math_device device;
};
```

Workspace is temporary memory for CUDA library calls.

`stream` is the CUDA stream used for async operations.

### 12.4 `cf_math_context`

This owns CUDA library handles:

```text
cublas
cublasLt
cudnn
cusparse
cusolverDn
curand
```

It is similar to a big runtime context object.

### 12.5 `cf_math_handle`

```c
struct cf_math_handle
{
  cf_math_arena storage;
  cf_math_context *ctx;
  cf_math_workspace *workspace;
  cf_math_device device;
};
```

The handle connects:

```text
storage arena + CUDA context + workspace
```

Most math functions receive this handle because they need to find:

```text
where tensor data is
which CUDA stream to use
which cuDNN/cuBLAS handle to use
where scratchpad memory is
```

---

## 13. Global Variables in `app.cu`

```c
static const char *g_train_csv = "public/train.csv";
static const char *g_test_csv = "public/test.csv";
static const char *g_dataset_root = "public";
static cf_u32 g_rng_state = 0x12345678u;
```

These store default dataset paths and RNG state.

`g_rng_state` is used by a small xorshift random generator for shuffling
training samples.

---

## 14. Helper Functions in `app.cu`

### 14.1 `app_check_cf_status`

```c
static int app_check_cf_status(const char *step, cf_status state)
```

This checks framework return codes.

If state is `CF_OK`, return 0.

If not, log an error and return 1.

The code uses it like:

```c
if(app_check_cf_status("step name", some_call())) goto cleanup;
```

So return `1` means failure.

### 14.2 `app_check_cuda_status`

```c
static int app_check_cuda_status(const char *step, cudaError_t state)
```

Same idea, but for CUDA runtime errors.

### 14.3 `app_device_ptr`

```c
static void *app_device_ptr(const cf_math_handle *handle, const cf_math *math)
{
  return (void *)((cf_uptr)handle->storage.backend + (cf_uptr)math->byte_offset);
}
```

This computes the actual pointer for a tensor.

The framework stores one big memory block:

```text
handle->storage.backend
```

Each tensor stores an offset into that block:

```text
math->byte_offset
```

So pointer is:

```text
base + offset
```

Very C. Very important.

### 14.4 `app_normalize_u16_to_f16_kernel`

```c
__global__ static void app_normalize_u16_to_f16_kernel(uint4 *out, const uint4 *in, int chunks)
```

This is a CUDA kernel.

It converts raw `uint16_t` pixels to `__half` floats.

It processes 8 half-sized values at a time using `uint4`.

Why 8?

```text
uint4 = 4 * 32 bits = 128 bits
8 * uint16_t = 128 bits
8 * __half   = 128 bits
```

So each thread loads a 128-bit chunk and converts 8 pixels.

### 14.5 `app_normalize_u16_to_f16_tail_kernel`

Handles leftover pixels when the total count is not divisible by 8.

### 14.6 `app_normalize_u16_to_f16`

```c
static void app_normalize_u16_to_f16(cf_math_handle *handle, cf_math *dst, cf_math *src)
```

Host-side wrapper that launches the two kernels above.

It calculates:

```c
int n = (int)dst->elem_len;
int chunks = n / 8;
int tail_start = chunks * 8;
```

Then launches:

```text
vectorized kernel for chunks
tail kernel for leftovers
```

### 14.7 `app_join_path`

```c
static int app_join_path(char *dst, size_t dst_size, const char *root, const char *rel)
```

Builds:

```text
root/relative_path
```

using `snprintf`, and checks whether it fit.

### 14.8 `app_rand_u32`

Simple xorshift random generator.

Used for dataset shuffling.

### 14.9 `app_shuffle_indices`

```c
static void app_shuffle_indices(int *order, int count)
```

Implements Fisher-Yates shuffle:

```text
for i from count-1 down to 1:
  choose random j in [0, i]
  swap order[i], order[j]
```

This makes training batches see samples in different order.

---

## 15. Dataset Loading Functions

### 15.1 `load_csv`

```c
int load_csv(const char *csv_path, sample_t **out_samples, int *out_count)
```

Reads the dataset CSV.

Expected format:

```text
path,label
train/0/img.png,0
train/1/img.png,1
```

It:

```text
1. Opens file.
2. Skips header line.
3. Parses each row with sscanf.
4. Validates label is 0..9.
5. Grows sample array with realloc.
6. Stores path and label.
7. Returns samples and count.
```

`out_samples` receives a heap allocation. Caller must `free`.

### 15.2 `app_resize_or_copy_u16`

```c
static void app_resize_or_copy_u16(
  uint16_t *dst,
  const uint16_t *src,
  int width,
  int height)
```

If source already matches model image size, it copies:

```c
memcpy(dst, src, DIGIT_IMAGE_PIXELS * sizeof(uint16_t));
```

Otherwise it nearest-neighbor resizes:

```c
for(int y = 0; y < DIGIT_IMAGE_H; ++y) {
  int sy = (y * height) / DIGIT_IMAGE_H;
  for(int x = 0; x < DIGIT_IMAGE_W; ++x) {
    int sx = (x * width) / DIGIT_IMAGE_W;
    dst[y * DIGIT_IMAGE_W + x] = src[sy * width + sx];
  }
}
```

This is the GPU trainer version of the CPU predictor's `load_image` resize
logic, except this keeps `uint16_t` pixels. Normalization happens later on GPU.

### 15.3 `app_load_batch`

```c
static void app_load_batch(
  sample_t *dataset,
  int dataset_size,
  const int *order,
  int start,
  uint16_t *images_cpu,
  uint8_t *labels_cpu,
  int batch_size)
```

Fills one CPU batch.

For each batch slot:

```text
1. Pick dataset index.
2. Copy label to labels_cpu.
3. Zero destination image.
4. Build full image path.
5. Load grayscale 16-bit image using stb_image.
6. Resize/copy into images_cpu.
7. Free stb image memory.
```

The output buffers are:

```text
images_cpu: batch_size * DIGIT_IMAGE_PIXELS uint16_t values
labels_cpu: batch_size uint8_t values
```

### 15.4 `app_next_train_batch`

```c
static void app_next_train_batch(...)
```

Gets the next training batch. If the cursor would run past the dataset end, it
reshuffles the order and starts again.

---

## 16. Descriptor Creation and Tensor Binding

### 16.1 `app_desc_create`

```c
static int app_desc_create(digit_descs *d)
```

This creates all tensor descriptors.

Important shape definitions:

```c
int raw_dim[4]      = {64, 1, H, W};
int label_dim[1]    = {64};
int conv_w_dim[4]   = {16, 1, 3, 3};
int conv_out_dim[4] = {64, 16, H, W};
int pool_dim[4]     = {64, 16, 14, 14};
int flat_dim[2]     = {64, FLAT};
int dense_w_dim[2]  = {FLAT, 16};
int dense_b_dim[1]  = {16};
int logits_dim[2]   = {64, 16};
```

If image size changes, `pool_dim` and `FLAT` must change.

Each call:

```c
cf_math_desc_create(&desc, rank, dims, dtype)
```

does:

```text
1. Store rank.
2. Store dimensions.
3. Compute strides.
4. Create cuBLASLt descriptors if useful.
5. Create cuDNN descriptors if useful.
```

### 16.2 `app_desc_destroy`

Destroys every descriptor in `digit_descs`.

This matters because descriptors own CUDA/cuDNN/cuBLASLt objects.

### 16.3 `app_bind_tensors`

```c
static int app_bind_tensors(cf_math_handle *handle, digit_tensors *t, digit_descs *d)
```

This binds every `cf_math` tensor to a descriptor and reserves storage for it.

Example:

```c
cf_math_bind(handle, &t->input_raw, &d->raw_desc)
```

The bind operation:

```text
1. Clears the cf_math object.
2. Sets math->desc.
3. Reserves bytes in handle storage.
4. Sets byte_offset and elem_len.
```

Important special case:

```c
t->flat = t->pool_out;
t->flat.desc = &d->flat_desc;
```

This means `flat` uses the same memory as `pool_out`, but with a different
shape. No copy happens. It is a reshape/view.

Same idea:

```c
t->d_pool = t->d_flat;
t->d_pool.desc = &d->pool_desc;
```

Gradient memory is also viewed with different shapes.

---

## 17. Parameter Initialization

### 17.1 `app_init_param`

```c
static int app_init_param(cf_math_handle *handle, cf_math *param, float scale)
```

Allocates host half-float array, fills it with small deterministic values, then
copies it to GPU.

The values are based on:

```c
int centered = (int)(i % 17) - 8;
host[i] = __float2half_rn((float)centered * scale);
```

This gives small values around zero.

Why not all zeros? If all weights start equal, many neurons learn the same thing.
Small differences help the model learn different filters.

### 17.2 `app_init_zero_param`

Sets a parameter tensor to zero on GPU.

Used for bias:

```c
app_init_zero_param(handle, &tensors.dense_b)
```

Bias can safely start at zero.

### 17.3 `app_attach_grads`

```c
static void app_attach_grads(...)
```

Connects trainable parameters to their gradient tensors:

```text
conv_w  -> d_conv_w
dense_w -> d_dense_w
dense_b -> d_dense_b
```

This project mostly performs manual backward calls, but this metadata still
records which gradient belongs to which parameter.

---

## 18. Forward Pass in `app.cu`

### 18.1 `digit_forward`

```c
static void digit_forward(cf_math_handle *handle, digit_tensors *t)
{
  cf_math_conv2d_f16(handle, &t->conv_out, &t->input, &t->conv_w,
                     1, 1, 1, 1, 1, 1);
  cf_math_relu_f16(handle, &t->relu_out, &t->conv_out);
  cf_math_pooling_f16(handle, &t->pool_out, &t->relu_out,
                      CF_MATH_POOLING_MAX, 2, 2, 0, 0, 2, 2);
  cf_math_linear_bias_f16(handle, &t->logits, &t->flat,
                          &t->dense_w, &t->dense_b);
  cf_math_softmax_f16(handle, &t->probs, &t->logits, 1);
}
```

This is the GPU equivalent of the CPU predictor's forward pass.

Layer by layer:

```text
conv2d:
  input [64,1,H,W]
  weight [16,1,3,3]
  output [64,16,H,W]

relu:
  output [64,16,H,W]

pool:
  output [64,16,H/2,W/2]

linear:
  flat [64,FLAT]
  dense_w [FLAT,16]
  logits [64,16]

softmax:
  probs [64,16]
```

The softmax dimension is `1`, meaning softmax across the class dimension in
`[batch, class]`.

---

## 19. Checkpoint Functions

### 19.1 `cf_math_save_weights`

```c
static int cf_math_save_weights(cf_math_handle *handle, const cf_math *weights, const char *path)
```

Saves one tensor to a raw binary file.

Steps:

```text
1. Synchronize CUDA stream.
2. Allocate host memory.
3. Copy tensor from GPU to host.
4. Wrap host memory in cf_bytes.
5. Write file using cf_io_write_file.
6. Free host memory.
```

The file has no metadata. It is just raw half-float bytes.

That means the loader must already know the correct shape.

### 19.2 `app_save_checkpoints`

Builds these paths:

```text
layer1_weights_step<step>.bin
dense_weights_step<step>.bin
dense_bias_step<step>.bin
```

Then calls `cf_math_save_weights` for each trainable parameter.

### 19.3 `app_load_weights_from_file`

```c
static int app_load_weights_from_file(cf_math_handle *handle, cf_math *weights, const char *path)
```

Loads one raw checkpoint file into a GPU tensor.

It expects:

```c
bytes = weights->elem_len * sizeof(__half);
```

If the file size does not match exactly:

```c
if(buf.len != bytes) error
```

So if you change image size, old dense weights will fail here because their byte
count no longer matches.

---

## 20. Prediction Functions in `app.cu`

### 20.1 `predict_digit`

```c
int predict_digit(cf_math_handle *handle, digit_tensors *t, float *confidence)
```

Runs `digit_forward`, copies the first probability row to CPU, then chooses the
best digit from `0..9`.

This function expects the input tensor to already contain a prepared image.

### 20.2 `app_run_predict_mode`

This is the full GPU prediction command.

Usage:

```text
./app/build/app predict <image> <conv_weights.bin> <dense_weights.bin> <dense_bias.bin>
```

Flow:

```text
1. Create CUDA context/workspace/handle.
2. Create descriptors.
3. Bind tensors.
4. Load checkpoint files into GPU tensors.
5. Load image from disk with stb_image.
6. Resize/copy image into CPU stack buffer.
7. Zero full GPU batch.
8. Copy single image into batch slot 0.
9. Normalize image on GPU.
10. Run forward pass.
11. Copy first row probabilities to CPU.
12. Print prediction.
13. Cleanup.
```

Notice that even for one image, the tensors are shaped for batch size 64. The
code places the image in row 0 and zeros the rest.

---

## 21. Accuracy/Evaluation Functions

### 21.1 `app_count_batch_correct`

Copies `probs` from GPU to CPU and counts how many predictions match labels.

For each row in the batch:

```text
look at columns 0..9
choose largest probability
compare to label
```

### 21.2 `app_count_batch_correct_detail`

Same as above, but also counts accuracy per digit:

```text
digit 0: correct / total
digit 1: correct / total
...
digit 9: correct / total
```

### 21.3 `app_eval_dataset`

Runs evaluation over a whole dataset.

It:

```text
1. Loops over full batches.
2. Loads images.
3. Copies images to GPU.
4. Normalizes.
5. Runs forward pass.
6. Counts correct predictions.
7. Prints total accuracy and per-digit accuracy.
```

It only evaluates full batches:

```c
int full_batches = dataset_size / DIGIT_BATCH_SIZE;
int total = full_batches * DIGIT_BATCH_SIZE;
```

Any leftover samples smaller than one batch are ignored.

---

## 22. Training Function: `train_digit_recognizer`

```c
void train_digit_recognizer(cf_math_handle *handle, int total_steps, const char *save_dir)
```

This is the heart of training.

Setup:

```text
1. Load train CSV.
2. Load test CSV.
3. Create descriptors.
4. Bind tensors.
5. Attach gradients.
6. Create shuffled order array.
7. Initialize weights.
8. Allocate pinned CPU buffers for images and labels.
```

Pinned memory:

```c
cudaMallocHost(...)
```

Pinned CPU memory can be copied to GPU faster and more reliably for async CUDA
transfers.

Main training loop:

```text
for step in 1..total_steps:
  load next batch
  zero gradients
  copy images and labels to GPU
  normalize images
  forward pass
  compute loss and dY
  backprop dense layer
  backprop pooling
  backprop ReLU
  backprop convolution
  update weights
  copy loss to CPU
  print loss and accuracy
  save checkpoint sometimes
```

The actual loop body:

```c
app_next_train_batch(...);

cf_math_zero_grad_f16(handle, &tensors.conv_w);
cf_math_zero_grad_f16(handle, &tensors.dense_w);
cf_math_zero_grad_f16(handle, &tensors.dense_b);

cudaMemcpyAsync(... images ...);
cudaMemcpyAsync(... labels ...);

app_normalize_u16_to_f16(handle, &tensors.input, &tensors.input_raw);
digit_forward(handle, &tensors);
cf_math_fused_cross_entropy(...);

cf_math_matmul_trans_b_f16(handle, &tensors.d_flat, &tensors.dY, &tensors.dense_w);
cf_math_matmul_trans_a_f16(handle, &tensors.d_dense_w, &tensors.flat, &tensors.dY);
cf_math_reduce_sum_rows_f16(handle, &tensors.d_dense_b, &tensors.dY);

cf_math_pooling_backward_f16(...);
cf_math_relu_backward_f16(...);
cf_math_conv2d_backward_data_f16(...);
cf_math_conv2d_backward_filter_f16(...);

cf_math_sgd_update_f16(handle, &tensors.conv_w, &tensors.d_conv_w, lr);
cf_math_sgd_update_f16(handle, &tensors.dense_w, &tensors.d_dense_w, lr);
cf_math_sgd_update_f16(handle, &tensors.dense_b, &tensors.d_dense_b, lr);
```

### 22.1 Forward Pass During Training

Same as prediction:

```text
input -> conv -> relu -> pool -> dense -> softmax
```

### 22.2 Cross Entropy

```c
cf_math_fused_cross_entropy(handle, &tensors.dY, &tensors.batch_loss,
                            &tensors.loss, &tensors.probs, &tensors.labels);
```

Cross entropy compares predicted probabilities to true labels.

If true label is `7`, the model wants probability at index 7 to be high.

This function also computes `dY`, the gradient at the output.

### 22.3 Dense Backward

Dense forward was:

```text
logits = flat * dense_w + dense_b
```

Backward computes:

```text
d_flat    = dY * dense_w^T
dense_dW  = flat^T * dY
dense_db  = sum rows of dY
```

In code:

```c
cf_math_matmul_trans_b_f16(handle, &tensors.d_flat, &tensors.dY, &tensors.dense_w);
cf_math_matmul_trans_a_f16(handle, &tensors.d_dense_w, &tensors.flat, &tensors.dY);
cf_math_reduce_sum_rows_f16(handle, &tensors.d_dense_b, &tensors.dY);
```

### 22.4 Pool Backward

Max pooling forward kept only the max value from each 2x2 block.

Pool backward sends gradient back only to the position that won the max.

```c
cf_math_pooling_backward_f16(...)
```

cuDNN handles the details.

### 22.5 ReLU Backward

ReLU forward:

```text
if x < 0: output = 0
else: output = x
```

ReLU backward:

```text
if original x <= 0: gradient = 0
else: gradient passes through
```

```c
cf_math_relu_backward_f16(handle, &tensors.d_conv_out,
                          &tensors.d_relu, &tensors.conv_out);
```

### 22.6 Convolution Backward

There are two useful gradients:

```text
d_input   how loss changes with respect to input image
d_conv_w  how loss changes with respect to convolution weights
```

Code:

```c
cf_math_conv2d_backward_data_f16(handle, &tensors.d_input,
                                 &tensors.d_conv_out, &tensors.conv_w,
                                 1, 1, 1, 1, 1, 1);

cf_math_conv2d_backward_filter_f16(handle, &tensors.d_conv_w,
                                   &tensors.d_conv_out, &tensors.input,
                                   1, 1, 1, 1, 1, 1);
```

The model does not update input images. It only needs `d_conv_w` for learning
conv weights. `d_input` is computed because it is part of normal backprop flow,
but there is no earlier trainable layer before the image.

### 22.7 SGD Update

SGD means stochastic gradient descent.

Formula:

```c
weight = weight - lr * gradient;
```

Here:

```c
const float lr = 0.01f;
```

Updates:

```c
cf_math_sgd_update_f16(handle, &tensors.conv_w, &tensors.d_conv_w, lr);
cf_math_sgd_update_f16(handle, &tensors.dense_w, &tensors.d_dense_w, lr);
cf_math_sgd_update_f16(handle, &tensors.dense_b, &tensors.d_dense_b, lr);
```

After many steps, weights should get better.

---

## 23. `main` in `app.cu`

```c
int main(int argc, char **argv)
```

There are two modes.

### 23.1 GPU Predict Mode

If first argument is:

```text
predict
```

then:

```c
return app_run_predict_mode(argc, argv);
```

Usage:

```text
./app/build/app predict <image> <conv_weights.bin> <dense_weights.bin> <dense_bias.bin>
```

### 23.2 Training Mode

Otherwise it trains.

Arguments:

```text
./app/build/app <steps> <save_dir> <train_csv> <test_csv> <dataset_root>
```

Defaults:

```text
steps        2
save_dir     public/checkpoints
train_csv    public/train.csv
test_csv     public/test.csv
dataset_root public
```

Then it creates:

```text
cf_math_context
cf_math_workspace
cf_math_handle
```

and calls:

```c
train_digit_recognizer(&handle, total_steps, save_dir);
```

Finally it synchronizes the CUDA stream and destroys resources.

---

## 24. How GPU Math Functions Work Under the Hood

The functions in `app.cu` call framework functions implemented in:

```text
lib/src/MATH/TYPES/cf_math_f16.cu
```

### 24.1 `cf_math_conv2d_f16`

Uses cuDNN convolution.

It gets raw device pointers by:

```c
__half *In_D = (__half *)(ptr + In->byte_offset);
```

Then configures convolution:

```c
cudnnSetConvolution2dDescriptor(... pad, stride, dilation ...);
```

Then launches:

```c
cudnnConvolutionForward(...)
```

### 24.2 `cf_math_pooling_f16`

Uses cuDNN pooling:

```c
cudnnSetPooling2dDescriptor(...)
cudnnPoolingForward(...)
```

For this model:

```text
mode = max
window = 2x2
stride = 2x2
padding = 0
```

### 24.3 `cf_math_linear_bias_f16`

Uses cuBLASLt matrix multiplication.

It tries to use cuBLASLt with a bias epilogue:

```text
Output = Input * Weight + Bias
```

If cuBLASLt setup fails, it falls back to a custom CUDA kernel.

### 24.4 `cf_math_softmax_f16`

Uses cuDNN softmax.

It creates a temporary tensor descriptor and calls:

```c
cudnnSoftmaxForward(...)
```

### 24.5 `cf_math_fused_cross_entropy`

Uses a custom CUDA kernel.

It expects the last dimension to be 16:

```c
if(P->desc->rank < 2 || P->desc->dim[P->desc->rank - 1] != 16) return;
```

So this project's padded class count is baked into that function.

### 24.6 Backward Functions

The backward functions call either cuDNN or custom kernels:

```text
relu backward       custom CUDA kernel
pool backward       cuDNN
conv backward data  cuDNN
conv backward filter cuDNN
SGD update          custom CUDA kernel
```

---

## 25. CPU Predictor vs GPU App

They should match conceptually:

```text
GPU app:
  uses half-float tensors
  uses CUDA/cuDNN/cuBLAS
  can train and predict
  works with batches of 64

CPU predictor:
  converts weights to float
  uses plain C loops
  predicts one image
  cannot train
```

Same model:

```text
conv -> relu -> pool -> dense -> softmax
```

Important difference:

```text
app.cu currently defines image size as 28x28
predict_cpu.c currently defines image size as 800x800
```

That means they do not currently agree.

If `app.cu` trained 28x28 checkpoints, then `predict_cpu.c` must also use
28x28 when loading those checkpoints.

If you want 800x800 prediction, then `app.cu` must train an 800x800 model and
save new checkpoints.

---

## 26. Exactly What to Change When Image Size Changes

Suppose you want:

```text
IMAGE_H = 800
IMAGE_W = 800
```

You must update both files.

### 26.1 In `predict_cpu.c`

Use:

```c
#define IMAGE_H    800
#define IMAGE_W    800
#define IMAGE_PIX  (IMAGE_H * IMAGE_W)
#define CONV_CH    16
#define POOL_H     (IMAGE_H / 2)
#define POOL_W     (IMAGE_W / 2)
#define FLAT       (CONV_CH * POOL_H * POOL_W)
```

Also update comments saying `28x28`, `14x14`, and `3136`.

### 26.2 In `app.cu`

Add pool macros:

```c
#define DIGIT_IMAGE_H 800
#define DIGIT_IMAGE_W 800
#define DIGIT_IMAGE_PIXELS (DIGIT_IMAGE_H * DIGIT_IMAGE_W)
#define DIGIT_POOL_H (DIGIT_IMAGE_H / 2)
#define DIGIT_POOL_W (DIGIT_IMAGE_W / 2)
#define DIGIT_FLATTENED_FEATURES \
  (DIGIT_CONV_CHANNELS * DIGIT_POOL_H * DIGIT_POOL_W)
```

Then change:

```c
int pool_dim[4] = {DIGIT_BATCH_SIZE, DIGIT_CONV_CHANNELS, 14, 14};
```

to:

```c
int pool_dim[4] = {
  DIGIT_BATCH_SIZE,
  DIGIT_CONV_CHANNELS,
  DIGIT_POOL_H,
  DIGIT_POOL_W
};
```

### 26.3 Retrain

You must retrain because dense weight shape changed.

Old checkpoint:

```text
dense_weights [3136, 16]
```

New 800x800 checkpoint:

```text
dense_weights [2560000, 16]
```

These are not compatible.

### 26.4 Memory Warning

800x800 is much larger than 28x28.

For batch size 64:

```text
input:    64 * 1  * 800 * 800
conv_out: 64 * 16 * 800 * 800
pool_out: 64 * 16 * 400 * 400
```

That is a lot of GPU memory.

`conv_out` alone in f16:

```text
64 * 16 * 800 * 800 * 2 bytes
= 1,310,720,000 bytes
= about 1.22 GiB
```

So changing to 800x800 may require reducing:

```c
#define DIGIT_BATCH_SIZE 64
```

to something much smaller, maybe `1`, `2`, `4`, or `8`, depending on GPU memory.

If batch size changes, descriptors and batch buffers already use the macro, so
most batch-related sizes follow automatically.

---

## 27. Recommended Fixes to Make the Code Less Fragile

### 27.1 Stop hardcoding pooled size

In `predict_cpu.c`:

```c
#define POOL_H (IMAGE_H / 2)
#define POOL_W (IMAGE_W / 2)
```

In `app.cu`:

```c
#define DIGIT_POOL_H (DIGIT_IMAGE_H / 2)
#define DIGIT_POOL_W (DIGIT_IMAGE_W / 2)
#define DIGIT_FLATTENED_FEATURES \
  (DIGIT_CONV_CHANNELS * DIGIT_POOL_H * DIGIT_POOL_W)
```

### 27.2 Make comments use macros

Comments currently mention old fixed numbers like:

```text
28x28
14x14
3136
```

Those comments become wrong when macros change.

### 27.3 Check file sizes in `predict_cpu.c`

The CPU loader currently reports short reads. It would be even clearer to check
for exact file size and print:

```text
expected dense_weights for FLAT=<value>, PAD_CLS=16
```

This would make shape mismatch easier to debug.

### 27.4 Reject odd image sizes or handle them deliberately

Pooling uses integer division:

```c
out_H = in_H / 2;
out_W = in_W / 2;
```

For odd sizes, one row/column is ignored. That may be okay, but it should be
intentional.

You can enforce:

```c
#if (IMAGE_H % 2) != 0 || (IMAGE_W % 2) != 0
#error "IMAGE_H and IMAGE_W must be even for 2x2 stride-2 pooling"
#endif
```

And similarly for `DIGIT_IMAGE_H/W`.

---

## 28. Running the Programs

### 28.1 Build GPU app

```sh
make app
```

This builds and runs the app target from the Makefile.

Direct training form:

```sh
./app/build/app <steps> <save_dir> <train_csv> <test_csv> <dataset_root>
```

Example:

```sh
./app/build/app 1000 public/checkpoints public/train.csv public/test.csv public
```

### 28.2 GPU prediction

```sh
./app/build/app predict \
  public/img/test_image.jpg \
  public/checkpoints/layer1_weights_step100000.bin \
  public/checkpoints/dense_weights_step100000.bin \
  public/checkpoints/dense_bias_step100000.bin
```

### 28.3 CPU predictor

Build:

```sh
make predict_cpu
```

Run:

```sh
./app/build/predict_cpu \
  public/img/test_image.jpg \
  public/checkpoints/layer1_weights_step100000.bin \
  public/checkpoints/dense_weights_step100000.bin \
  public/checkpoints/dense_bias_step100000.bin
```

The CPU predictor and GPU training code must agree on:

```text
image size
conv channel count
pool shape
flat size
padded class count
checkpoint shapes
```

---

## 29. Final Shape Checklist

When debugging dimensions, write the whole chain:

```text
IMAGE_H = ?
IMAGE_W = ?

after conv:
  CONV_CH x IMAGE_H x IMAGE_W

after pool:
  CONV_CH x (IMAGE_H / 2) x (IMAGE_W / 2)

FLAT:
  CONV_CH * (IMAGE_H / 2) * (IMAGE_W / 2)

dense weights:
  FLAT * PAD_CLS

dense bias:
  PAD_CLS

output:
  PAD_CLS, but only first REAL_CLS are real digits
```

For `28x28`:

```text
conv_out      16 x 28 x 28
pool_out      16 x 14 x 14
FLAT          3136
dense_weights 3136 x 16
```

For `800x800`:

```text
conv_out      16 x 800 x 800
pool_out      16 x 400 x 400
FLAT          2,560,000
dense_weights 2,560,000 x 16
```

If any one of these numbers is wrong, the model will either:

```text
fail to load weights,
read wrong memory,
produce nonsense predictions,
or run out of memory.
```

The code is mostly ordinary C memory management plus CUDA library calls. The
"AI" part is the agreement of these array shapes and the training loop that
changes weight arrays using gradients.
