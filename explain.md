# CUDA MNIST Digit Recognizer

## Core Shape

The app trainer in `app/src/app.cu` uses a fixed MNIST-style classifier:

```text
raw input      [64, 1, 28, 28]  cf_u16 staging storage
normalized     [64, 1, 28, 28]  f16
conv weights   [16, 1, 3, 3]    f16
conv output    [64, 16, 28, 28] f16
relu output    [64, 16, 28, 28] f16
max pool       [64, 16, 14, 14] f16
flatten view   [64, 3136]       f16, same memory as pool output
dense weights  [3136, 16]       f16
dense bias     [16]             f16
logits         [64, 16]         f16
probabilities  [64, 16]         f16
```

The true digit classes are `0..9`. Output classes `10..15` are padding only, so
the dense output width stays aligned to 16 values for Tensor Core-friendly
storage and math. Prediction ignores padded classes and only returns `0..9`.

## Mixed Precision

Tensor storage and transfers use f16 where possible. The raw image staging
tensor reuses 16-bit storage for `cf_u16` pixels, then a CUDA normalization
kernel converts each pixel to:

```text
__half((float)pixel / 65535.0f)
```

cuDNN convolution and cuBLASLt matrix operations are configured through
`cf_math` to use FP32 accumulation while reading and writing f16 tensors. The
cross-entropy kernel stores gradients as f16 but computes row loss and gradient
scaling in float.

## Dataset Pipeline

The default dataset files are:

```text
public/train.csv
public/test.csv
public/train/0..9/*.png
public/test/0..9/*.png
```

CSV rows use:

```text
filepath,label
train/0/16585.png,0
```

Paths are relative to the dataset root, which defaults to `public`. The app
loads the CSV into `sample_t { path[256], label }`, then `get_next_batch`
sequentially fills pinned CPU memory with 64 images and labels. PNG files are
read with `stbi_load_16`, forced to grayscale, and resized to 28x28 with
nearest-neighbor if needed.

To replace the batch loader, keep the same pinned-memory contract:

```c
void get_next_batch(sample_t *dataset, int dataset_size,
                    uint16_t *images_cpu, uint8_t *labels_cpu,
                    int batch_size);
```

Fill `images_cpu` with `batch_size * 28 * 28` grayscale `uint16_t` pixels and
`labels_cpu` with values in `0..9`.

## Training Flow

Each step runs:

```text
zero grads
copy pinned image and label batch to GPU
normalize cf_u16 -> f16
conv2d -> relu -> max pool -> linear+bias -> softmax
fused cross entropy
dense dX, dense dW, dense db
pool backward -> relu backward -> conv data/filter backward
SGD update for conv weights, dense weights, dense bias
```

`cf_math_fused_cross_entropy` now infers the batch size from the probability
tensor shape and supports `[64,16]`, scaling gradients by `1.0f / batch`.

## Checkpoints

Checkpoints are written every 512 steps and at the final step. Paths are built
with `snprintf` and remain relative to the supplied save directory:

```text
<save_dir>/layer1_weights_step<step>.bin
<save_dir>/dense_weights_step<step>.bin
<save_dir>/dense_bias_step<step>.bin
```

Files contain raw `__half` bytes copied from device memory after synchronizing
the CUDA stream. There is no header; shape and dtype are defined by the model
contract above.

## Running

Default smoke run:

```sh
./app/build/app 2 checkpoints
```

Full argument form:

```sh
./app/build/app <steps> <save_dir> <train_csv> <test_csv> <dataset_root>
```

Defaults are:

```text
steps        2
save_dir     checkpoints
train_csv    public/train.csv
test_csv     public/test.csv
dataset_root public
```
