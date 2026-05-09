/*
 * CF Framework — CPU-only digit predictor
 * Copyright (C) 2026 Orion
 *
 * No CUDA / GPU required. Loads saved checkpoint .bin files (raw __half bytes)
 * and runs the full forward pass in float32 entirely on the CPU.
 *
 * Architecture:
 *   input  [1, 28, 28]   float32, normalized 0..1
 *   conv   [16, 1, 3x3]  pad=1, stride=1  → [16, 28, 28]
 *   relu                               → [16, 28, 28]
 *   maxpool 2x2, stride=2              → [16, 14, 14]
 *   flatten                            → [3136]
 *   linear [3136, 16] + bias[16]       → [16]
 *   softmax                            → [16]
 *   argmax of [0..9]
 *
 * Weight files (raw little-endian IEEE 754 half-precision):
 *   layer1_weights_stepN.bin  — conv weights  [16,1,3,3]  = 144 halves
 *   dense_weights_stepN.bin   — dense weights [3136,16]   = 50176 halves
 *   dense_bias_stepN.bin      — dense bias    [16]        = 16 halves
 *
 * Usage:
 *   ./app/build/predict_cpu <image> <conv_w.bin> <dense_w.bin> <dense_b.bin>
 */

#define STB_IMAGE_IMPLEMENTATION
#include "RUNTIME/stb_image.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Constants matching the CUDA trainer                                  */
/* ------------------------------------------------------------------ */

#define IMAGE_H    28
#define IMAGE_W    28
#define IMAGE_PIX  (IMAGE_H * IMAGE_W)   /* 784 */
#define CONV_CH    16                    /* output channels of conv1 */
#define POOL_H     14
#define POOL_W     14
#define FLAT       (CONV_CH * POOL_H * POOL_W)  /* 3136 */
#define PAD_CLS    16                   /* padded logit width (Tensor Core aligned) */
#define REAL_CLS   10                   /* true digit classes 0..9 */

/* ------------------------------------------------------------------ */
/* IEEE 754 half-precision → float32                                    */
/* ------------------------------------------------------------------ */

static float f16_to_f32(uint16_t h)
{
  uint32_t sign = (uint32_t)((h >> 15) & 1u);
  uint32_t exp  = (uint32_t)((h >> 10) & 0x1Fu);
  uint32_t mant = (uint32_t)(h & 0x3FFu);
  uint32_t f;

  if(exp == 0u) {
    if(mant == 0u) {
      f = sign << 31;
    } else {
      /* Subnormal half → normalised float */
      exp = 1u;
      while(!(mant & 0x400u)) { mant <<= 1; --exp; }
      mant &= 0x3FFu;
      f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
  } else if(exp == 31u) {
    /* Inf / NaN */
    f = (sign << 31) | 0x7F800000u | (mant << 13);
  } else {
    f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
  }

  float result;
  memcpy(&result, &f, sizeof(f));
  return result;
}

/* ------------------------------------------------------------------ */
/* Weight loading helpers                                               */
/* ------------------------------------------------------------------ */

static float *load_f16_weights(const char *path, size_t n_elems)
{
  FILE *f = fopen(path, "rb");
  if(f == NULL) { fprintf(stderr, "predict_cpu: cannot open %s\n", path); return NULL; }

  size_t bytes = n_elems * sizeof(uint16_t);
  uint16_t *raw = (uint16_t *)malloc(bytes);
  if(raw == NULL) { fclose(f); return NULL; }

  if(fread(raw, 1, bytes, f) != bytes) {
    fprintf(stderr, "predict_cpu: short read on %s (expected %zu bytes)\n", path, bytes);
    free(raw); fclose(f); return NULL;
  }
  fclose(f);

  float *out = (float *)malloc(n_elems * sizeof(float));
  if(out == NULL) { free(raw); return NULL; }

  for(size_t i = 0; i < n_elems; ++i) out[i] = f16_to_f32(raw[i]);
  free(raw);
  return out;
}

/* ------------------------------------------------------------------ */
/* Image loading                                                         */
/* ------------------------------------------------------------------ */

static int load_image(const char *path, float *out_pixels)
{
  int w = 0, h = 0, c = 0;
  uint16_t *raw = stbi_load_16(path, &w, &h, &c, 1);
  if(raw == NULL || w <= 0 || h <= 0) {
    fprintf(stderr, "predict_cpu: cannot load image: %s\n", path);
    if(raw != NULL) stbi_image_free(raw);
    return 0;
  }

  /* Nearest-neighbour resize to 28×28 */
  for(int y = 0; y < IMAGE_H; ++y) {
    int sy = (y * h) / IMAGE_H;
    for(int x = 0; x < IMAGE_W; ++x) {
      int sx = (x * w) / IMAGE_W;
      out_pixels[y * IMAGE_W + x] = (float)raw[sy * w + sx] * (1.0f / 65535.0f);
    }
  }

  stbi_image_free(raw);
  return 1;
}

/* ------------------------------------------------------------------ */
/* Forward pass — pure C / float32                                      */
/* ------------------------------------------------------------------ */

/*
 * conv2d_3x3_pad1
 *   weight layout: [out_ch][in_ch=1][ky][kx] row-major
 *   input  layout: [in_ch=1][H][W]
 *   output layout: [out_ch][H][W]
 */
static void conv2d_3x3_pad1(const float *input, const float *weight,
                             float *output,
                             int out_ch, int H, int W)
{
  for(int oc = 0; oc < out_ch; ++oc) {
    const float *krow = weight + oc * 9; /* kernel [3×3] for this channel, in_ch=1 */
    float *orow = output + oc * H * W;

    for(int y = 0; y < H; ++y) {
      for(int x = 0; x < W; ++x) {
        float acc = 0.0f;
        for(int ky = 0; ky < 3; ++ky) {
          int sy = y + ky - 1;
          if(sy < 0 || sy >= H) continue;
          for(int kx = 0; kx < 3; ++kx) {
            int sx = x + kx - 1;
            if(sx < 0 || sx >= W) continue;
            acc += krow[ky * 3 + kx] * input[sy * W + sx];
          }
        }
        orow[y * W + x] = acc;
      }
    }
  }
}

/* ReLU in-place: [ch][H][W] */
static void relu_inplace(float *buf, int n)
{
  for(int i = 0; i < n; ++i)
    if(buf[i] < 0.0f) buf[i] = 0.0f;
}

/*
 * max_pool_2x2_stride2
 *   input  [ch][in_H][in_W]
 *   output [ch][out_H][out_W] where out = in / 2
 */
static void max_pool_2x2(const float *input, float *output,
                         int ch, int in_H, int in_W)
{
  int out_H = in_H / 2;
  int out_W = in_W / 2;

  for(int c = 0; c < ch; ++c) {
    const float *iplane = input  + c * in_H  * in_W;
    float       *oplane = output + c * out_H * out_W;

    for(int y = 0; y < out_H; ++y) {
      for(int x = 0; x < out_W; ++x) {
        int sy = y * 2;
        int sx = x * 2;
        float m = iplane[sy * in_W + sx];
        if(iplane[sy * in_W + sx + 1]       > m) m = iplane[sy * in_W + sx + 1];
        if(iplane[(sy+1) * in_W + sx]       > m) m = iplane[(sy+1) * in_W + sx];
        if(iplane[(sy+1) * in_W + sx + 1]   > m) m = iplane[(sy+1) * in_W + sx + 1];
        oplane[y * out_W + x] = m;
      }
    }
  }
}

/*
 * linear_bias
 *   weight layout: [in_features][out_features] row-major  (matches dense_w_dim[2])
 *   logits[j] = sum_i(flat[i] * weight[i*out + j]) + bias[j]
 */
static void linear_bias(const float *flat, const float *weight, const float *bias,
                        float *logits, int in_features, int out_features)
{
  for(int j = 0; j < out_features; ++j) {
    float acc = bias[j];
    for(int i = 0; i < in_features; ++i)
      acc += flat[i] * weight[i * out_features + j];
    logits[j] = acc;
  }
}

/* Softmax in-place over first `n` elements */
static void softmax_inplace(float *v, int n)
{
  float mx = v[0];
  for(int i = 1; i < n; ++i) if(v[i] > mx) mx = v[i];

  float sum = 0.0f;
  for(int i = 0; i < n; ++i) { v[i] = expf(v[i] - mx); sum += v[i]; }

  float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
  for(int i = 0; i < n; ++i) v[i] *= inv;
}

/* ------------------------------------------------------------------ */
/* main                                                                  */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
  if(argc < 5) {
    printf("CF Framework — CPU digit predictor (no GPU required)\n\n");
    printf("Usage:\n");
    printf("  %s <image> <conv_weights.bin> <dense_weights.bin> <dense_bias.bin>\n\n", argv[0]);
    printf("Typical invocation after training:\n");
    printf("  %s public/img/test_image.jpg \\\n", argv[0]);
    printf("    public/checkpoints/layer1_weights_step2.bin \\\n");
    printf("    public/checkpoints/dense_weights_step2.bin \\\n");
    printf("    public/checkpoints/dense_bias_step2.bin\n");
    return 1;
  }

  const char *img_path     = argv[1];
  const char *conv_w_path  = argv[2];
  const char *dense_w_path = argv[3];
  const char *dense_b_path = argv[4];

  /* ----- load weights ----- */
  /* conv_w: [CONV_CH][1][3][3] = 16*9 = 144 elements */
  float *conv_w  = load_f16_weights(conv_w_path,  (size_t)CONV_CH * 9);
  /* dense_w: [FLAT][PAD_CLS] = 3136*16 = 50176 elements */
  float *dense_w = load_f16_weights(dense_w_path, (size_t)FLAT * PAD_CLS);
  /* dense_b: [PAD_CLS] = 16 elements */
  float *dense_b = load_f16_weights(dense_b_path, (size_t)PAD_CLS);

  if(!conv_w || !dense_w || !dense_b) {
    fprintf(stderr, "predict_cpu: failed to load weights\n");
    free(conv_w); free(dense_w); free(dense_b);
    return 1;
  }

  /* ----- allocate intermediate buffers ----- */
  float *input    = (float *)calloc(IMAGE_PIX,            sizeof(float)); /* [1,28,28] */
  float *conv_out = (float *)calloc(CONV_CH * IMAGE_PIX,  sizeof(float)); /* [16,28,28] */
  float *pool_out = (float *)calloc(FLAT,                 sizeof(float)); /* [16,14,14] */
  float  logits[PAD_CLS];

  if(!input || !conv_out || !pool_out) {
    fprintf(stderr, "predict_cpu: out of memory\n");
    free(conv_w); free(dense_w); free(dense_b);
    free(input); free(conv_out); free(pool_out);
    return 1;
  }

  /* ----- load and normalise image ----- */
  if(!load_image(img_path, input)) {
    free(conv_w); free(dense_w); free(dense_b);
    free(input); free(conv_out); free(pool_out);
    return 1;
  }

  /* ----- forward pass ----- */
  conv2d_3x3_pad1(input, conv_w, conv_out, CONV_CH, IMAGE_H, IMAGE_W);
  relu_inplace(conv_out, CONV_CH * IMAGE_PIX);
  max_pool_2x2(conv_out, pool_out, CONV_CH, IMAGE_H, IMAGE_W);
  linear_bias(pool_out, dense_w, dense_b, logits, FLAT, PAD_CLS);
  softmax_inplace(logits, PAD_CLS); /* softmax over all 16; padding slots soak near-zero prob */

  /* ----- pick best real class ----- */
  int   best   = 0;
  float best_p = logits[0];
  for(int i = 1; i < REAL_CLS; ++i)
    if(logits[i] > best_p) { best_p = logits[i]; best = i; }

  /* Renormalise real-class probs for display (remove padding probability mass) */
  float real_sum = 0.0f;
  for(int i = 0; i < REAL_CLS; ++i) real_sum += logits[i];

  /* ----- output ----- */
  printf("\nCF Framework — CPU digit predictor\n");
  printf("image : %s\n", img_path);
  printf("weights : conv=%s  dense=%s  bias=%s\n\n", conv_w_path, dense_w_path, dense_b_path);

  printf("predicted digit : %d", best);
  if(real_sum > 0.0f) printf("  (confidence %.2f%%)", (best_p / real_sum) * 100.0f);
  printf("\n\n");

  printf("class probabilities (digits 0-9):\n");
  for(int i = 0; i < REAL_CLS; ++i) {
    float p_norm = real_sum > 0.0f ? logits[i] / real_sum : 0.0f;
    int bar = (int)(p_norm * 40.0f + 0.5f);
    printf("  %d |", i);
    for(int b = 0; b < bar; ++b) printf(i == best ? "=" : "-");
    printf("%*s %.2f%%", 40 - bar, "", p_norm * 100.0f);
    if(i == best) printf("  <-- predicted");
    printf("\n");
  }
  printf("\n");

  free(conv_w); free(dense_w); free(dense_b);
  free(input); free(conv_out); free(pool_out);
  return 0;
}
