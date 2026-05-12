#if !defined(DL_KERNELS_CUH)
#define DL_KERNELS_CUH

/******************************************************************************\
 * Kernels.cuh — Shared CUDA kernels and helper functions for MODEL layers.
 *
 * Mixed-precision strategy:
 *   STORAGE:  __half  (fp16) — halves memory bandwidth
 *   COMPUTE:  float   (fp32) — full precision for accumulation
 *   VECTOR:   half2 via uint4 — 8 halfs per uint4 load (128-bit)
 *
 * This header-only file contains:
 *   1. Element-wise kernels  (ReLU fwd/bwd via half2/uint4 vectorization)
 *   2. Loss kernels          (softmax + cross-entropy in fp32 compute)
 *   3. Linear-algebra helpers (cublas Hgemm wrapper for fp16, bias add/bwd)
 *   4. SGD update kernel     (fp16 weights updated in fp32)
 *   5. Weight-init helpers   (random normal → fp16, zero fill)
 *   6. Checkpoint I/O        (saveMath / loadMath)
 *   7. Diagnostic helpers    (print training progress, estimation bars)
 *
 * All kernel functions are declared `static` so each translation unit that
 * includes this header gets its own copy — no ODR violations.
\******************************************************************************/

#include "VIEW/Math.hpp"
#include "HANDLER/HandleCuda.hpp"

#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cudnn.h>

/* ═══════════════════════════════════════════════════════════════════════════ *
 *  Helper: half2 ↔ uint32 for vectorized access
 * ═══════════════════════════════════════════════════════════════════════════ */

static __device__ __forceinline__ unsigned int half2_to_u32(__half2 x)
{
  union { unsigned int u; __half2 h; } v;
  v.h = x;
  return v.u;
}

static __device__ __forceinline__ __half2 u32_to_half2(unsigned int x)
{
  union { unsigned int u; __half2 h; } v;
  v.u = x;
  return v.h;
}

/* ═══════════════════════════════════════════════════════════════════════════ *
 *  1. Element-wise Kernels — ReLU via half2/uint4 vectorization
 *
 *  Each uint4 = 4 × uint32 = 4 × half2 = 8 __half elements.
 *  We process in half2 pairs, using __hgt2 for the comparison
 *  and fp16 zero masking so we never leave fp16 domain for ReLU.
 * ═══════════════════════════════════════════════════════════════════════════ */

static __global__ void reluForwardHalf2Kernel(const uint4 *z, uint4 *act, int N8)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= N8) return;

  uint4 v = z[index];
  __half2 zero = __float2half2_rn(0.0f);

  /* For each half2 pair: keep if > 0, else zero */
  __half2 h0 = u32_to_half2(v.x);
  __half2 h1 = u32_to_half2(v.y);
  __half2 h2 = u32_to_half2(v.z);
  __half2 h3 = u32_to_half2(v.w);

  /* mask = (h > 0) ? 0xFFFF : 0x0000 per lane */
  __half2 m0 = __hgt2(h0, zero);
  __half2 m1 = __hgt2(h1, zero);
  __half2 m2 = __hgt2(h2, zero);
  __half2 m3 = __hgt2(h3, zero);

  /* Bitwise AND: positive values pass, negatives → 0 */
  uint4 out;
  out.x = half2_to_u32(__hmul2(h0, m0));
  out.y = half2_to_u32(__hmul2(h1, m1));
  out.z = half2_to_u32(__hmul2(h2, m2));
  out.w = half2_to_u32(__hmul2(h3, m3));

  act[index] = out;
}

static __global__ void reluBackwardHalf2Kernel(const uint4 *z, const uint4 *da, uint4 *dz, int N8)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= N8) return;

  uint4 vz = z[index];
  uint4 va = da[index];
  __half2 zero = __float2half2_rn(0.0f);

  __half2 z0 = u32_to_half2(vz.x);
  __half2 z1 = u32_to_half2(vz.y);
  __half2 z2 = u32_to_half2(vz.z);
  __half2 z3 = u32_to_half2(vz.w);

  __half2 a0 = u32_to_half2(va.x);
  __half2 a1 = u32_to_half2(va.y);
  __half2 a2 = u32_to_half2(va.z);
  __half2 a3 = u32_to_half2(va.w);

  __half2 m0 = __hgt2(z0, zero);
  __half2 m1 = __hgt2(z1, zero);
  __half2 m2 = __hgt2(z2, zero);
  __half2 m3 = __hgt2(z3, zero);

  uint4 out;
  out.x = half2_to_u32(__hmul2(a0, m0));
  out.y = half2_to_u32(__hmul2(a1, m1));
  out.z = half2_to_u32(__hmul2(a2, m2));
  out.w = half2_to_u32(__hmul2(a3, m3));

  dz[index] = out;
}

/* ═══════════════════════════════════════════════════════════════════════════ *
 *  2. Loss Kernels — fp32 compute for numerical stability
 *
 *  Softmax + cross-entropy is always done in fp32 even though
 *  inputs/outputs are __half. We load __half, widen to float,
 *  compute, and write back __half.
 * ═══════════════════════════════════════════════════════════════════════════ */

static __global__ void softmaxCrossEntropyKernel(
  const __half *z,
  const __half *targets,
  __half *act,
  __half *dz,
  float *lossVec)
{
  int n = blockIdx.x; // one block per image

  const int BATCH = 64;
  const int CLASSES = 10;

  if (n >= BATCH)
    return;

  const __half *row = z + n * CLASSES;
  __half *out = act + n * CLASSES;
  __half *grad = dz + n * CLASSES;

  /* Load to fp32 for stable softmax */
  float vals[CLASSES];
  for (int j = 0; j < CLASSES; j++)
    vals[j] = __half2float(row[j]);

  float maxv = vals[0];
  #pragma unroll
  for (int j = 1; j < CLASSES; j++)
    if (vals[j] > maxv) maxv = vals[j];

  float sum = 0.0f;
  float e[CLASSES];

  #pragma unroll
  for (int j = 0; j < CLASSES; j++)
  {
    e[j] = expf(vals[j] - maxv);
    sum += e[j];
  }

  int label = (int)__half2float(targets[n]);
  float invBatch = 1.0f / 64.0f;

  float gradF[CLASSES];
  #pragma unroll
  for (int j = 0; j < CLASSES; j++)
  {
    float p = e[j] / sum;
    out[j] = __float2half(p);
    gradF[j] = p * invBatch;
  }

  gradF[label] -= invBatch;

  /* Write gradients back as __half */
  for (int j = 0; j < CLASSES; j++)
    grad[j] = __float2half(gradF[j]);

  float pTarget = __half2float(out[label]);
  lossVec[n] = -logf(fmaxf(pTarget, 1e-12f));
}

static __global__ void reduceLoss64Kernel(float *lossVec, float *lossOut)
{
  __shared__ float s[64];

  int i = threadIdx.x;
  s[i] = lossVec[i];
  __syncthreads();

  for (int stride = 32; stride > 0; stride >>= 1)
  {
    if (i < stride)
      s[i] += s[i + stride];
    __syncthreads();
  }

  if (i == 0)
    lossOut[0] = s[0] / 64.0f;
}

/* ═══════════════════════════════════════════════════════════════════════════ *
 *  3. Linear-Algebra Helpers — fp16 storage, fp32 compute via Hgemm
 *
 *  cublasGemmEx with CUDA_R_16F inputs and CUBLAS_COMPUTE_32F accumulation.
 *  This uses Tensor Cores when available (sm_70+).
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Row-major GEMM wrapper for fp16 storage + fp32 compute.
 *
 * cuBLAS is column-major, so we swap A↔B and m↔n to get row-major semantics:
 *   C[m×n] = A[m×k] · B[k×n]   (all stored as __half)
 */
static void rowMajorGemmHalf
(
  cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
  int m, int n, int k, const __half *a, int aCols, const __half *b, int bCols,
  __half *c
)
{
  float alpha = 1.0f;
  float beta = 0.0f;

  cublasGemmEx(
    handle,
    transB, transA,
    n, m, k,
    &alpha,
    b, CUDA_R_16F, bCols,
    a, CUDA_R_16F, aCols,
    &beta,
    c, CUDA_R_16F, n,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
  );
}

/**
 * Bias backward: sum dz over the batch dimension.
 * Uses fp32 compute to avoid fp16 accumulation overflow.
 */
static __global__ void biasBackwardKernel(const __half *dz, __half *db, int batch, int output)
{
  int o = blockIdx.x * blockDim.x + threadIdx.x;

  if (o >= output)
    return;

  float sum = 0.0f;

  for (int n = 0; n < batch; n++)
    sum += __half2float(dz[n * output + o]);

  db[o] = __float2half(sum);
}

/**
 * Add bias to each row of z.
 * Load __half → fp32 add → store __half.
 */
static __global__ void addBiasKernel(__half *z, const __half *b, int batch, int output)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * output;

  if (index >= total)
    return;

  float val = __half2float(z[index]) + __half2float(b[index % output]);
  z[index] = __float2half(val);
}

/* ═══════════════════════════════════════════════════════════════════════════ *
 *  4. SGD Update Kernel — fp32 compute on fp16 weights
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * w = w - lr * dw   (element-wise, fp32 compute, fp16 storage)
 * Uses half2/uint4 vectorization for maximum throughput.
 */
static __global__ void sgdUpdateHalf2Kernel(uint4 *w, const uint4 *dw, float lr, int N8)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= N8) return;

  uint4 vw = w[index];
  uint4 vd = dw[index];

  /* Process 8 halfs (4 × half2) per thread */
  __half2 w0 = u32_to_half2(vw.x);
  __half2 w1 = u32_to_half2(vw.y);
  __half2 w2 = u32_to_half2(vw.z);
  __half2 w3 = u32_to_half2(vw.w);

  __half2 d0 = u32_to_half2(vd.x);
  __half2 d1 = u32_to_half2(vd.y);
  __half2 d2 = u32_to_half2(vd.z);
  __half2 d3 = u32_to_half2(vd.w);

  __half2 hlr = __float2half2_rn(lr);

  /* w -= lr * dw  (in half2) */
  w0 = __hsub2(w0, __hmul2(hlr, d0));
  w1 = __hsub2(w1, __hmul2(hlr, d1));
  w2 = __hsub2(w2, __hmul2(hlr, d2));
  w3 = __hsub2(w3, __hmul2(hlr, d3));

  uint4 out;
  out.x = half2_to_u32(w0);
  out.y = half2_to_u32(w1);
  out.z = half2_to_u32(w2);
  out.w = half2_to_u32(w3);

  w[index] = out;
}

/**
 * Scalar tail for elements not divisible by 8.
 */
static __global__ void sgdUpdateHalfTailKernel(__half *w, const __half *dw, float lr, int start, int N)
{
  int index = start + blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= N) return;

  float wf = __half2float(w[index]);
  float df = __half2float(dw[index]);
  w[index] = __float2half(wf - lr * df);
}

/**
 * Convenience: run vectorized + tail SGD update.
 */
static void sgdUpdateHalf(__half *w, const __half *dw, float lr, int N, cudaStream_t stream)
{
  int N8 = N / 8;
  int tail = N - N8 * 8;

  if (N8 > 0)
  {
    int threads = 256;
    int blocks = (N8 + threads - 1) / threads;
    sgdUpdateHalf2Kernel<<<blocks, threads, 0, stream>>>((uint4 *)w, (const uint4 *)dw, lr, N8);
  }

  if (tail > 0)
  {
    int threads = 256;
    int blocks = (tail + threads - 1) / threads;
    sgdUpdateHalfTailKernel<<<blocks, threads, 0, stream>>>(w, dw, lr, N8 * 8, N);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════ *
 *  5. Weight Initialization — generate in fp32, store as __half
 * ═══════════════════════════════════════════════════════════════════════════ */

static void initMathRandom(HandleCuda& gpu, Math& m, float scale)
{
  static std::mt19937 rng(1337);
  std::normal_distribution<float> dist(0.0f, scale);

  size_t count = m.getCapacity() / sizeof(__half);
  std::vector<__half> host(count);

  for (size_t i = 0; i < count; i++)
    host[i] = __float2half(dist(rng));

  cudaMemcpy((uint8_t *)gpu.getData() + m.getOffset(), host.data(), m.getCapacity(), cudaMemcpyHostToDevice);
}

static void initMathZero(HandleCuda& gpu, Math& m)
{
  cudaMemset((uint8_t *)gpu.getData() + m.getOffset(), 0, m.getCapacity());
}

/* ═══════════════════════════════════════════════════════════════════════════ *
 *  6. Checkpoint I/O
 * ═══════════════════════════════════════════════════════════════════════════ */

static void saveMath(HandleCuda& gpu, Math& m, const char *path)
{
  void *host;
  cudaMallocHost(&host, m.getCapacity());
  cudaMemcpy(host, (uint8_t*)gpu.getData() + m.getOffset(), m.getCapacity(), cudaMemcpyDeviceToHost);

  std::ofstream file(path, std::ios::binary);
  file.write((char*)host, m.getCapacity());

  cudaFreeHost(host);
}

static void loadMath(HandleCuda& gpu, Math& m, const char *path)
{
  void *host;
  cudaMallocHost(&host, m.getCapacity());

  std::ifstream file(path, std::ios::binary);
  file.read((char*)host, m.getCapacity());

  cudaMemcpy((uint8_t*)gpu.getData() + m.getOffset(), host, m.getCapacity(), cudaMemcpyHostToDevice);
  cudaFreeHost(host);
}

/* ═══════════════════════════════════════════════════════════════════════════ *
 *  7. Diagnostic Helpers — read fp16 from GPU, convert to fp32 for display
 * ═══════════════════════════════════════════════════════════════════════════ */

static void printTrainingProgress(HandleCuda& gpu, int iteration, float *lossDevice, Math& probs, Math& targets)
{
  float loss = 0.0f;

  /* Probs & targets are fp16 on device — copy and convert */
  __half probH[64 * 10];
  __half targetH[64];

  cudaMemcpyAsync(&loss, lossDevice, sizeof(float), cudaMemcpyDeviceToHost, gpu.getWorkspaceStream());
  cudaMemcpyAsync(probH, (uint8_t *)gpu.getData() + probs.getOffset(), sizeof(probH), cudaMemcpyDeviceToHost, gpu.getWorkspaceStream());
  cudaMemcpyAsync(targetH, (uint8_t *)gpu.getData() + targets.getOffset(), sizeof(targetH), cudaMemcpyDeviceToHost, gpu.getWorkspaceStream());
  cudaStreamSynchronize(gpu.getWorkspaceStream());

  /* Convert to float for analysis */
  float prob[64 * 10];
  float target[64];
  for (int i = 0; i < 64 * 10; i++) prob[i] = __half2float(probH[i]);
  for (int i = 0; i < 64; i++) target[i] = __half2float(targetH[i]);

  int correct = 0;
  float confidence = 0.0f;

  for (int n = 0; n < 64; n++)
  {
    int best = 0;

    for (int c = 1; c < 10; c++)
      if (prob[n * 10 + c] > prob[n * 10 + best])
        best = c;

    confidence += prob[n * 10 + best];

    if (best == (int)target[n])
      correct++;
  }

  std::cout << "iter " << iteration
            << " loss: " << loss
            << " accuracy: " << (float)correct / 64.0f
            << " avg confidence: " << confidence / 64.0f
            << std::endl;
}

static void printEstimationByNumber(const __half *probH)
{
  const int batch = 64;
  const int classes = 10;
  const int barWidth = 15;

  /* Convert to float */
  float prob[batch * classes];
  for (int i = 0; i < batch * classes; i++) prob[i] = __half2float(probH[i]);

  float totals[classes] = {0.0f};

  for (int n = 0; n < batch; n++)
    for (int c = 0; c < classes; c++)
      totals[c] += prob[n * classes + c];

  for (int c = 0; c < classes; c++)
  {
    float avg = totals[c] / (float)batch;
    int percent = (int)(avg * 100.0f + 0.5f);
    int filled = (percent * barWidth + 50) / 100;

    if (filled < 0) filled = 0;
    if (filled > barWidth) filled = barWidth;

    std::cout << "[";

    for (int i = 0; i < barWidth; i++)
      std::cout << (i < filled ? "=" : " ");

    std::cout << "] " << c << " " << percent << "%" << std::endl;
  }
}

#endif /* DL_KERNELS_CUH */
