#include "MODEL/DLModel.hpp"

#include "HANDLER/HandleCpu.hpp"
#include "HANDLER/HandleCuda.hpp"
#include "HANDLER/HandleImage.hpp"

#include <fstream>
#include <filesystem>
#include <iostream>

#include <cublasLt.h>
#include <cudnn.h>

static __global__ void reluForwardUint4Kernel(uint4 *z, uint4 *act)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  uint4 v = z[index];

  float x0 = __uint_as_float(v.x);
  float x1 = __uint_as_float(v.y);
  float x2 = __uint_as_float(v.z);
  float x3 = __uint_as_float(v.w);

  v.x = __float_as_uint(x0 > 0.0f ? x0 : 0.0f);
  v.y = __float_as_uint(x1 > 0.0f ? x1 : 0.0f);
  v.z = __float_as_uint(x2 > 0.0f ? x2 : 0.0f);
  v.w = __float_as_uint(x3 > 0.0f ? x3 : 0.0f);

  act[index] = v;
}

static __global__ void reluBackwardUint4Kernel(uint4 *z, uint4 *da, uint4 *dz)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  uint4 vz = z[index];
  uint4 va = da[index];

  float z0 = __uint_as_float(vz.x);
  float z1 = __uint_as_float(vz.y);
  float z2 = __uint_as_float(vz.z);
  float z3 = __uint_as_float(vz.w);

  float a0 = __uint_as_float(va.x);
  float a1 = __uint_as_float(va.y);
  float a2 = __uint_as_float(va.z);
  float a3 = __uint_as_float(va.w);

  uint4 out;

  out.x = __float_as_uint(z0 > 0.0f ? a0 : 0.0f);
  out.y = __float_as_uint(z1 > 0.0f ? a1 : 0.0f);
  out.z = __float_as_uint(z2 > 0.0f ? a2 : 0.0f);
  out.w = __float_as_uint(z3 > 0.0f ? a3 : 0.0f);

  dz[index] = out;
}

static __global__ void softmaxCrossEntropyKernel(
  const float *z,
  const float *targets,
  float *act,
  float *dz,
  float *lossVec)
{
  int n = blockIdx.x; // one block per image

  const int BATCH = 64;
  const int CLASSES = 10;

  if (n >= BATCH)
    return;

  const float *row = z + n * CLASSES;
  float *out = act + n * CLASSES;
  float *grad = dz + n * CLASSES;

  float maxv = row[0];
  #pragma unroll
  for (int j = 1; j < CLASSES; j++)
    if (row[j] > maxv) maxv = row[j];

  float sum = 0.0f;
  float e[CLASSES];

  #pragma unroll
  for (int j = 0; j < CLASSES; j++)
  {
    e[j] = expf(row[j] - maxv);
    sum += e[j];
  }

  int label = (int)targets[n];
  float invBatch = 1.0f / 64.0f;

  #pragma unroll
  for (int j = 0; j < CLASSES; j++)
  {
    float p = e[j] / sum;
    out[j] = p;
    grad[j] = p * invBatch;
  }

  grad[label] -= invBatch;

  float pTarget = out[label];
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

static void rowMajorGemm
(
  cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
  int m, int n, int k, const float *a, int aCols, const float *b, int bCols,
  float *c
)
{
  float alpha = 1.0f;
  float beta = 0.0f;

  cublasSgemm
  (
    handle,
    transB,
    transA,
    n,
    m,
    k,
    &alpha,
    b,
    bCols,
    a,
    aCols,
    &beta,
    c,
    n
  );
}

static __global__ void biasBackwardKernel(const float *dz, float *db, int batch, int output)
{
  int o = blockIdx.x * blockDim.x + threadIdx.x;

  if (o >= output)
    return;

  float sum = 0.0f;

  for (int n = 0; n < batch; n++)
    sum += dz[n * output + o];

  db[o] = sum;
}


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

static void printEstimationByNumber(const float *prob)
{
  const int batch = 64;
  const int classes = 10;
  const int barWidth = 15;

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

class ConvLayer: public Layer
{
public:
  ConvLayer(HandleCuda& handle): Layer(handle)
  {
#ifdef use_old
    this->setX((int[])   {64, 1, 28, 28}, 4, LT_TENSOR, OT_CONV);
    this->setW((int[])   {1, 1, 3, 3},    4, LT_FILTER, OT_CONV);
    this->setB((int[])   {1, 1, 1, 1},    4, LT_TENSOR, OT_CONV);

    this->setZ((int[])   {64, 1, 28, 28}, 4, LT_TENSOR, OT_CONV);
    this->setAct((int[]) {64, 1, 28, 28}, 4, LT_TENSOR, OT_CONV);

    this->setDz((int[])  {64, 1, 28, 28}, 4, LT_TENSOR, OT_CONV);
    this->setDw((int[])  {1, 1, 3, 3},    4, LT_FILTER, OT_CONV);
    this->setDb((int[])  {1, 1, 1, 1},    4, LT_TENSOR, OT_CONV);
#else
    int xDim[HIGHEST_RANK]   = {64, 1, 28, 28};
    int wDim[HIGHEST_RANK]   = {1, 1, 3, 3};
    int bDim[HIGHEST_RANK]   = {1, 1, 1, 1};
    int zDim[HIGHEST_RANK]   = {64, 1, 28, 28};
    int actDim[HIGHEST_RANK] = {64, 1, 28, 28};
    int dzDim[HIGHEST_RANK]  = {64, 1, 28, 28};
    int dwDim[HIGHEST_RANK]  = {1, 1, 3, 3};
    int dbDim[HIGHEST_RANK]  = {1, 1, 1, 1};

    this->setX(xDim,     4, LT_TENSOR, OT_CONV);
    this->setW(wDim,     4, LT_FILTER, OT_CONV);
    this->setB(bDim,     4, LT_TENSOR, OT_CONV);

    this->setZ(zDim,     4, LT_TENSOR, OT_CONV);
    this->setAct(actDim, 4, LT_TENSOR, OT_CONV);

    this->setDz(dzDim,   4, LT_TENSOR, OT_CONV);
    this->setDw(dwDim,   4, LT_FILTER, OT_CONV);
    this->setDb(dbDim,   4, LT_TENSOR, OT_CONV);
#endif
  }

  void convXWpB()
  {
    float alpha = 1.0f;
    float beta = 0.0f;

    void *x = (uint8_t *)this->handle->getData() + this->x.getOffset();
    void *w = (uint8_t *)this->handle->getData() + this->w.getOffset();
    void *b = (uint8_t *)this->handle->getData() + this->b.getOffset();
    void *z = (uint8_t *)this->handle->getData() + this->z.getOffset();

    cudnnConvolutionForward(
      this->handle->getContextCudnnHanler(),
      &alpha,
      this->x.getLayout()->getLayoutDescriptor().tensor,
      x,
      this->w.getLayout()->getLayoutDescriptor().filter,
      w,
      this->x.getLayout()->getOpDescriptor().conv,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
      this->handle->getWorkspaceScratchpad(),
      this->handle->getWorkspaceScratchpadSize(),
      &beta,
      this->z.getLayout()->getLayoutDescriptor().tensor,
      z
    );

    beta = 1.0f;

    cudnnAddTensor(
      this->handle->getContextCudnnHanler(),
      &alpha,
      this->b.getLayout()->getLayoutDescriptor().tensor,
      b,
      &beta,
      this->z.getLayout()->getLayoutDescriptor().tensor,
      z
    );
  }

  void activationReLU()
  {
    cudnnActivationDescriptor_t activDesc;
    cudnnCreateActivationDescriptor(&activDesc);
    cudnnSetActivationDescriptor(activDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);

    float alpha = 1.0f;
    float beta = 0.0f;

    void *z   = (uint8_t *)this->handle->getData() + this->z.getOffset();
    void *act = (uint8_t *)this->handle->getData() + this->act.getOffset();

    cudnnActivationForward
    (
      this->handle->getContextCudnnHanler(),
      activDesc,
      &alpha,
      this->z.getLayout()->getLayoutDescriptor().tensor,
      z,
      &beta,
      this->act.getLayout()->getLayoutDescriptor().tensor,
      act
    );

    cudnnDestroyActivationDescriptor(activDesc);
  }

  void backActivation(Math& daUpperLayer)
  {
    float alpha = 1.0f;
    float beta = 0.0f;

    void *x   = (uint8_t *)this->handle->getData() + this->x.getOffset();
    void *z   = (uint8_t *)this->handle->getData() + this->z.getOffset();
    void *act = (uint8_t *)this->handle->getData() + this->act.getOffset();

    void *da  = (uint8_t *)this->handle->getData() + daUpperLayer.getOffset();
    void *dz  = (uint8_t *)this->handle->getData() + this->dz.getOffset();
    void *dw  = (uint8_t *)this->handle->getData() + this->dw.getOffset();
    void *db  = (uint8_t *)this->handle->getData() + this->db.getOffset();

    cudnnActivationDescriptor_t activDesc;
    cudnnCreateActivationDescriptor(&activDesc);
    cudnnSetActivationDescriptor(activDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);

    cudnnActivationBackward
    (
      this->handle->getContextCudnnHanler(),
      activDesc,
      &alpha,
      this->act.getLayout()->getLayoutDescriptor().tensor,
      act,
      daUpperLayer.getLayout()->getLayoutDescriptor().tensor,
      da,
      this->z.getLayout()->getLayoutDescriptor().tensor,
      z,
      &beta,
      this->dz.getLayout()->getLayoutDescriptor().tensor,
      dz
    );

    cudnnDestroyActivationDescriptor(activDesc);

    cudnnConvolutionBackwardBias
    (
      this->handle->getContextCudnnHanler(),
      &alpha,
      this->dz.getLayout()->getLayoutDescriptor().tensor,
      dz,
      &beta,
      this->db.getLayout()->getLayoutDescriptor().tensor,
      db
    );

    cudnnConvolutionBackwardFilter
    (
      this->handle->getContextCudnnHanler(),
      &alpha,
      this->x.getLayout()->getLayoutDescriptor().tensor,
      x,
      this->dz.getLayout()->getLayoutDescriptor().tensor,
      dz,
      this->x.getLayout()->getOpDescriptor().conv,
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
      this->handle->getWorkspaceScratchpad(),
      this->handle->getWorkspaceScratchpadSize(),
      &beta,
      this->dw.getLayout()->getLayoutDescriptor().filter,
      dw
    );
  }

  void learnFunc(float lr)
  {
    float negLr = -lr;

    float *w  = (float *)((uint8_t *)this->handle->getData() + this->w.getOffset());
    float *b  = (float *)((uint8_t *)this->handle->getData() + this->b.getOffset());
    float *dw = (float *)((uint8_t *)this->handle->getData() + this->dw.getOffset());
    float *db = (float *)((uint8_t *)this->handle->getData() + this->db.getOffset());

    cublasSaxpy(this->handle->getContextCublasHanler(), 1 * 1 * 3 * 3, &negLr, dw, 1, w, 1);
    cublasSaxpy(this->handle->getContextCublasHanler(), 1, &negLr, db, 1, b, 1);
  }

};


class PoolLayer: public Layer
{
public:
  PoolLayer(HandleCuda& handle): Layer(handle)
  {
#ifdef use_old
    this->setX((int[])  {64, 1, 28, 28}, 4, LT_TENSOR, OT_POOL);
    this->setZ((int[])  {64, 1, 14, 14}, 4, LT_TENSOR, OT_POOL);
    this->setDz((int[]) {64, 1, 14, 14}, 4, LT_TENSOR, OT_POOL);
    this->setDa((int[]) {64, 1, 28, 28}, 4, LT_TENSOR, OT_POOL);
#else
    int xDim[HIGHEST_RANK]  = {64, 1, 28, 28};
    int zDim[HIGHEST_RANK]  = {64, 1, 14, 14};
    int dzDim[HIGHEST_RANK] = {64, 1, 14, 14};
    int daDim[HIGHEST_RANK] = {64, 1, 28, 28};

    this->setX(xDim,   4, LT_TENSOR, OT_POOL);
    this->setZ(zDim,   4, LT_TENSOR, OT_POOL);
    this->setDz(dzDim, 4, LT_TENSOR, OT_POOL);
    this->setDa(daDim, 4, LT_TENSOR, OT_POOL);
#endif
  }

  void maxPool(Math& a1)
  {
    float alpha = 1.0f;
    float beta = 0.0f;

    void *x = (uint8_t *)this->handle->getData() + a1.getOffset();
    void *poolX = (uint8_t *)this->handle->getData() + this->x.getOffset();
    void *z = (uint8_t *)this->handle->getData() + this->z.getOffset();

    cudaMemcpyAsync(poolX, x, a1.getCapacity(), cudaMemcpyDeviceToDevice, this->handle->getWorkspaceStream());

    cudnnPoolingForward(
      this->handle->getContextCudnnHanler(),
      this->x.getLayout()->getOpDescriptor().pool,
      &alpha,

      a1.getLayout()->getLayoutDescriptor().tensor,
      x,

      &beta,

      this->z.getLayout()->getLayoutDescriptor().tensor,
      z
    );
  }

  void backPool(Math& daUpperLayer)
  {
    float alpha = 1.0f;
    float beta = 0.0f;

    void *x  = (uint8_t *)this->handle->getData() + this->x.getOffset();
    void *z  = (uint8_t *)this->handle->getData() + this->z.getOffset();
    void *dz = (uint8_t *)this->handle->getData() + daUpperLayer.getOffset();
    void *da = (uint8_t *)this->handle->getData() + this->da.getOffset();

    cudnnPoolingBackward(
      this->handle->getContextCudnnHanler(),
      this->x.getLayout()->getOpDescriptor().pool,
      &alpha,

      this->z.getLayout()->getLayoutDescriptor().tensor,
      z,

      daUpperLayer.getLayout()->getLayoutDescriptor().tensor,
      dz,

      this->x.getLayout()->getLayoutDescriptor().tensor,
      x,

      &beta,

      this->da.getLayout()->getLayoutDescriptor().tensor,
      da
    );
  }
};


class ExtractFeaturesLayer: public Layer
{
public:
  ExtractFeaturesLayer(HandleCuda& handle): Layer(handle)
  {
#ifdef use_old
    this->setX((int[])   {64, 196, 0, 0}, 2, LT_MATRIX, OT_MATRIX);
    this->setW((int[])   {196, 128, 0, 0}, 2, LT_MATRIX, OT_MATRIX);
    this->setB((int[])   {1, 128, 0, 0}, 2, LT_MATRIX, OT_MATRIX);

    this->setZ((int[])   {64, 128, 0, 0}, 2, LT_MATRIX, OT_MATRIX);
    this->setAct((int[]) {64, 128, 0, 0}, 2, LT_MATRIX, OT_MATRIX);

    this->setDz((int[])  {64, 128, 0, 0}, 2, LT_MATRIX, OT_MATRIX);
    this->setDw((int[])  {196, 128, 0, 0}, 2, LT_MATRIX, OT_MATRIX);
    this->setDb((int[])  {1, 128, 0, 0}, 2, LT_MATRIX, OT_MATRIX);

    this->setDa((int[])  {64, 196, 0, 0}, 2, LT_MATRIX, OT_MATRIX);
#else
    int xDim[HIGHEST_RANK]   = {64, 196, 0, 0};
    int wDim[HIGHEST_RANK]   = {196, 128, 0, 0};
    int bDim[HIGHEST_RANK]   = {1, 128, 0, 0};
    int zDim[HIGHEST_RANK]   = {64, 128, 0, 0};
    int actDim[HIGHEST_RANK] = {64, 128, 0, 0};
    int dzDim[HIGHEST_RANK]  = {64, 128, 0, 0};
    int dwDim[HIGHEST_RANK]  = {196, 128, 0, 0};
    int dbDim[HIGHEST_RANK]  = {1, 128, 0, 0};
    int daDim[HIGHEST_RANK]  = {64, 196, 0, 0};

    this->setX(xDim,     2, LT_MATRIX, OT_MATRIX);
    this->setW(wDim,     2, LT_MATRIX, OT_MATRIX);
    this->setB(bDim,     2, LT_MATRIX, OT_MATRIX);

    this->setZ(zDim,     2, LT_MATRIX, OT_MATRIX);
    this->setAct(actDim, 2, LT_MATRIX, OT_MATRIX);

    this->setDz(dzDim,   2, LT_MATRIX, OT_MATRIX);
    this->setDw(dwDim,   2, LT_MATRIX, OT_MATRIX);
    this->setDb(dbDim,   2, LT_MATRIX, OT_MATRIX);

    this->setDa(daDim,   2, LT_MATRIX, OT_MATRIX);
#endif
  }

  void linear(Math& a1)
  {
    float alpha = 1.0f;
    float beta = 0.0f;

    void *x = (uint8_t *)this->handle->getData() + a1.getOffset();
    void *w = (uint8_t *)this->handle->getData() + this->w.getOffset();
    void *b = (uint8_t *)this->handle->getData() + this->b.getOffset();
    void *z = (uint8_t *)this->handle->getData() + this->z.getOffset();

    cublasLtMatmulDescSetAttribute(
      this->z.getLayout()->getOpDescriptor().matrix,
      CUBLASLT_MATMUL_DESC_BIAS_POINTER,
      &b,
      sizeof(b)
    );

    cublasLtMatmul(
      this->handle->getContextCublasLtHanler(),

      this->z.getLayout()->getOpDescriptor().matrix,

      &alpha,

      x,
      a1.getLayout()->getLayoutDescriptor().matrix,

      w,
      this->w.getLayout()->getLayoutDescriptor().matrix,

      &beta,

      z,
      this->z.getLayout()->getLayoutDescriptor().matrix,

      z,
      this->z.getLayout()->getLayoutDescriptor().matrix,

      NULL,

      this->handle->getWorkspaceScratchpad(),
      this->handle->getWorkspaceScratchpadSize(),

      this->handle->getWorkspaceStream()
    );
  }

  void activationReLU()
  {
    void *z = (uint8_t *)this->handle->getData() + this->z.getOffset();
    void *act = (uint8_t *)this->handle->getData() + this->act.getOffset();

    int chunks = this->act.getCapacity() / sizeof(uint4);
    int threads = 256;
    int blocks = (chunks + threads - 1) / threads;

    reluForwardUint4Kernel<<<blocks, threads, 0, this->handle->getWorkspaceStream()>>>(
      (uint4 *)z,
      (uint4 *)act
    );
  }

  void backActivation(Math& daUpperLayer)
  {
    void *z = (uint8_t *)this->handle->getData() + this->z.getOffset();
    void *da = (uint8_t *)this->handle->getData() + daUpperLayer.getOffset();
    void *dz = (uint8_t *)this->handle->getData() + this->dz.getOffset();

    int chunks = this->dz.getCapacity() / sizeof(uint4);
    int threads = 256;
    int blocks = (chunks + threads - 1) / threads;

    reluBackwardUint4Kernel<<<blocks, threads, 0, this->handle->getWorkspaceStream()>>>(
      (uint4 *)z,
      (uint4 *)da,
      (uint4 *)dz
    );
  }

  void backLinear(Math& a1)
  {
    float *x  = (float *)((uint8_t *)this->handle->getData() + a1.getOffset());
    float *w  = (float *)((uint8_t *)this->handle->getData() + this->w.getOffset());
    float *dz = (float *)((uint8_t *)this->handle->getData() + this->dz.getOffset());

    float *da = (float *)((uint8_t *)this->handle->getData() + this->da.getOffset());
    float *dw = (float *)((uint8_t *)this->handle->getData() + this->dw.getOffset());
    float *db = (float *)((uint8_t *)this->handle->getData() + this->db.getOffset());

    rowMajorGemm(this->handle->getContextCublasHanler(), CUBLAS_OP_N, CUBLAS_OP_T, 64, 196, 128, dz, 128, w, 128, da);

    rowMajorGemm(this->handle->getContextCublasHanler(), CUBLAS_OP_T, CUBLAS_OP_N, 196, 128, 64, x, 196, dz, 128, dw);

    biasBackwardKernel<<<1, 128, 0, this->handle->getWorkspaceStream()>>>(dz, db, 64, 128);
  }

  void learnFunc(float lr)
  {
    float negLr = -lr;

    float *w  = (float *)((uint8_t *)this->handle->getData() + this->w.getOffset());
    float *b  = (float *)((uint8_t *)this->handle->getData() + this->b.getOffset());
    float *dw = (float *)((uint8_t *)this->handle->getData() + this->dw.getOffset());
    float *db = (float *)((uint8_t *)this->handle->getData() + this->db.getOffset());

    cublasSaxpy(this->handle->getContextCublasHanler(), 196 * 128, &negLr, dw, 1, w, 1);
    cublasSaxpy(this->handle->getContextCublasHanler(), 128, &negLr, db, 1, b, 1);
  }

};


class DenseLayer: public Layer
{
public:
  DenseLayer(HandleCuda& handle): Layer(handle)
  {
#ifdef use_old
    this->setX((int[])   {64, 128, 0, 0}, 2, LT_MATRIX, OT_MATRIX);
    this->setW((int[])   {128, 10, 0, 0}, 2, LT_MATRIX, OT_MATRIX);
    this->setB((int[])   {1, 10, 0, 0}, 2, LT_MATRIX, OT_MATRIX);

    this->setZ((int[])   {64, 10, 0, 0}, 2, LT_MATRIX, OT_MATRIX);
    this->setAct((int[]) {64, 10, 0, 0}, 2, LT_MATRIX, OT_MATRIX);

    this->setDz((int[])  {64, 10, 0, 0}, 2, LT_MATRIX, OT_MATRIX);
    this->setDw((int[])  {128, 10, 0, 0}, 2, LT_MATRIX, OT_MATRIX);
    this->setDb((int[])  {1, 10, 0, 0}, 2, LT_MATRIX, OT_MATRIX);

    this->setDa((int[])  {64, 128, 0, 0}, 2, LT_MATRIX, OT_MATRIX);
#else
    int xDim[HIGHEST_RANK]   = {64, 128, 0, 0};
    int wDim[HIGHEST_RANK]   = {128, 10, 0, 0};
    int bDim[HIGHEST_RANK]   = {1, 10, 0, 0};
    int zDim[HIGHEST_RANK]   = {64, 10, 0, 0};
    int actDim[HIGHEST_RANK] = {64, 10, 0, 0};
    int dzDim[HIGHEST_RANK]  = {64, 10, 0, 0};
    int dwDim[HIGHEST_RANK]  = {128, 10, 0, 0};
    int dbDim[HIGHEST_RANK]  = {1, 10, 0, 0};
    int daDim[HIGHEST_RANK]  = {64, 128, 0, 0};

    this->setX(xDim,     2, LT_MATRIX, OT_MATRIX);
    this->setW(wDim,     2, LT_MATRIX, OT_MATRIX);
    this->setB(bDim,     2, LT_MATRIX, OT_MATRIX);

    this->setZ(zDim,     2, LT_MATRIX, OT_MATRIX);
    this->setAct(actDim, 2, LT_MATRIX, OT_MATRIX);

    this->setDz(dzDim,   2, LT_MATRIX, OT_MATRIX);
    this->setDw(dwDim,   2, LT_MATRIX, OT_MATRIX);
    this->setDb(dbDim,   2, LT_MATRIX, OT_MATRIX);

    this->setDa(daDim,   2, LT_MATRIX, OT_MATRIX);
#endif
  }

  void linear(Math& a1)
  {
    float alpha = 1.0f;
    float beta = 0.0f;

    void *x = (uint8_t *)this->handle->getData() + a1.getOffset();
    void *w = (uint8_t *)this->handle->getData() + this->w.getOffset();
    void *b = (uint8_t *)this->handle->getData() + this->b.getOffset();
    void *z = (uint8_t *)this->handle->getData() + this->z.getOffset();

    cublasLtMatmulDescSetAttribute(
      this->z.getLayout()->getOpDescriptor().matrix,
      CUBLASLT_MATMUL_DESC_BIAS_POINTER,
      &b,
      sizeof(b)
    );

    cublasLtMatmul(
      this->handle->getContextCublasLtHanler(),
      this->z.getLayout()->getOpDescriptor().matrix,
      &alpha,

      x,
      a1.getLayout()->getLayoutDescriptor().matrix,

      w,
      this->w.getLayout()->getLayoutDescriptor().matrix,

      &beta,

      z,
      this->z.getLayout()->getLayoutDescriptor().matrix,

      z,
      this->z.getLayout()->getLayoutDescriptor().matrix,

      NULL,
      this->handle->getWorkspaceScratchpad(),
      this->handle->getWorkspaceScratchpadSize(),
      this->handle->getWorkspaceStream()
    );
  }

  float *softmaxCrossEntropy(Math& targets)
  {
    void *z = (uint8_t *)this->handle->getData() + this->z.getOffset();
    void *act = (uint8_t *)this->handle->getData() + this->act.getOffset();
    void *dz = (uint8_t *)this->handle->getData() + this->dz.getOffset();
    void *target = (uint8_t *)this->handle->getData() + targets.getOffset();

    float *lossOut = (float *)this->handle->getWorkspaceScratchpad();
    float *lossVec = lossOut + 1;

    softmaxCrossEntropyKernel<<<64, 1, 0, this->handle->getWorkspaceStream()>>>(
      (const float *)z,
      (const float *)target,
      (float *)act,
      (float *)dz,
      lossVec
    );

    reduceLoss64Kernel<<<1, 64, 0, this->handle->getWorkspaceStream()>>>(
      lossVec,
      lossOut
    );

    return lossOut;
  }

  void backLinear(Math& a1)
  {
    float *x  = (float *)((uint8_t *)this->handle->getData() + a1.getOffset());
    float *w  = (float *)((uint8_t *)this->handle->getData() + this->w.getOffset());
    float *dz = (float *)((uint8_t *)this->handle->getData() + this->dz.getOffset());

    float *da = (float *)((uint8_t *)this->handle->getData() + this->da.getOffset());
    float *dw = (float *)((uint8_t *)this->handle->getData() + this->dw.getOffset());
    float *db = (float *)((uint8_t *)this->handle->getData() + this->db.getOffset());

    rowMajorGemm(this->handle->getContextCublasHanler(), CUBLAS_OP_N, CUBLAS_OP_T, 64, 128, 10, dz, 10, w, 10, da);

    rowMajorGemm(this->handle->getContextCublasHanler(), CUBLAS_OP_T, CUBLAS_OP_N, 128, 10, 64, x, 128, dz, 10, dw);

    biasBackwardKernel<<<1, 32, 0, this->handle->getWorkspaceStream()>>>(dz, db, 64, 10);
  }

  void learnFunc(float lr)
  {
    float negLr = -lr;

    float *w  = (float *)((uint8_t *)this->handle->getData() + this->w.getOffset());
    float *b  = (float *)((uint8_t *)this->handle->getData() + this->b.getOffset());
    float *dw = (float *)((uint8_t *)this->handle->getData() + this->dw.getOffset());
    float *db = (float *)((uint8_t *)this->handle->getData() + this->db.getOffset());

    cublasSaxpy(this->handle->getContextCublasHanler(), 128 * 10, &negLr, dw, 1, w, 1);
    cublasSaxpy(this->handle->getContextCublasHanler(), 10, &negLr, db, 1, b, 1);
  }
};


DLModel::DLModel()
{
  this->cpu = new HandleCpu();
  this->gpu = new HandleCuda();
  this->images = new HandleImage();

  this->convL = new ConvLayer(*this->gpu);
  this->poolL = new PoolLayer(*this->gpu);
  this->exfL = new ExtractFeaturesLayer(*this->gpu);
  this->dL = new DenseLayer(*this->gpu);
}

DLModel::~DLModel()
{
  delete this->dL;
  delete this->exfL;
  delete this->poolL;
  delete this->convL;

  delete this->images;
  delete this->gpu;
  delete this->cpu;
}

void DLModel::train(int iterations)
{
  std::filesystem::create_directories("public/checkpoints");

  for (int i = 0; i < iterations; i++)
  {
    this->images->getTensorAndTargets(*this->cpu, this->hostTensor, this->hostTargets);

    this->cpu->copyToDevice(*this->gpu, this->convL->x, this->hostTensor);
    this->cpu->copyToDevice(*this->gpu, this->deviceTargets, this->hostTargets);

    this->convL->convXWpB();
    this->convL->activationReLU();

    this->poolL->maxPool(this->convL->act);

    this->exfL->linear(this->poolL->z);
    this->exfL->activationReLU();

    this->dL->linear(this->exfL->act);
    this->dL->softmaxCrossEntropy(this->deviceTargets);

    float lr = 0.01f;

    this->dL->backLinear(this->exfL->act);
    this->dL->learnFunc(lr);

    this->exfL->backActivation(this->dL->da);
    this->exfL->backLinear(this->poolL->z);
    this->exfL->learnFunc(lr);

    this->poolL->backPool(this->exfL->da);

    this->convL->backActivation(this->poolL->da);
    this->convL->learnFunc(lr);


    if ((i + 1) % 32 == 0)
    {
      std::string base = "public/checkpoints/iter_" + std::to_string(i + 1) + "_";

      saveMath(*this->gpu, this->convL->w, (base + "conv_w.bin").c_str());
      saveMath(*this->gpu, this->convL->b, (base + "conv_b.bin").c_str());

      saveMath(*this->gpu, this->exfL->w, (base + "exf_w.bin").c_str());
      saveMath(*this->gpu, this->exfL->b, (base + "exf_b.bin").c_str());

      saveMath(*this->gpu, this->dL->w, (base + "dense_w.bin").c_str());
      saveMath(*this->gpu, this->dL->b, (base + "dense_b.bin").c_str());
    }
  }
}

void DLModel::estimate(const char *convW, const char *convB,
                       const char *exfW, const char *exfB,
                       const char *denseW, const char *denseB)
{
  loadMath(*this->gpu, this->convL->w, convW);
  loadMath(*this->gpu, this->convL->b, convB);

  loadMath(*this->gpu, this->exfL->w, exfW);
  loadMath(*this->gpu, this->exfL->b, exfB);

  loadMath(*this->gpu, this->dL->w, denseW);
  loadMath(*this->gpu, this->dL->b, denseB);

  this->images->getTensorAndTargets(*this->cpu, this->hostTensor, this->hostTargets);

  this->cpu->copyToDevice(*this->gpu, this->convL->x, this->hostTensor);
  this->cpu->copyToDevice(*this->gpu, this->deviceTargets, this->hostTargets);

  this->convL->convXWpB();
  this->convL->activationReLU();

  this->poolL->maxPool(this->convL->act);

  this->exfL->linear(this->poolL->z);
  this->exfL->activationReLU();

  this->dL->linear(this->exfL->act);
  this->dL->softmaxCrossEntropy(this->deviceTargets);

  cudaStreamSynchronize(this->gpu->getWorkspaceStream());

  Math hostProb;
  Math hostTarget;

  this->gpu->copyToHost(*this->cpu, hostProb, this->dL->act);
  this->gpu->copyToHost(*this->cpu, hostTarget, this->deviceTargets);

  float *prob = (float *)((uint8_t *)this->cpu->getData() + hostProb.getOffset());
  float *target = (float *)((uint8_t *)this->cpu->getData() + hostTarget.getOffset());

  printEstimationByNumber(prob);

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

  std::cout << "accuracy: " << (float)correct / 64.0f << std::endl;
  std::cout << "avg confidence: " << confidence / 64.0f << std::endl;
}
