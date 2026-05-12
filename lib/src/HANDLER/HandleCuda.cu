#include "VIEW/Math.hpp"
#include "HANDLER/HandleCpu.hpp"
#include "HANDLER/HandleCuda.hpp"

#include <iostream>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#define ALIGN_128(x) (x + 0x0F) & ~0x0F

HandleCuda::HandleCuda(size_t capacity)
{
  /* Initial Var */
  std::string err_msg = "";
  this->workspace.scratchpad_size = ALIGN_128(WORKSPACE_SCRATCH_SPACE * sizeof(float));
  this->capacity = ALIGN_128(capacity * sizeof(__half));

  /* Initialize Workspace */
  if(cudaMalloc(&this->workspace.scratchpad, this->workspace.scratchpad_size) != cudaSuccess)
  {err_msg = "FATAL ERROR: Workspace Memory was not initialized before execution!"; goto _throw;}
  
  if(cudaStreamCreate(&this->workspace.stream) != cudaSuccess)
  {err_msg = "FATAL ERROR: CUDA Stream was not initialized before execution!"; goto _free_workspace;}

  /* Initialize Context */
  if(cublasCreate(&this->ctx.cublas) != CUBLAS_STATUS_SUCCESS)
  {err_msg = "FATAL ERROR: Cublas was not initialized before execution!"; goto _destroy_stream;}
  cublasSetStream(this->ctx.cublas, this->workspace.stream);

  if(cublasLtCreate(&this->ctx.cublasLt) != CUBLAS_STATUS_SUCCESS)
  {err_msg = "FATAL ERROR: CublasLT was not initialized before execution!"; goto _destroy_cublas;}
  
  if(cudnnCreate(&this->ctx.cudnn) != CUDNN_STATUS_SUCCESS)
  {err_msg = "FATAL ERROR: Cudnn was not initialized before execution!"; goto _destroy_cublasLT;}
  cudnnSetStream(this->ctx.cudnn, this->workspace.stream);
  
  /* Initialize Handler backend data */
  if(cudaMalloc(&this->backend, this->capacity) != cudaSuccess)
  {err_msg = "FATAL ERROR: Handler Memory was not initialized before execution!"; goto _destroy_cudnn;}
  
  return;

_destroy_cudnn:
  if(cudnnDestroy(this->ctx.cudnn) != CUDNN_STATUS_SUCCESS)
    std::cerr << "[WARNING] Cudnn Failed to be Destroyed!\n";
_destroy_cublasLT:
  if(cublasLtDestroy(this->ctx.cublasLt) != CUBLAS_STATUS_SUCCESS)
    std::cerr << "[WARNING] CublasLT Failed to be Destroyed!\n";
_destroy_cublas:
  if(cublasDestroy(this->ctx.cublas) != CUBLAS_STATUS_SUCCESS)
    std::cerr << "[WARNING] Cublas Failed to be Destroyed!\n";
_destroy_stream:
  if(cudaStreamDestroy(this->workspace.stream) != cudaSuccess)
    std::cerr << "[WARNING] CUDA Stream Failed to be Destroyed!\n";
_free_workspace:
  if(cudaFree(this->workspace.scratchpad) != cudaSuccess)
    std::cerr << "[WARNING] Handler Memory Failed to be Freed!\n";
_throw:
    throw std::runtime_error(err_msg);
}


HandleCuda::~HandleCuda()
{
  /* Destroy Workspace */
  if(cudaFree(this->workspace.scratchpad) != cudaSuccess)
    std::cerr << "[WARNING] Handler Memory Failed to be Freed!\n";
    
  if(cudaStreamDestroy(this->workspace.stream) != cudaSuccess)
    std::cerr << "[WARNING] CUDA Stream Failed to be Destroyed!\n";

  if(cublasDestroy(this->ctx.cublas) != CUBLAS_STATUS_SUCCESS)
    std::cerr << "[WARNING] Cublas Failed to be Destroyed!\n";
    
  if(cublasLtDestroy(this->ctx.cublasLt) != CUBLAS_STATUS_SUCCESS)
    std::cerr << "[WARNING] CublasLT Failed to be Destroyed!\n";
    
  if(cudnnDestroy(this->ctx.cudnn) != CUDNN_STATUS_SUCCESS)
    std::cerr << "[WARNING] Cudnn Failed to be Destroyed!\n";
  
  if(cudaFree(this->backend) != cudaSuccess)
    std::cerr << "[WARNING] Backend Memory Failed to be Freed!\n";
}


void *HandleCuda::getData()
{
  return this->backend;
}

size_t HandleCuda::getCurrentOffset()
{
  return this->offset;
}

size_t HandleCuda::getCapacity()
{
  return this->capacity;
}


void *HandleCuda::getWorkspaceScratchpad()
{
  return this->workspace.scratchpad;
}

cudaStream_t HandleCuda::getWorkspaceStream()
{
  return this->workspace.stream;
}

size_t HandleCuda::getWorkspaceScratchpadSize()
{
  return this->workspace.scratchpad_size;
}


cublasHandle_t HandleCuda::getContextCublasHanler()
{
  return this->ctx.cublas;
}

cublasLtHandle_t HandleCuda::getContextCublasLtHanler()
{
  return this->ctx.cublasLt;
}

cudnnHandle_t HandleCuda::getContextCudnnHanler()
{
  return this->ctx.cudnn;
}

void HandleCuda::copyToHost(HandleCpu& handle, Math& hostMath, Math& deviceMath)
{
  auto dim = deviceMath.getLayout()->getDim();
  int rank = deviceMath.getLayout()->getRank();

  hostMath.createLayout(dim.data(), rank, deviceMath.getLayout()->getLayoutType(), deviceMath.getLayout()->getOperationType());

  handle.bind(hostMath);

  size_t count = deviceMath.getCapacity();

  void *cudaPtr = (void *)((uint8_t *)this->backend + deviceMath.getOffset());
  void *hostPtr = (void *)((uint8_t *)handle.getData() + hostMath.getOffset());

  if (cudaMemcpy(hostPtr, cudaPtr, count, cudaMemcpyDeviceToHost) != cudaSuccess)
    std::cerr << "[WARNING] CUDA Memory Failed to be copied to Host!\n";
}


void HandleCuda::bind(Math& math)
{
  /* fp16 storage: sizeof(__half) = 2 bytes per element */
  size_t len = ALIGN_128(math.getLayout()->getStrides().at(0) * math.getLayout()->getDim().at(0) * sizeof(__half));

  if (this->offset > this->capacity || len > this->capacity - this->offset)
  {
    std::cerr << "[WARNING] CUDA Handler Memory Not Enough to allocate requested Capacity, Not Allocating!\n";
    return;
  }

  math.setOffset(this->offset);
  math.setCapacity(len);

  this->offset += math.getCapacity();
}

void HandleCuda::unbind(Math& math)
{
  math.setOffset(0);
  math.setCapacity(0);
}

void HandleCuda::reset()
{
  this->offset = 0;
}