#if !defined(DL_HANDLE_CUDA_HPP)
#define DL_HANDLE_CUDA_HPP

#include <stdint.h>

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>

#define WORKSPACE_SCRATCH_SPACE 512ULL * 1024 * 1024 /* 512 MB */
#define HANDLER_CUDA_MEMORY_SPACE 2ULL * 1024 * 1024 * 1024 /* 2GB */

class Math;
class HandleCpu;

class HandleCuda
{
private:
  void *backend = NULL;
  size_t offset = 0;
  size_t capacity = 0;
  
  struct Workspace 
  {
    void *scratchpad;
    cudaStream_t stream;
    size_t scratchpad_size;
  } workspace = {0};

  struct Context
  {
    cublasHandle_t cublas;
    cublasLtHandle_t cublasLt;
    cudnnHandle_t cudnn;
  } ctx = {0};

public:
  HandleCuda(size_t capacity = HANDLER_CUDA_MEMORY_SPACE);
  ~HandleCuda();

  HandleCuda(const HandleCuda&) = delete;
  HandleCuda& operator=(const HandleCuda&) = delete;

  void *getData();
  size_t getCurrentOffset();
  size_t getCapacity();

  void *getWorkspaceScratchpad();
  cudaStream_t getWorkspaceStream();
  size_t getWorkspaceScratchpadSize();

  cublasHandle_t getContextCublasHanler();
  cublasLtHandle_t getContextCublasLtHanler();
  cudnnHandle_t getContextCudnnHanler();

  void copyToHost(HandleCpu& handle, Math& hostMath, Math& deviceMath);

  void bind(Math& math);
  void unbind(Math& math);

  void reset();
};

#endif /* DL_HANDLE_CUDA_HPP */