#include "MATH/cf_math.h"
#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_time.h"
#include "MATH/cf_math_print.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
  enum { TENSOR_COUNT = 50 };

  cf_math_context context;
  cf_math_workspace workspace;
  cf_math_handle handle;
  cf_math_desc desc[TENSOR_COUNT];
  cf_math math[TENSOR_COUNT];
  int shape[TENSOR_COUNT][CF_MATH_MAX_RANK] =
  {
    {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10},
    {2, 2}, {2, 3}, {3, 3}, {3, 4}, {4, 4}, {4, 5}, {5, 5}, {5, 6}, {6, 6}, {6, 7},
    {2, 2, 2}, {2, 2, 3}, {2, 3, 3}, {3, 3, 3}, {3, 3, 4}, {3, 4, 4}, {4, 4, 4}, {4, 4, 5}, {4, 5, 5}, {5, 5, 5},
    {2, 2, 2, 2}, {2, 2, 2, 3}, {2, 2, 3, 3}, {2, 3, 3, 3}, {3, 3, 3, 3}, {3, 3, 3, 4}, {3, 3, 4, 4}, {3, 4, 4, 4}, {4, 4, 4, 4}, {4, 4, 4, 5},
    {2, 2, 2, 2, 2}, {2, 2, 2, 2, 3}, {2, 2, 2, 3, 3}, {2, 2, 3, 3, 3}, {2, 3, 3, 3, 3}, {3, 3, 3, 3, 3}, {2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 3}, {2, 2, 2, 2, 3, 3}, {2, 2, 2, 3, 3, 3}
  };
  int rank[TENSOR_COUNT] =
  {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 6, 6, 6, 6
  };
  cf_usize offset = 0;
  cf_time_point start;
  cf_time_point end;
  cf_time elapsed;

  cf_math_context_create(&context, 0, CF_MATH_DEVICE_CUDA);
  cf_math_workspace_create(&workspace, 2*1024*1024, CF_MATH_DEVICE_CUDA);
  cf_math_handle_create(&handle, &context, &workspace, 1024*1024, CF_MATH_DEVICE_CUDA);
  cudaStreamSynchronize(workspace.stream);

  for (int i = 0; i < TENSOR_COUNT; i++)
  {
    cf_math_desc_create(&desc[i], rank[i], shape[i], CF_MATH_DTYPE_F32, CF_MATH_DESC_NONE);
    cf_usize len = (cf_usize)desc[i].dim[0] * (cf_usize)desc[i].strides[0];
    float *values = (float *)malloc(len * sizeof(float));

    for (cf_usize j = 0; j < len; j++)
      values[j] = (float)(i * 1000) + (float)j;

    cudaMemcpy((cf_u8 *)handle.storage.backend + offset, values, len * sizeof(float), cudaMemcpyHostToDevice);
    free(values);

    offset += len * sizeof(float);
  }

  cf_math_handle_reset(&handle);
  cf_time_now_mono(&start);

  for (int i = 0; i < TENSOR_COUNT; i++)
    cf_math_bind(&handle, &math[i], &desc[i]);

  cf_time_now_mono(&end);
  elapsed = cf_time_elapsed(start, end);

  for (int i = 0; i < TENSOR_COUNT; i++)
    cf_math_print(&handle, &math[i]);

  printf("cuda bind only: %lld ns (%lld ms)\n", (long long)cf_time_as_ns(elapsed), (long long)cf_time_as_ms(elapsed));

  for (int i = 0; i < TENSOR_COUNT; i++)
    cf_math_desc_destroy(&desc[i]);

  cf_math_handle_destroy(&handle);
  cf_math_workspace_destroy(&workspace);
  cf_math_context_destroy(&context);
  return 0;
}
