#include "MATH/cf_math.h"
#include "RUNTIME/cf_status.h"
#include "MATH/cf_math_print.h"

#include <stdio.h>

int main(void)
{
  cf_math_cuda_context ctx = {0};
  cf_math_handle_t handler = {0};
  cf_math_metadata metadata = {0};
  cf_math a = {0}, b = {0}, c = {0};

  cf_math_cuda_context_init(&ctx, 256, 0);
  cf_math_handle_init(&handler, &ctx, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CUDA, CF_MATH_MEM_POOLED | CF_MATH_MEM_ALIGNED128, CF_MATH_HANDLE_OPT_NONE, 2048);
  cf_math_metadata_init(&metadata, (cf_usize[]) {2,2,2,2, 0, 0 ,0 ,0}, 4, CF_MATH_SHAPE_TENSOR, CF_MATH_LAYOUT_COL_MAJOR);
  cf_math_bind(&a, &handler, &metadata);
  cf_math_bind(&b, &handler, &metadata);
  cf_math_bind(&c, &handler, &metadata);

  cf_math_cpy_h2d(&a, (float[]){2,3,5,7, 2,3,5,7, 2,3,15,7, 2,13,5,7}, 16);
  cf_math_cpy_h2d(&b, (float[]){2,38,5,7, 2,3,25,7, 26,33,5,7, 2,23,5,7}, 16);
  cf_math_cpy_h2d(&c, (float[]){2,3,5,7, 21,3,25,7, 2,31,5,72, 2,3,5,7}, 16);

  cf_math_print_tensor(&a);
  printf("\n");
  cf_math_print_shape(&b);
  printf("\n");
  cf_math_print_shape(&c);
  printf("\n");

  return 0;
}
