#include "RUNTIME/cf_time.h"
#include "RUNTIME/cf_log.h"
#include "MATH/cf_math.h"

#include <stdio.h>

int main()
{
  cf_math a, b;
  cf_math_alloc(&a, (cf_usize[8]){2, 2, 0, 0, 0, 0, 0, 0}, 2, CF_DTYPE_I32, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL);
  cf_math_alloc(&b, (cf_usize[8]){2, 2, 0, 0, 0, 0, 0, 0}, 2, CF_DTYPE_I32, CF_DEVICE_CPU, CF_MEM_DEFAULT, CF_NULL);
  cf_time t;
  cf_time_now_mono(&t);
  cf_math_rand_uniform(&a, 1, 5, (cf_u64) cf_time_as_sec(t), CF_NULL);
  cf_time_now_mono(&t);
  cf_math_rand_uniform(&b, 1, 5, (cf_u64) cf_time_as_sec(t), CF_NULL);
  cf_math_print(&a);
  printf("\n");
  cf_math_print(&b);
  cf_math_matmul(&a, &a, &b, CF_NULL);
  printf("\n");
  cf_math_print(&a);

  
  
  return 0;
}