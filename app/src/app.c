#include "MEMORY/cf_memory.h"
#include "MEMORY/cf_array.h"

#include "RUNTIME/cf_io.h"
#include "RUNTIME/cf_log.h"
#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_time.h"

#include "SECURITY/cf_aes.h"
#include "SECURITY/cf_base64.h"
#include "SECURITY/cf_hex.h"

#include "MATH/cf_math.h"
#include "MATH/cf_tensor.h"

#include "TEXT/cf_ascii.h"
#include "TEXT/cf_string.h"

#include <stdio.h>

#define CF_APP_TENSOR_ADD_LEN ((cf_usize) 16777216)

int main(void)
{
  cf_tensor a, b, cpu_out;
  cf_time_point start, end;

  CF_LOG_INFO(cf_status_as_char(cf_tensor_init(&a, (cf_usize[]){CF_APP_TENSOR_ADD_LEN, 0, 0, 0, 0, 0, 0, 0}, 1, CF_TENSOR_DOUBLE)));
  CF_LOG_INFO(cf_status_as_char(cf_tensor_init(&b, (cf_usize[]){CF_APP_TENSOR_ADD_LEN, 0, 0, 0, 0, 0, 0, 0}, 1, CF_TENSOR_DOUBLE)));
  CF_LOG_INFO(cf_status_as_char(cf_tensor_init(&cpu_out, (cf_usize[]){CF_APP_TENSOR_ADD_LEN, 0, 0, 0, 0, 0, 0, 0}, 1, CF_TENSOR_DOUBLE)));

  for (cf_usize i = 0; i < CF_APP_TENSOR_ADD_LEN; i++)
  {
    ((double *) a.data)[i] = (double) i * 0.25;
    ((double *) b.data)[i] = (double) i * 0.5;
  }

  cf_time_now_mono(&start);
  cf_status cpu_status = cf_tensor_add_cpu(&a, &b, &cpu_out);
  cf_time_now_mono(&end);

  cf_time cpu_elapsed = cf_time_elapsed(start, end);
  double cpu_seconds = (double) cf_time_as_ns(cpu_elapsed) / 1000000000.0;
  double bytes = (double) CF_APP_TENSOR_ADD_LEN * sizeof(double) * 3.0;
  double cpu_gib_per_sec = bytes / cpu_seconds / 1073741824.0;

  printf("cpu add: %s | %.6f s | %.3f GiB/s | sample %.2f %.2f\n",
         cf_status_as_char(cpu_status),
         cpu_seconds,
         cpu_gib_per_sec,
         ((double *) cpu_out.data)[0],
         ((double *) cpu_out.data)[CF_APP_TENSOR_ADD_LEN - 1]);

#ifdef CF_CUDA_AVAILABLE
  cf_tensor gpu_out;
  CF_LOG_INFO(cf_status_as_char(cf_tensor_init(&gpu_out, (cf_usize[]){CF_APP_TENSOR_ADD_LEN, 0, 0, 0, 0, 0, 0, 0}, 1, CF_TENSOR_DOUBLE)));

  cf_time_now_mono(&start);
  cf_status gpu_status = cf_tensor_add_gpu(&a, &b, &gpu_out);
  cf_time_now_mono(&end);

  cf_time gpu_elapsed = cf_time_elapsed(start, end);
  double gpu_seconds = (double) cf_time_as_ns(gpu_elapsed) / 1000000000.0;
  double gpu_gib_per_sec = bytes / gpu_seconds / 1073741824.0;

  printf("gpu add: %s | %.6f s | %.3f GiB/s | sample %.2f %.2f\n",
         cf_status_as_char(gpu_status),
         gpu_seconds,
         gpu_gib_per_sec,
         ((double *) gpu_out.data)[0],
         ((double *) gpu_out.data)[CF_APP_TENSOR_ADD_LEN - 1]);

  cf_tensor_destroy(&gpu_out);
#else
  printf("gpu add: skipped; build without CF_CUDA_AVAILABLE\n");
#endif

  cf_tensor_destroy(&a);
  cf_tensor_destroy(&b);
  cf_tensor_destroy(&cpu_out);
  return 0;
}
