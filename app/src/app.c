#include "AI/cf_gradient.h"
#include "AI/cf_model.h"
#include "MATH/cf_math.h"
#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_time.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CF_GPU_BENCH_SIZE_COUNT 4U
#define CF_GPU_BENCH_WARMUP 2U
#define CF_GPU_BENCH_CUDA_WORKSPACE_BYTES ((cf_usize)256U * 1024U * 1024U)

#if defined(CF_CUDA_AVAILABLE)
typedef struct cf_gpu_bench_case
{
  cf_usize n;
  cf_u32 matmul_iters;
  cf_u32 dense_iters;
  cf_u32 forward_loss_iters;
  double matmul_ms;
  double dense_ms;
  double forward_loss_ms;
  double forward_backward_ms;
  cf_status backward_status;
  float checksum;
} cf_gpu_bench_case;

static const cf_gpu_bench_case CF_GPU_BENCH_CASES[CF_GPU_BENCH_SIZE_COUNT] = {
  {512U, 30U, 30U, 30U, 0.0, 0.0, 0.0, 0.0, CF_OK, 0.0f},
  {1024U, 20U, 20U, 20U, 0.0, 0.0, 0.0, 0.0, CF_OK, 0.0f},
  {2048U, 10U, 10U, 10U, 0.0, 0.0, 0.0, 0.0, CF_OK, 0.0f},
  {4096U, 4U, 4U, 4U, 0.0, 0.0, 0.0, 0.0, CF_OK, 0.0f},
};

static double cf_gpu_bench_elapsed_ms(cf_time_point start, cf_time_point end, cf_u32 iters)
{
  cf_time elapsed = cf_time_elapsed(start, end);
  return (double)cf_time_as_ns(elapsed) / 1000000.0 / (double)iters;
}

static cf_status cf_gpu_bench_now(cf_time_point *out)
{
  cf_status status = cf_time_now_mono(out);
  if(status != CF_OK) printf("time error: %s\n", cf_status_as_char(status));
  return status;
}

static cf_status cf_gpu_bench_check(cf_status status, const char *label)
{
  if(status != CF_OK) printf("%s failed: %s\n", label, cf_status_as_char(status));
  return status;
}

static void cf_gpu_bench_fill(float *data, cf_usize len, float scale, float shift)
{
  for(cf_usize i = 0; i < len; ++i)
  {
    data[i] = shift + scale * (float)((i % 251U) + 1U);
  }
}

static double cf_gpu_bench_gflops(cf_usize n, double ms)
{
  if(ms <= 0.0) return 0.0;
  return (2.0 * (double)n * (double)n * (double)n / 1000000000.0) / (ms / 1000.0);
}

static void cf_gpu_bench_print_header(void)
{
  printf("\nCUDA AI speed benchmark\n");
  printf("  dtype      : f32\n");
  printf("  sizes      : 512x512, 1024x1024, 2048x2048, 4096x4096\n");
  printf("  dense      : output = relu(input @ weights + bias), batch = size\n");
  printf("  loss       : MSE(output, target)\n");
  printf("  backward   : current cf_gradient API returns unsupported\n\n");
  printf("%10s  %14s  %14s  %14s  %18s  %10s\n", "size", "batch matmul", "dense+bias+act", "forward+loss", "forward+backward", "checksum");
  printf("%10s  %14s  %14s  %14s  %18s  %10s\n", "", "ms / GFLOP/s", "ms / GFLOP/s", "ms / GFLOP/s", "status", "");
}

static void cf_gpu_bench_print_case(const cf_gpu_bench_case *bench)
{
  char backward[32];
  if(bench->backward_status == CF_OK)
    snprintf(backward, sizeof(backward), "%.4f ms", bench->forward_backward_ms);
  else
    snprintf(backward, sizeof(backward), "%s", cf_status_as_char(bench->backward_status));

  printf("%4zux%-5zu  %6.3f/%7.1f  %6.3f/%7.1f  %6.3f/%7.1f  %18s  %10.5f\n",
         bench->n,
         bench->n,
         bench->matmul_ms,
         cf_gpu_bench_gflops(bench->n, bench->matmul_ms),
         bench->dense_ms,
         cf_gpu_bench_gflops(bench->n, bench->dense_ms),
         bench->forward_loss_ms,
         cf_gpu_bench_gflops(bench->n, bench->forward_loss_ms),
         backward,
         bench->checksum);
}

static cf_status cf_gpu_bench_alloc_and_copy(cf_math *view, float **host, cf_usize len, float scale, float shift, const char *label)
{
  cf_status status = CF_OK;

  *host = (float *)malloc(len * sizeof(float));
  if(*host == CF_NULL) return cf_gpu_bench_check(CF_ERR_OOM, label);

  cf_gpu_bench_fill(*host, len, scale, shift);
  status = cf_gpu_bench_check(cf_math_cpy_h2d(view, *host, len), label);
  if(status == CF_OK) status = cf_gpu_bench_check(cf_math_handle_sync(view->handler), label);
  free(*host);
  *host = CF_NULL;
  return status;
}

static cf_status cf_gpu_bench_time_matmul(cf_gpu_bench_case *bench, cf_math *out, const cf_math *a, const cf_math *b)
{
  cf_time_point start = {0};
  cf_time_point end = {0};
  cf_status status = CF_OK;

  for(cf_u32 i = 0; i < CF_GPU_BENCH_WARMUP; ++i)
  {
    status = cf_gpu_bench_check(cf_math_matmul(out, a, b), "warm batch matmul");
    if(status != CF_OK) return status;
  }
  status = cf_gpu_bench_check(cf_math_handle_sync(out->handler), "sync warm batch matmul");
  if(status != CF_OK) return status;

  status = cf_gpu_bench_now(&start);
  if(status != CF_OK) return status;
  for(cf_u32 i = 0; i < bench->matmul_iters; ++i)
  {
    status = cf_math_matmul(out, a, b);
    if(status != CF_OK) return cf_gpu_bench_check(status, "batch matmul");
  }
  status = cf_gpu_bench_check(cf_math_handle_sync(out->handler), "sync batch matmul");
  if(status != CF_OK) return status;
  status = cf_gpu_bench_now(&end);
  if(status != CF_OK) return status;

  bench->matmul_ms = cf_gpu_bench_elapsed_ms(start, end, bench->matmul_iters);
  return CF_OK;
}

static cf_status cf_gpu_bench_time_dense(cf_gpu_bench_case *bench, cf_ai_dense *dense, const cf_math *input)
{
  cf_time_point start = {0};
  cf_time_point end = {0};
  cf_status status = CF_OK;

  for(cf_u32 i = 0; i < CF_GPU_BENCH_WARMUP; ++i)
  {
    status = cf_gpu_bench_check(cf_ai_dense_forward(dense, input), "warm dense+bias+activation");
    if(status != CF_OK) return status;
  }
  status = cf_gpu_bench_check(cf_math_handle_sync(dense->output.handler), "sync warm dense+bias+activation");
  if(status != CF_OK) return status;

  status = cf_gpu_bench_now(&start);
  if(status != CF_OK) return status;
  for(cf_u32 i = 0; i < bench->dense_iters; ++i)
  {
    status = cf_ai_dense_forward(dense, input);
    if(status != CF_OK) return cf_gpu_bench_check(status, "dense+bias+activation");
  }
  status = cf_gpu_bench_check(cf_math_handle_sync(dense->output.handler), "sync dense+bias+activation");
  if(status != CF_OK) return status;
  status = cf_gpu_bench_now(&end);
  if(status != CF_OK) return status;

  bench->dense_ms = cf_gpu_bench_elapsed_ms(start, end, bench->dense_iters);
  return CF_OK;
}

static cf_status cf_gpu_bench_time_forward_loss(cf_gpu_bench_case *bench, cf_ai_dense *dense, const cf_math *input, cf_math *loss, const cf_math *target)
{
  cf_time_point start = {0};
  cf_time_point end = {0};
  cf_status status = CF_OK;

  for(cf_u32 i = 0; i < CF_GPU_BENCH_WARMUP; ++i)
  {
    status = cf_gpu_bench_check(cf_ai_dense_forward(dense, input), "warm forward");
    if(status != CF_OK) return status;
    status = cf_gpu_bench_check(cf_ai_loss_forward(CF_AI_LOSS_MSE, loss, &dense->output, target), "warm loss");
    if(status != CF_OK) return status;
  }
  status = cf_gpu_bench_check(cf_math_handle_sync(loss->handler), "sync warm forward+loss");
  if(status != CF_OK) return status;

  status = cf_gpu_bench_now(&start);
  if(status != CF_OK) return status;
  for(cf_u32 i = 0; i < bench->forward_loss_iters; ++i)
  {
    status = cf_ai_dense_forward(dense, input);
    if(status != CF_OK) return cf_gpu_bench_check(status, "forward");
    status = cf_ai_loss_forward(CF_AI_LOSS_MSE, loss, &dense->output, target);
    if(status != CF_OK) return cf_gpu_bench_check(status, "loss");
  }
  status = cf_gpu_bench_check(cf_math_handle_sync(loss->handler), "sync forward+loss");
  if(status != CF_OK) return status;
  status = cf_gpu_bench_now(&end);
  if(status != CF_OK) return status;

  bench->forward_loss_ms = cf_gpu_bench_elapsed_ms(start, end, bench->forward_loss_iters);
  return CF_OK;
}

static cf_status cf_gpu_bench_probe_backward(cf_gpu_bench_case *bench, cf_ai_dense *dense, const cf_math *input, cf_math *grad_prediction, const cf_math *target)
{
  cf_status status = cf_ai_loss_backward(CF_AI_LOSS_MSE, grad_prediction, &dense->output, target);
  if(status != CF_OK)
  {
    bench->backward_status = status;
    return CF_OK;
  }

  status = cf_ai_dense_backward(dense, input, grad_prediction);
  bench->backward_status = status;
  return CF_OK;
}

static cf_status cf_gpu_bench_run_case(cf_math_cuda_context *ctx, cf_gpu_bench_case *bench)
{
  cf_math_handle_t matmul_handler = {0};
  cf_math_handle_t parameter_handler = {0};
  cf_math_handle_t activation_handler = {0};
  cf_math_metadata matrix_meta = {0};
  cf_math_metadata scalar_meta = {0};
  cf_math a = {0};
  cf_math b = {0};
  cf_math c = {0};
  cf_math input = {0};
  cf_math target = {0};
  cf_math loss = {0};
  cf_math grad_prediction = {0};
  cf_ai_dense dense = {0};
  cf_usize matrix_dims[CF_MATH_MAX_RANK] = {0};
  cf_usize scalar_dims[CF_MATH_MAX_RANK] = {1U};
  cf_usize matrix_len = bench->n * bench->n;
  cf_usize matmul_capacity = matrix_len * 3U * sizeof(float);
  cf_usize parameter_capacity = (matrix_len + bench->n) * sizeof(float);
  cf_usize activation_capacity = (matrix_len * 4U + 8U) * sizeof(float);
  float *host = CF_NULL;
  float checksum_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  cf_status status = CF_OK;

  matrix_dims[0] = bench->n;
  matrix_dims[1] = bench->n;
  bench->backward_status = CF_ERR_UNSUPPORTED;

  status = cf_gpu_bench_check(cf_math_handle_init(&matmul_handler, ctx, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CUDA, CF_MATH_MEM_POOLED | CF_MATH_MEM_ALIGNED128, CF_MATH_HANDLE_OPT_MATMUL, matmul_capacity), "matmul handler");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_check(cf_math_handle_init(&parameter_handler, ctx, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CUDA, CF_MATH_MEM_POOLED | CF_MATH_MEM_ALIGNED128, CF_MATH_HANDLE_OPT_MATMUL, parameter_capacity), "parameter handler");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_check(cf_math_handle_init(&activation_handler, ctx, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CUDA, CF_MATH_MEM_POOLED | CF_MATH_MEM_ALIGNED128, CF_MATH_HANDLE_OPT_MATMUL | CF_MATH_HANDLE_OPT_ELEMENTWISE | CF_MATH_HANDLE_OPT_REDUCTION, activation_capacity), "activation handler");
  if(status != CF_OK) goto cleanup;

  status = cf_gpu_bench_check(cf_math_metadata_init(&matrix_meta, matrix_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR), "matrix metadata");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_check(cf_math_metadata_init(&scalar_meta, scalar_dims, 1, CF_MATH_SHAPE_SCALAR, CF_MATH_LAYOUT_ROW_MAJOR), "scalar metadata");
  if(status != CF_OK) goto cleanup;

  status = cf_gpu_bench_check(cf_math_bind(&a, &matmul_handler, &matrix_meta), "bind matmul a");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_check(cf_math_bind(&b, &matmul_handler, &matrix_meta), "bind matmul b");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_check(cf_math_bind(&c, &matmul_handler, &matrix_meta), "bind matmul c");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_check(cf_math_bind(&input, &activation_handler, &matrix_meta), "bind dense input");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_check(cf_ai_dense_init(&dense, &parameter_handler, &activation_handler, bench->n, bench->n, bench->n, CF_AI_ACT_RELU), "dense init");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_check(cf_math_bind(&target, &activation_handler, &matrix_meta), "bind target");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_check(cf_math_bind(&grad_prediction, &activation_handler, &matrix_meta), "bind grad prediction");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_check(cf_math_bind(&loss, &activation_handler, &scalar_meta), "bind loss");
  if(status != CF_OK) goto cleanup;

  status = cf_gpu_bench_alloc_and_copy(&a, &host, matrix_len, 0.0001f, -0.02f, "copy matmul a");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_alloc_and_copy(&b, &host, matrix_len, 0.00015f, 0.01f, "copy matmul b");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_alloc_and_copy(&input, &host, matrix_len, 0.00012f, -0.04f, "copy dense input");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_alloc_and_copy(&dense.weights, &host, matrix_len, 0.00008f, -0.01f, "copy dense weights");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_alloc_and_copy(&dense.bias, &host, bench->n, 0.0001f, 0.02f, "copy dense bias");
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_alloc_and_copy(&target, &host, matrix_len, 0.00005f, 0.0f, "copy target");
  if(status != CF_OK) goto cleanup;

  status = cf_gpu_bench_time_matmul(bench, &c, &a, &b);
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_time_dense(bench, &dense, &input);
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_time_forward_loss(bench, &dense, &input, &loss, &target);
  if(status != CF_OK) goto cleanup;
  status = cf_gpu_bench_probe_backward(bench, &dense, &input, &grad_prediction, &target);
  if(status != CF_OK) goto cleanup;

  status = cf_gpu_bench_check(cf_math_cpy_d2h(&c, checksum_data, 4U), "read matmul checksum");
  if(status != CF_OK) goto cleanup;
  bench->checksum += checksum_data[0] + checksum_data[1] + checksum_data[2] + checksum_data[3];
  status = cf_gpu_bench_check(cf_math_cpy_d2h(&dense.output, checksum_data, 4U), "read dense checksum");
  if(status != CF_OK) goto cleanup;
  bench->checksum += checksum_data[0] + checksum_data[1] + checksum_data[2] + checksum_data[3];
  status = cf_gpu_bench_check(cf_math_cpy_d2h(&loss, checksum_data, 1U), "read loss checksum");
  if(status != CF_OK) goto cleanup;
  bench->checksum += checksum_data[0];

cleanup:
  free(host);
  if(loss.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&loss));
  if(grad_prediction.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&grad_prediction));
  if(target.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&target));
  if(dense.output.handler != CF_NULL || dense.bias.handler != CF_NULL || dense.weights.handler != CF_NULL) CF_UNUSED(cf_ai_dense_destroy(&dense));
  if(input.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&input));
  if(c.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&c));
  if(b.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&b));
  if(a.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&a));
  CF_UNUSED(cf_math_handle_destroy(&activation_handler));
  CF_UNUSED(cf_math_handle_destroy(&parameter_handler));
  CF_UNUSED(cf_math_handle_destroy(&matmul_handler));
  return status;
}
#endif

int main(void)
{
#if defined(CF_CUDA_AVAILABLE)
  cf_math_cuda_context ctx = {0};
  cf_status status = CF_OK;

  cf_gpu_bench_print_header();

  status = cf_math_cuda_context_init(&ctx, CF_GPU_BENCH_CUDA_WORKSPACE_BYTES, 0);
  if(status != CF_OK)
  {
    printf("CUDA init failed: %s\n", cf_status_as_char(status));
    return (int)status;
  }

  for(cf_u32 i = 0; i < CF_GPU_BENCH_SIZE_COUNT; ++i)
  {
    cf_gpu_bench_case bench = CF_GPU_BENCH_CASES[i];
    status = cf_gpu_bench_run_case(&ctx, &bench);
    if(status != CF_OK)
    {
      CF_UNUSED(cf_math_cuda_context_destroy(&ctx));
      return (int)status;
    }
    cf_gpu_bench_print_case(&bench);
  }

  status = cf_math_cuda_context_destroy(&ctx);
  if(status != CF_OK)
  {
    printf("CUDA cleanup failed: %s\n", cf_status_as_char(status));
    return (int)status;
  }

  return 0;
#else
  printf("CUDA benchmark skipped: build does not define CF_CUDA_AVAILABLE.\n");
  return 0;
#endif
}
