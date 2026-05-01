#include "AI/cf_model.h"
#include "MATH/cf_math.h"
#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_time.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CF_BENCH_ELEM_LEN ((cf_usize)1U << 20U)
#define CF_BENCH_ELEM_ITERS 200U
#define CF_BENCH_UNARY_ITERS 100U
#define CF_BENCH_SCALAR_ITERS 200U
#define CF_BENCH_REDUCE_ITERS 100U
#define CF_BENCH_MATMUL_M 256U
#define CF_BENCH_MATMUL_K 256U
#define CF_BENCH_MATMUL_N 256U
#define CF_BENCH_MATMUL_ITERS 8U
#define CF_BENCH_DENSE_BATCH 256U
#define CF_BENCH_DENSE_IN 256U
#define CF_BENCH_DENSE_OUT 256U
#define CF_BENCH_DENSE_ITERS 12U
#define CF_BENCH_LOSS_ITERS 100U
#define CF_BENCH_CUDA_WORKSPACE_BYTES ((cf_usize)64U * 1024U * 1024U)

typedef struct cf_bench_result
{
  double elem_add_ms;
  double unary_relu_ms;
  double scalar_mul_ms;
  double reduce_mean_ms;
  double matmul_ms;
  double dense_ms;
  double loss_mse_ms;
  float checksum;
} cf_bench_result;

static double cf_bench_elapsed_ms(cf_time_point start, cf_time_point end, cf_u32 iters)
{
  cf_time elapsed = cf_time_elapsed(start, end);
  return (double)cf_time_as_ns(elapsed) / 1000000.0 / (double)iters;
}

static cf_status cf_bench_now(cf_time_point *out)
{
  cf_status status = cf_time_now_mono(out);
  if(status != CF_OK) printf("time error: %s\n", cf_status_as_char(status));
  return status;
}

static void cf_bench_fill(float *data, cf_usize len, float scale, float shift)
{
  for(cf_usize i = 0; i < len; ++i)
  {
    data[i] = shift + scale * (float)((i % 97U) + 1U);
  }
}

static cf_status cf_bench_check(cf_status status, const char *label)
{
  if(status != CF_OK) printf("%s failed: %s\n", label, cf_status_as_char(status));
  return status;
}

static void cf_bench_print_result(const char *name, const cf_bench_result *result)
{
  printf("\n%s\n", name);
  printf("  elementwise add   : %9.4f ms/iter  %.2f M elems/s\n", result->elem_add_ms, ((double)CF_BENCH_ELEM_LEN / 1000000.0) / (result->elem_add_ms / 1000.0));
  printf("  unary relu        : %9.4f ms/iter  %.2f M elems/s\n", result->unary_relu_ms, ((double)CF_BENCH_ELEM_LEN / 1000000.0) / (result->unary_relu_ms / 1000.0));
  printf("  scalar mul        : %9.4f ms/iter  %.2f M elems/s\n", result->scalar_mul_ms, ((double)CF_BENCH_ELEM_LEN / 1000000.0) / (result->scalar_mul_ms / 1000.0));
  printf("  reduce mean       : %9.4f ms/iter\n", result->reduce_mean_ms);
  printf("  matmul 256x256    : %9.4f ms/iter  %.2f GFLOP/s\n", result->matmul_ms, (2.0 * (double)CF_BENCH_MATMUL_M * (double)CF_BENCH_MATMUL_K * (double)CF_BENCH_MATMUL_N / 1000000000.0) / (result->matmul_ms / 1000.0));
  printf("  dense forward     : %9.4f ms/iter  %.2f GFLOP/s\n", result->dense_ms, (2.0 * (double)CF_BENCH_DENSE_BATCH * (double)CF_BENCH_DENSE_IN * (double)CF_BENCH_DENSE_OUT / 1000000000.0) / (result->dense_ms / 1000.0));
  printf("  mse loss          : %9.4f ms/iter\n", result->loss_mse_ms);
  printf("  checksum          : %.6f\n", result->checksum);
}

static cf_status cf_bench_run(cf_math_device device, cf_math_cuda_context *ctx, const char *name, cf_bench_result *result)
{
  cf_math_handle_t elem_handler = {0};
  cf_math_handle_t matmul_handler = {0};
  cf_math_handle_t parameter_handler = {0};
  cf_math_handle_t activation_handler = {0};
  cf_math_metadata elem_meta = {0};
  cf_math_metadata scalar_meta = {0};
  cf_math_metadata a_meta = {0};
  cf_math_metadata b_meta = {0};
  cf_math_metadata c_meta = {0};
  cf_math_metadata dense_input_meta = {0};
  cf_math x = {0};
  cf_math y = {0};
  cf_math reduce_out = {0};
  cf_math ma = {0};
  cf_math mb = {0};
  cf_math mc = {0};
  cf_math dense_input = {0};
  cf_ai_dense dense = {0};
  cf_math loss_target = {0};
  cf_math loss_out = {0};
  cf_usize elem_dims[CF_MATH_MAX_RANK] = {CF_BENCH_ELEM_LEN};
  cf_usize scalar_dims[CF_MATH_MAX_RANK] = {1};
  cf_usize a_dims[CF_MATH_MAX_RANK] = {CF_BENCH_MATMUL_M, CF_BENCH_MATMUL_K};
  cf_usize b_dims[CF_MATH_MAX_RANK] = {CF_BENCH_MATMUL_K, CF_BENCH_MATMUL_N};
  cf_usize c_dims[CF_MATH_MAX_RANK] = {CF_BENCH_MATMUL_M, CF_BENCH_MATMUL_N};
  cf_usize dense_input_dims[CF_MATH_MAX_RANK] = {CF_BENCH_DENSE_BATCH, CF_BENCH_DENSE_IN};
  float *host_x = CF_NULL;
  float *host_y = CF_NULL;
  float *host_a = CF_NULL;
  float *host_b = CF_NULL;
  float *host_dense_input = CF_NULL;
  float *host_dense_weight = CF_NULL;
  float *host_dense_bias = CF_NULL;
  float *host_loss_target = CF_NULL;
  float checksum_data[4] = {0};
  cf_time_point start = {0};
  cf_time_point end = {0};
  cf_status status = CF_OK;
  cf_usize mat_a_len = CF_BENCH_MATMUL_M * CF_BENCH_MATMUL_K;
  cf_usize mat_b_len = CF_BENCH_MATMUL_K * CF_BENCH_MATMUL_N;
  cf_usize mat_c_len = CF_BENCH_MATMUL_M * CF_BENCH_MATMUL_N;
  cf_usize dense_weight_len = CF_BENCH_DENSE_IN * CF_BENCH_DENSE_OUT;
  cf_usize dense_input_len = CF_BENCH_DENSE_BATCH * CF_BENCH_DENSE_IN;
  cf_usize dense_output_len = CF_BENCH_DENSE_BATCH * CF_BENCH_DENSE_OUT;
  cf_usize elem_capacity = (CF_BENCH_ELEM_LEN * 3U + 4U) * sizeof(float);
  cf_usize matmul_capacity = (mat_a_len + mat_b_len + mat_c_len) * sizeof(float);
  cf_usize parameter_capacity = (dense_weight_len + CF_BENCH_DENSE_OUT) * sizeof(float);
  cf_usize activation_capacity = (dense_input_len + dense_output_len + dense_output_len + 4U) * sizeof(float);

  memset(result, 0, sizeof(*result));
  printf("\nstarting %s benchmark...\n", name);

  host_x = (float *)malloc(CF_BENCH_ELEM_LEN * sizeof(float));
  host_y = (float *)malloc(CF_BENCH_ELEM_LEN * sizeof(float));
  host_a = (float *)malloc(mat_a_len * sizeof(float));
  host_b = (float *)malloc(mat_b_len * sizeof(float));
  host_dense_input = (float *)malloc(dense_input_len * sizeof(float));
  host_dense_weight = (float *)malloc(dense_weight_len * sizeof(float));
  host_dense_bias = (float *)malloc(CF_BENCH_DENSE_OUT * sizeof(float));
  host_loss_target = (float *)malloc(dense_output_len * sizeof(float));
  if(host_x == CF_NULL || host_y == CF_NULL || host_a == CF_NULL || host_b == CF_NULL || host_dense_input == CF_NULL || host_dense_weight == CF_NULL || host_dense_bias == CF_NULL || host_loss_target == CF_NULL)
  {
    status = CF_ERR_OOM;
    goto cleanup;
  }

  cf_bench_fill(host_x, CF_BENCH_ELEM_LEN, 0.001f, -0.05f);
  cf_bench_fill(host_y, CF_BENCH_ELEM_LEN, 0.002f, 0.1f);
  cf_bench_fill(host_a, mat_a_len, 0.001f, 0.0f);
  cf_bench_fill(host_b, mat_b_len, 0.0015f, 0.0f);
  cf_bench_fill(host_dense_input, dense_input_len, 0.002f, -0.1f);
  cf_bench_fill(host_dense_weight, dense_weight_len, 0.001f, -0.02f);
  cf_bench_fill(host_dense_bias, CF_BENCH_DENSE_OUT, 0.001f, 0.01f);
  cf_bench_fill(host_loss_target, dense_output_len, 0.001f, 0.0f);

  status = cf_bench_check(cf_math_handle_init(&elem_handler, ctx, CF_MATH_DTYPE_F32, device, CF_MATH_MEM_POOLED | CF_MATH_MEM_ALIGNED128, CF_MATH_HANDLE_OPT_ELEMENTWISE | CF_MATH_HANDLE_OPT_REDUCTION, elem_capacity), "elem handler");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_handle_init(&matmul_handler, ctx, CF_MATH_DTYPE_F32, device, CF_MATH_MEM_POOLED | CF_MATH_MEM_ALIGNED128, CF_MATH_HANDLE_OPT_MATMUL, matmul_capacity), "matmul handler");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_handle_init(&parameter_handler, ctx, CF_MATH_DTYPE_F32, device, CF_MATH_MEM_POOLED | CF_MATH_MEM_ALIGNED128, CF_MATH_HANDLE_OPT_MATMUL, parameter_capacity), "parameter handler");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_handle_init(&activation_handler, ctx, CF_MATH_DTYPE_F32, device, CF_MATH_MEM_POOLED | CF_MATH_MEM_ALIGNED128, CF_MATH_HANDLE_OPT_MATMUL | CF_MATH_HANDLE_OPT_ELEMENTWISE | CF_MATH_HANDLE_OPT_REDUCTION, activation_capacity), "activation handler");
  if(status != CF_OK) goto cleanup;

  status = cf_bench_check(cf_math_metadata_init(&elem_meta, elem_dims, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR), "elem metadata");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_metadata_init(&scalar_meta, scalar_dims, 1, CF_MATH_SHAPE_SCALAR, CF_MATH_LAYOUT_ROW_MAJOR), "scalar metadata");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_metadata_init(&a_meta, a_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR), "a metadata");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_metadata_init(&b_meta, b_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR), "b metadata");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_metadata_init(&c_meta, c_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR), "c metadata");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_metadata_init(&dense_input_meta, dense_input_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR), "dense input metadata");
  if(status != CF_OK) goto cleanup;

  status = cf_bench_check(cf_math_bind(&x, &elem_handler, &elem_meta), "bind x");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_bind(&y, &elem_handler, &elem_meta), "bind y");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_bind(&reduce_out, &elem_handler, &scalar_meta), "bind reduce");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_bind(&ma, &matmul_handler, &a_meta), "bind ma");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_bind(&mb, &matmul_handler, &b_meta), "bind mb");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_bind(&mc, &matmul_handler, &c_meta), "bind mc");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_bind(&dense_input, &activation_handler, &dense_input_meta), "bind dense input");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_ai_dense_init(&dense, &parameter_handler, &activation_handler, CF_BENCH_DENSE_BATCH, CF_BENCH_DENSE_IN, CF_BENCH_DENSE_OUT, CF_AI_ACT_RELU), "dense init");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_bind(&loss_target, &activation_handler, &dense.output_meta), "bind loss target");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_bind(&loss_out, &activation_handler, &scalar_meta), "bind loss out");
  if(status != CF_OK) goto cleanup;

  status = cf_bench_check(cf_math_cpy_h2d(&x, host_x, CF_BENCH_ELEM_LEN), "copy x");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_cpy_h2d(&y, host_y, CF_BENCH_ELEM_LEN), "copy y");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_cpy_h2d(&ma, host_a, mat_a_len), "copy ma");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_cpy_h2d(&mb, host_b, mat_b_len), "copy mb");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_cpy_h2d(&dense_input, host_dense_input, dense_input_len), "copy dense input");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_cpy_h2d(&dense.weights, host_dense_weight, dense_weight_len), "copy dense weights");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_cpy_h2d(&dense.bias, host_dense_bias, CF_BENCH_DENSE_OUT), "copy dense bias");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_cpy_h2d(&loss_target, host_loss_target, dense_output_len), "copy loss target");
  if(status != CF_OK) goto cleanup;

  status = cf_bench_check(cf_math_op(CF_MATH_OP_ADD, &x, &y), "warm elem add");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_now(&start);
  if(status != CF_OK) goto cleanup;
  for(cf_u32 i = 0; i < CF_BENCH_ELEM_ITERS; ++i)
  {
    status = cf_math_op(CF_MATH_OP_ADD, &x, &y);
    if(status != CF_OK) goto cleanup;
  }
  status = cf_bench_now(&end);
  if(status != CF_OK) goto cleanup;
  result->elem_add_ms = cf_bench_elapsed_ms(start, end, CF_BENCH_ELEM_ITERS);

  status = cf_bench_check(cf_math_cpy_h2d(&x, host_x, CF_BENCH_ELEM_LEN), "reset x");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_check(cf_math_unary(CF_MATH_OP_RELU, &x), "warm relu");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_now(&start);
  if(status != CF_OK) goto cleanup;
  for(cf_u32 i = 0; i < CF_BENCH_UNARY_ITERS; ++i)
  {
    status = cf_math_unary(CF_MATH_OP_RELU, &x);
    if(status != CF_OK) goto cleanup;
  }
  status = cf_bench_now(&end);
  if(status != CF_OK) goto cleanup;
  result->unary_relu_ms = cf_bench_elapsed_ms(start, end, CF_BENCH_UNARY_ITERS);

  status = cf_bench_check(cf_math_scalar(CF_MATH_OP_MUL, &x, 1.000001), "warm scalar");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_now(&start);
  if(status != CF_OK) goto cleanup;
  for(cf_u32 i = 0; i < CF_BENCH_SCALAR_ITERS; ++i)
  {
    status = cf_math_scalar(CF_MATH_OP_MUL, &x, 1.000001);
    if(status != CF_OK) goto cleanup;
  }
  status = cf_bench_now(&end);
  if(status != CF_OK) goto cleanup;
  result->scalar_mul_ms = cf_bench_elapsed_ms(start, end, CF_BENCH_SCALAR_ITERS);

  status = cf_bench_check(cf_math_reduce_mean(&reduce_out, &x), "warm reduce");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_now(&start);
  if(status != CF_OK) goto cleanup;
  for(cf_u32 i = 0; i < CF_BENCH_REDUCE_ITERS; ++i)
  {
    status = cf_math_reduce_mean(&reduce_out, &x);
    if(status != CF_OK) goto cleanup;
  }
  status = cf_bench_now(&end);
  if(status != CF_OK) goto cleanup;
  result->reduce_mean_ms = cf_bench_elapsed_ms(start, end, CF_BENCH_REDUCE_ITERS);

  status = cf_bench_check(cf_math_matmul(&mc, &ma, &mb), "warm matmul");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_now(&start);
  if(status != CF_OK) goto cleanup;
  for(cf_u32 i = 0; i < CF_BENCH_MATMUL_ITERS; ++i)
  {
    status = cf_math_matmul(&mc, &ma, &mb);
    if(status != CF_OK) goto cleanup;
  }
  status = cf_bench_now(&end);
  if(status != CF_OK) goto cleanup;
  result->matmul_ms = cf_bench_elapsed_ms(start, end, CF_BENCH_MATMUL_ITERS);

  status = cf_bench_check(cf_ai_dense_forward(&dense, &dense_input), "warm dense");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_now(&start);
  if(status != CF_OK) goto cleanup;
  for(cf_u32 i = 0; i < CF_BENCH_DENSE_ITERS; ++i)
  {
    status = cf_ai_dense_forward(&dense, &dense_input);
    if(status != CF_OK) goto cleanup;
  }
  status = cf_bench_now(&end);
  if(status != CF_OK) goto cleanup;
  result->dense_ms = cf_bench_elapsed_ms(start, end, CF_BENCH_DENSE_ITERS);

  status = cf_bench_check(cf_ai_loss_forward(CF_AI_LOSS_MSE, &loss_out, &dense.output, &loss_target), "warm loss");
  if(status != CF_OK) goto cleanup;
  status = cf_bench_now(&start);
  if(status != CF_OK) goto cleanup;
  for(cf_u32 i = 0; i < CF_BENCH_LOSS_ITERS; ++i)
  {
    status = cf_ai_loss_forward(CF_AI_LOSS_MSE, &loss_out, &dense.output, &loss_target);
    if(status != CF_OK) goto cleanup;
  }
  status = cf_bench_now(&end);
  if(status != CF_OK) goto cleanup;
  result->loss_mse_ms = cf_bench_elapsed_ms(start, end, CF_BENCH_LOSS_ITERS);

  status = cf_bench_check(cf_math_cpy_d2h(&reduce_out, checksum_data, 1), "read reduce checksum");
  if(status != CF_OK) goto cleanup;
  result->checksum += checksum_data[0];
  status = cf_bench_check(cf_math_cpy_d2h(&loss_out, checksum_data, 1), "read loss checksum");
  if(status != CF_OK) goto cleanup;
  result->checksum += checksum_data[0];
  status = cf_bench_check(cf_math_cpy_d2h(&mc, checksum_data, 4), "read matmul checksum");
  if(status != CF_OK) goto cleanup;
  result->checksum += checksum_data[0] + checksum_data[1] + checksum_data[2] + checksum_data[3];
  status = cf_bench_check(cf_math_cpy_d2h(&dense.output, checksum_data, 4), "read dense checksum");
  if(status != CF_OK) goto cleanup;
  result->checksum += checksum_data[0] + checksum_data[1] + checksum_data[2] + checksum_data[3];

cleanup:
  if(loss_out.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&loss_out));
  if(loss_target.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&loss_target));
  if(dense.output.handler != CF_NULL || dense.bias.handler != CF_NULL || dense.weights.handler != CF_NULL) CF_UNUSED(cf_ai_dense_destroy(&dense));
  if(dense_input.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&dense_input));
  if(mc.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&mc));
  if(mb.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&mb));
  if(ma.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&ma));
  if(reduce_out.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&reduce_out));
  if(y.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&y));
  if(x.handler != CF_NULL) CF_UNUSED(cf_math_unbind(&x));
  CF_UNUSED(cf_math_handle_destroy(&activation_handler));
  CF_UNUSED(cf_math_handle_destroy(&parameter_handler));
  CF_UNUSED(cf_math_handle_destroy(&matmul_handler));
  CF_UNUSED(cf_math_handle_destroy(&elem_handler));
  free(host_loss_target);
  free(host_dense_bias);
  free(host_dense_weight);
  free(host_dense_input);
  free(host_b);
  free(host_a);
  free(host_y);
  free(host_x);

  if(status == CF_OK) cf_bench_print_result(name, result);
  return status;
}

#if defined(CF_CUDA_AVAILABLE)
static void cf_bench_print_speedup(const cf_bench_result *cpu, const cf_bench_result *gpu)
{
  printf("\nGPU speedup vs CPU\n");
  printf("  elementwise add   : %.2fx\n", cpu->elem_add_ms / gpu->elem_add_ms);
  printf("  unary relu        : %.2fx\n", cpu->unary_relu_ms / gpu->unary_relu_ms);
  printf("  scalar mul        : %.2fx\n", cpu->scalar_mul_ms / gpu->scalar_mul_ms);
  printf("  reduce mean       : %.2fx\n", cpu->reduce_mean_ms / gpu->reduce_mean_ms);
  printf("  matmul            : %.2fx\n", cpu->matmul_ms / gpu->matmul_ms);
  printf("  dense forward     : %.2fx\n", cpu->dense_ms / gpu->dense_ms);
  printf("  mse loss          : %.2fx\n", cpu->loss_mse_ms / gpu->loss_mse_ms);
}
#endif

int main(void)
{
  cf_bench_result cpu = {0};
  cf_status status = CF_OK;

  printf("Cypher math/AI benchmark\n");
  printf("  element count : %u\n", (unsigned int)CF_BENCH_ELEM_LEN);
  printf("  matmul        : %ux%u @ %ux%u\n", (unsigned int)CF_BENCH_MATMUL_M, (unsigned int)CF_BENCH_MATMUL_K, (unsigned int)CF_BENCH_MATMUL_K, (unsigned int)CF_BENCH_MATMUL_N);
  printf("  dense         : batch=%u in=%u out=%u\n", (unsigned int)CF_BENCH_DENSE_BATCH, (unsigned int)CF_BENCH_DENSE_IN, (unsigned int)CF_BENCH_DENSE_OUT);

  status = cf_bench_run(CF_MATH_DEVICE_CPU, CF_NULL, "CPU", &cpu);
  if(status != CF_OK) return (int)status;

#if defined(CF_CUDA_AVAILABLE)
  {
    cf_bench_result gpu = {0};
    cf_math_cuda_context ctx = {0};
    status = cf_math_cuda_context_init(&ctx, CF_BENCH_CUDA_WORKSPACE_BYTES, 0);
    if(status == CF_OK)
    {
      status = cf_bench_run(CF_MATH_DEVICE_CUDA, &ctx, "CUDA", &gpu);
      CF_UNUSED(cf_math_cuda_context_destroy(&ctx));
      if(status != CF_OK) return (int)status;
      cf_bench_print_speedup(&cpu, &gpu);
    }
    else
    {
      printf("\nCUDA skipped: cf_math_cuda_context_init failed with %s\n", cf_status_as_char(status));
    }
  }
#else
  printf("\nCUDA skipped: build does not define CF_CUDA_AVAILABLE.\n");
#endif

  return 0;
}
