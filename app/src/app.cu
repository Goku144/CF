#include "MATH/cf_math.h"
#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_time.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CF_APP_DEVICE
#define CF_APP_DEVICE 0
#endif

#if CF_APP_DEVICE != 0 && CF_APP_DEVICE != 1
#error "CF_APP_DEVICE must be 0 for CPU or 1 for CUDA"
#endif

#define CF_APP_CUDA 1

typedef struct app_dtype_case
{
  cf_math_dtype dtype;
  const char *name;
  cf_usize elem_size;
  double tolerance;
} app_dtype_case;

static cf_usize app_round_up(cf_usize n, cf_usize d)
{
  return ((n + d - 1) / d) * d;
}

static void *app_math_data(cf_math_handle *handle, const cf_math *math)
{
  return (void *)((cf_u8 *)handle->storage.backend + math->byte_offset);
}

static float app_f16_to_f32(cf_u16 x)
{
  __half h;
  memcpy(&h, &x, sizeof h);
  return __half2float(h);
}

static cf_u16 app_f32_to_f16(float x)
{
  __half h = __float2half_rn(x);
  cf_u16 y;
  memcpy(&y, &h, sizeof y);
  return y;
}

static float app_bf16_to_f32(cf_u16 x)
{
  unsigned int bits = (unsigned int)x << 16;
  float y;
  memcpy(&y, &bits, sizeof y);
  return y;
}

static cf_u16 app_f32_to_bf16(float x)
{
  unsigned int bits;
  memcpy(&bits, &x, sizeof bits);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return (cf_u16)(bits >> 16);
}

static int app_status_ok(cf_status status, const char *what)
{
  if(status == CF_OK) return 1;
  printf("FAIL %s -> %s\n", what, cf_status_as_char(status));
  return 0;
}

static int app_cuda_ok(cudaError_t status, const char *what)
{
  if(status == cudaSuccess) return 1;
  printf("FAIL %s -> %s\n", what, cudaGetErrorString(status));
  return 0;
}

static int app_sync(cf_math_handle *handle)
{
  if(handle->device != CF_MATH_DEVICE_CUDA) return 1;
  return app_cuda_ok(cudaStreamSynchronize(handle->workspace->stream), "cudaStreamSynchronize");
}

static int app_copy_h2d(cf_math_handle *handle, const cf_math *math, const void *host, cf_usize bytes)
{
  void *dst = app_math_data(handle, math);
  if(handle->device == CF_MATH_DEVICE_CUDA)
    return app_cuda_ok(cudaMemcpyAsync(dst, host, bytes, cudaMemcpyHostToDevice, handle->workspace->stream), "cudaMemcpyAsync H2D");

  memcpy(dst, host, bytes);
  return 1;
}

static int app_copy_d2h(cf_math_handle *handle, const cf_math *math, void *host, cf_usize bytes)
{
  const void *src = app_math_data(handle, math);
  if(handle->device == CF_MATH_DEVICE_CUDA)
  {
    if(!app_cuda_ok(cudaMemcpyAsync(host, src, bytes, cudaMemcpyDeviceToHost, handle->workspace->stream), "cudaMemcpyAsync D2H")) return 0;
    return app_sync(handle);
  }

  memcpy(host, src, bytes);
  return 1;
}

static const char *app_op_name(cf_math_op_kind op)
{
  switch (op)
  {
    case CF_MATH_OP_ADD: return "ADD";
    case CF_MATH_OP_SUB: return "SUB";
    case CF_MATH_OP_MUL: return "MUL";
    case CF_MATH_OP_DIV: return "DIV";
    case CF_MATH_OP_NEG: return "NEG";
    default: return "OP";
  }
}

static int app_op_is_binary(cf_math_op_kind op)
{
  return op != CF_MATH_OP_NEG;
}

static void app_fill_inputs(cf_math_dtype dtype, void *a, void *b, void *c, cf_usize padded_n, cf_usize bytes)
{
  memset(c, 0, bytes);

  switch (dtype)
  {
    case CF_MATH_DTYPE_F16:
    {
      cf_u16 *ah = (cf_u16 *)a;
      cf_u16 *bh = (cf_u16 *)b;
      for (cf_usize i = 0; i < padded_n; i++)
      {
        ah[i] = app_f32_to_f16((float)((i % 251) + 1) * 0.25f);
        bh[i] = app_f32_to_f16((float)((i % 31) + 1) * 0.5f + 1.0f);
      }
    }
    break;

    case CF_MATH_DTYPE_BF16:
    {
      cf_u16 *ah = (cf_u16 *)a;
      cf_u16 *bh = (cf_u16 *)b;
      for (cf_usize i = 0; i < padded_n; i++)
      {
        ah[i] = app_f32_to_bf16((float)((i % 251) + 1) * 0.25f);
        bh[i] = app_f32_to_bf16((float)((i % 31) + 1) * 0.5f + 1.0f);
      }
    }
    break;

    case CF_MATH_DTYPE_F32:
    {
      float *af = (float *)a;
      float *bf = (float *)b;
      for (cf_usize i = 0; i < padded_n; i++)
      {
        af[i] = (float)((i % 997) + 1) * 0.125f;
        bf[i] = (float)((i % 37) + 1) * 0.25f + 1.0f;
      }
    }
    break;

    case CF_MATH_DTYPE_F64:
    {
      double *ad = (double *)a;
      double *bd = (double *)b;
      for (cf_usize i = 0; i < padded_n; i++)
      {
        ad[i] = (double)((i % 997) + 1) * 0.125;
        bd[i] = (double)((i % 37) + 1) * 0.25 + 1.0;
      }
    }
    break;

    case CF_MATH_DTYPE_I32:
    {
      cf_i32 *ai = (cf_i32 *)a;
      cf_i32 *bi = (cf_i32 *)b;
      for (cf_usize i = 0; i < padded_n; i++)
      {
        ai[i] = (cf_i32)(1000 + (i % 1000));
        bi[i] = (cf_i32)(1 + (i % 13));
      }
    }
    break;

    default:
    break;
  }
}

static double app_value_as_double(const void *data, cf_math_dtype dtype, cf_usize index)
{
  switch (dtype)
  {
    case CF_MATH_DTYPE_F16: return (double)app_f16_to_f32(((const cf_u16 *)data)[index]);
    case CF_MATH_DTYPE_BF16: return (double)app_bf16_to_f32(((const cf_u16 *)data)[index]);
    case CF_MATH_DTYPE_F32: return (double)((const float *)data)[index];
    case CF_MATH_DTYPE_F64: return ((const double *)data)[index];
    case CF_MATH_DTYPE_I32: return (double)((const cf_i32 *)data)[index];
    default: return 0.0;
  }
}

static cf_i32 app_value_as_i32(const void *data, cf_usize index)
{
  return ((const cf_i32 *)data)[index];
}

static double app_expected_float_value(cf_math_dtype dtype, double av, double bv, cf_math_op_kind op)
{
  double expected = 0.0;

  switch (op)
  {
    case CF_MATH_OP_ADD: expected = av + bv; break;
    case CF_MATH_OP_SUB: expected = av - bv; break;
    case CF_MATH_OP_MUL: expected = av * bv; break;
    case CF_MATH_OP_DIV: expected = av / bv; break;
    case CF_MATH_OP_NEG: expected = -av; break;
    default: expected = 0.0; break;
  }

  if(dtype == CF_MATH_DTYPE_F16)
    return (double)app_f16_to_f32(app_f32_to_f16((float)expected));
  if(dtype == CF_MATH_DTYPE_BF16)
    return (double)app_bf16_to_f32(app_f32_to_bf16((float)expected));
  if(dtype == CF_MATH_DTYPE_F32)
    return (double)(float)expected;
  return expected;
}

static int app_check_one(const app_dtype_case *spec,
                         const void *a,
                         const void *b,
                         const void *c,
                         cf_usize index,
                         cf_math_op_kind op)
{
  if(spec->dtype == CF_MATH_DTYPE_I32)
  {
    const cf_i32 av = app_value_as_i32(a, index);
    const cf_i32 bv = app_op_is_binary(op) ? app_value_as_i32(b, index) : 0;
    const cf_i32 got = app_value_as_i32(c, index);
    cf_i32 expected = 0;

    switch (op)
    {
      case CF_MATH_OP_ADD: expected = av + bv; break;
      case CF_MATH_OP_SUB: expected = av - bv; break;
      case CF_MATH_OP_MUL: expected = av * bv; break;
      case CF_MATH_OP_DIV: expected = av / bv; break;
      case CF_MATH_OP_NEG: expected = -av; break;
      default: break;
    }

    if(got != expected)
    {
      printf("FAIL %s %s[%llu]: got=%d expected=%d\n",
             spec->name,
             app_op_name(op),
             (unsigned long long)index,
             got,
             expected);
      return 0;
    }
    return 1;
  }

  const double av = app_value_as_double(a, spec->dtype, index);
  const double bv = app_op_is_binary(op) ? app_value_as_double(b, spec->dtype, index) : 0.0;
  const double got = app_value_as_double(c, spec->dtype, index);
  const double expected = app_expected_float_value(spec->dtype, av, bv, op);
  const double scale = fabs(expected) > 1.0 ? fabs(expected) : 1.0;
  const double diff = fabs(got - expected);

  if(diff > spec->tolerance * scale)
  {
    printf("FAIL %s %s[%llu]: got=%g expected=%g diff=%g\n",
           spec->name,
           app_op_name(op),
           (unsigned long long)index,
           got,
           expected,
           diff);
    return 0;
  }

  return 1;
}

static int app_check_result(const app_dtype_case *spec,
                            const void *a,
                            const void *b,
                            const void *c,
                            cf_usize n,
                            cf_math_op_kind op)
{
  const cf_usize probes[3] = {0, n / 2, n - 1};

  for (int i = 0; i < 3; i++)
    if(!app_check_one(spec, a, b, c, probes[i], op)) return 0;

  return 1;
}

static long long app_bench_op(cf_math_handle *handle,
                              cf_math *out,
                              const cf_math *a,
                              const cf_math *b,
                              cf_math_op_kind op,
                              int iter)
{
  cf_time_point start;
  cf_time_point end;

  app_sync(handle);
  cf_time_now_mono(&start);
  for (int i = 0; i < iter; i++)
    cf_math_wise_op(handle, out, a, b, op);
  app_sync(handle);
  cf_time_now_mono(&end);

  return (long long)cf_time_as_ns(cf_time_elapsed(start, end));
}

static void app_print_result(const app_dtype_case *spec,
                             cf_math_op_kind op,
                             long long ns,
                             int iter,
                             cf_usize n)
{
  const double seconds = (double)ns / 1000000000.0;
  const double ns_per_op = (double)ns / (double)iter;
  const double elems_per_sec = seconds > 0.0 ? ((double)n * (double)iter) / seconds : 0.0;
  const double traffic = (double)n * (double)iter * (double)spec->elem_size * (app_op_is_binary(op) ? 3.0 : 2.0);
  const double gb_per_sec = seconds > 0.0 ? (traffic / 1000000000.0) / seconds : 0.0;

  printf("%-4s total: %lld ns | per op: %.2f ns | %.2f Melem/s | %.2f GB/s\n",
         app_op_name(op),
         ns,
         ns_per_op,
         elems_per_sec / 1000000.0,
         gb_per_sec);
}

static void app_print_cpu_caps(void)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
  __builtin_cpu_init();
  printf("cpu caps: avx=%d avx2=%d fma=%d f16c=%d\n",
         __builtin_cpu_supports("avx") != 0,
         __builtin_cpu_supports("avx2") != 0,
         __builtin_cpu_supports("fma") != 0,
         __builtin_cpu_supports("f16c") != 0);
#else
  printf("cpu caps: unavailable on this compiler/arch\n");
#endif
}

static int app_print_cuda_device(void)
{
  cudaDeviceProp prop;
  if(!app_cuda_ok(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties")) return 0;
  printf("cuda device: %s | sm_%d%d | global memory %.2f GiB\n",
         prop.name,
         prop.major,
         prop.minor,
         (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  return 1;
}

static int app_run_dtype(cf_math_context *context,
                         cf_math_workspace *workspace,
                         cf_math_device device,
                         const app_dtype_case *spec)
{
  enum { ITER = 1000 };
  const cf_usize n = (cf_usize)(1024 * 1024 + 3);
  const cf_usize bytes = app_round_up(n * spec->elem_size, 16);
  const cf_usize padded_n = bytes / spec->elem_size;
  const cf_usize arena_size = bytes * 3;
  const int dim[1] = { (int)n };
  const cf_math_op_kind ops[5] = {
    CF_MATH_OP_ADD,
    CF_MATH_OP_SUB,
    CF_MATH_OP_MUL,
    CF_MATH_OP_DIV,
    CF_MATH_OP_NEG
  };

  cf_math_handle handle = {0};
  cf_math_desc desc = {0};
  cf_math a = {0};
  cf_math b = {0};
  cf_math c = {0};
  void *a_host = malloc(bytes);
  void *b_host = malloc(bytes);
  void *c_host = malloc(bytes);
  int ok = 0;

  if(a_host == CF_NULL || b_host == CF_NULL || c_host == CF_NULL)
  {
    printf("FAIL malloc host buffers for %s\n", spec->name);
    goto done;
  }

  app_fill_inputs(spec->dtype, a_host, b_host, c_host, padded_n, bytes);

  if(!app_status_ok(cf_math_handle_create(&handle, context, workspace, arena_size, device), "cf_math_handle_create")) goto done;
  if(!app_status_ok(cf_math_desc_create(&desc, 1, dim, spec->dtype, CF_MATH_DESC_NONE), "cf_math_desc_create")) goto done;
  if(!app_status_ok(cf_math_bind(&handle, &a, &desc), "cf_math_bind a")) goto done;
  if(!app_status_ok(cf_math_bind(&handle, &b, &desc), "cf_math_bind b")) goto done;
  if(!app_status_ok(cf_math_bind(&handle, &c, &desc), "cf_math_bind c")) goto done;
  if(!app_copy_h2d(&handle, &a, a_host, bytes)) goto done;
  if(!app_copy_h2d(&handle, &b, b_host, bytes)) goto done;
  if(!app_copy_h2d(&handle, &c, c_host, bytes)) goto done;
  if(!app_sync(&handle)) goto done;

  printf("\n%s elementwise: elements=%llu padded=%llu bytes/tensor=%llu iterations=%d\n",
         spec->name,
         (unsigned long long)n,
         (unsigned long long)padded_n,
         (unsigned long long)bytes,
         ITER);

  for (int i = 0; i < 5; i++)
  {
    const cf_math_op_kind op = ops[i];
    const cf_math *rhs = app_op_is_binary(op) ? &b : CF_NULL;
    const long long ns = app_bench_op(&handle, &c, &a, rhs, op, ITER);

    if(!app_copy_d2h(&handle, &c, c_host, bytes)) goto done;
    if(!app_check_result(spec, a_host, b_host, c_host, n, op)) goto done;
    app_print_result(spec, op, ns, ITER, n);
  }

  ok = 1;

done:
  if(handle.storage.backend != CF_NULL) app_sync(&handle);
  cf_math_desc_destroy(&desc);
  cf_math_handle_destroy(&handle);
  free(a_host);
  free(b_host);
  free(c_host);
  return ok;
}

int main(void)
{
  static const app_dtype_case cases[] = {
    {CF_MATH_DTYPE_F16,  "F16 ", sizeof(cf_u16), 2.5e-2},
    {CF_MATH_DTYPE_BF16, "BF16", sizeof(cf_u16), 8.0e-2},
    {CF_MATH_DTYPE_F32,  "F32 ", sizeof(float),  1.0e-5},
    {CF_MATH_DTYPE_F64,  "F64 ", sizeof(double), 1.0e-12},
    {CF_MATH_DTYPE_I32,  "I32 ", sizeof(cf_i32), 0.0}
  };

  const cf_math_device device = CF_APP_DEVICE == CF_APP_CUDA ? CF_MATH_DEVICE_CUDA : CF_MATH_DEVICE_CPU;
  cf_math_context context = {0};
  cf_math_workspace workspace = {0};
  int ok = 1;

  printf("cf_math_wise_op benchmark backend=%s (CF_APP_DEVICE=%d)\n",
         device == CF_MATH_DEVICE_CUDA ? "CUDA" : "CPU",
         CF_APP_DEVICE);
  app_print_cpu_caps();
  if(device == CF_MATH_DEVICE_CUDA && !app_print_cuda_device()) return 1;

  if(!app_status_ok(cf_math_context_create(&context, device == CF_MATH_DEVICE_CUDA ? 0 : 1, device), "cf_math_context_create")) return 1;
  if(!app_status_ok(cf_math_workspace_create(&workspace, 1024 * 1024, device), "cf_math_workspace_create"))
  {
    cf_math_context_destroy(&context);
    return 1;
  }

  for (cf_usize i = 0; i < sizeof cases / sizeof cases[0]; i++)
  {
    if(!app_run_dtype(&context, &workspace, device, &cases[i]))
    {
      ok = 0;
      break;
    }
  }

  cf_math_workspace_destroy(&workspace);
  cf_math_context_destroy(&context);

  return ok ? 0 : 1;
}
