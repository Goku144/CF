#include "MATH/cf_math.h"
#include "MATH/cf_math_print.h"
#include "RUNTIME/cf_status.h"

#include <string.h>
#include <stdio.h>

static int app_status(const char *name, cf_status status)
{
  printf("%-32s %s\n", name, cf_status_as_char(status));
  return status == CF_OK ? 0 : 1;
}

static cf_status app_fill_f32(cf_math *x, const float *values, cf_usize count)
{
  cf_usize bytes = 0;

  if(x == CF_NULL || values == CF_NULL) return CF_ERR_NULL;
  if(x->handler == CF_NULL || x->metadata == CF_NULL || x->data == CF_NULL) return CF_ERR_STATE;
  if(x->handler->storage.dtype != CF_MATH_DTYPE_F32) return CF_ERR_INVALID;
  if(count > x->metadata->len) return CF_ERR_BOUNDS;
  if(count > (cf_usize)-1 / sizeof(float)) return CF_ERR_OVERFLOW;

  bytes = count * sizeof(float);
  if(bytes > x->byte_size) return CF_ERR_BOUNDS;

  if(x->handler->storage.device == CF_MATH_DEVICE_CPU || (x->handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0)
  {
    memcpy(x->data, values, (size_t)bytes);
    return CF_OK;
  }

#if defined(CF_CUDA_AVAILABLE)
  if(cudaMemcpy(x->data, values, (size_t)bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    return CF_ERR_CUDA_COPY;
  if(x->handler->cuda_ctx != CF_NULL && x->handler->cuda_ctx->stream != CF_NULL)
    return cudaStreamSynchronize(x->handler->cuda_ctx->stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
  return cudaDeviceSynchronize() == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
#else
  return CF_ERR_UNSUPPORTED;
#endif
}

int main(void)
{
  cf_math_cuda_context ctx = {0};
  cf_math_handle_t handler = {0};
  cf_math_metadata matrix_meta = {0};
  cf_math_metadata vector_meta = {0};
  cf_math a = {0};
  cf_math b = {0};
  cf_math c = {0};
  cf_usize matrix_dim[CF_MATH_MAX_RANK] = {2, 2};
  cf_usize vector_dim[CF_MATH_MAX_RANK] = {4};
  float a_values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_values[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  float c_values[4] = {100.0f, 200.0f, 300.0f, 400.0f};
  float b_vector_values[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
  cf_status status = CF_OK;
  cf_usize first_offset = 0;

  status = cf_math_cuda_context_init(&ctx, 0, 0);
  if(app_status("cuda_context_init", status) != 0) return 0;

  status = cf_math_metadata_init(&matrix_meta, matrix_dim, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR);
  if(app_status("metadata matrix", status) != 0) goto cleanup_ctx;

  status = cf_math_metadata_init(&vector_meta, vector_dim, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR);
  if(app_status("metadata vector", status) != 0) goto cleanup_ctx;

  status = cf_math_handle_init(
    &handler,
    &ctx,
    CF_MATH_DTYPE_F32,
    CF_MATH_DEVICE_CUDA,
    CF_MATH_MEM_MANAGED | CF_MATH_MEM_ALIGNED128,
    CF_MATH_HANDLE_OPT_MATMUL | CF_MATH_HANDLE_OPT_ELEMENTWISE,
    0
  );
  if(app_status("handle_init", status) != 0) goto cleanup_ctx;
  printf("%-32s %s\n", "shared cuda context", handler.cuda_ctx == &ctx ? "yes" : "no");
  printf("%-32s %s\n", "running on cuda gpu", status == CF_OK && handler.storage.device == CF_MATH_DEVICE_CUDA && handler.cuda_ctx != CF_NULL && handler.cuda_ctx->stream != CF_NULL ? "yes" : "no");

  status = cf_math_bind(&a, &handler, &matrix_meta);
  if(app_status("bind a", status) != 0) goto cleanup_handler;
  first_offset = a.byte_offset;
  status = app_fill_f32(&a, a_values, 4);
  if(app_status("fill a", status) != 0) goto cleanup_handler;
  status = cf_math_print_shape(&a);
  if(app_status("print a", status) != 0) goto cleanup_handler;

  status = cf_math_bind(&b, &handler, &matrix_meta);
  if(app_status("bind b", status) != 0) goto cleanup_handler;
  status = app_fill_f32(&b, b_values, 4);
  if(app_status("fill b", status) != 0) goto cleanup_handler;
  status = cf_math_print_shape(&b);
  if(app_status("print b", status) != 0) goto cleanup_handler;

  status = cf_math_unbind(&a);
  if(app_status("unbind a", status) != 0) goto cleanup_handler;

  status = cf_math_bind(&c, &handler, &matrix_meta);
  if(app_status("bind c reuse", status) != 0) goto cleanup_handler;
  printf("%-32s %s\n", "reuse old slice", c.byte_offset == first_offset ? "yes" : "no");
  status = app_fill_f32(&c, c_values, 4);
  if(app_status("fill c", status) != 0) goto cleanup_handler;
  status = cf_math_print_shape(&c);
  if(app_status("print c", status) != 0) goto cleanup_handler;

  status = cf_math_rebind(&b, &handler, &vector_meta);
  if(app_status("rebind b vector", status) != 0) goto cleanup_handler;
  status = app_fill_f32(&b, b_vector_values, 4);
  if(app_status("fill b vector", status) != 0) goto cleanup_handler;
  status = cf_math_print_shape(&b);
  if(app_status("print b rebound", status) != 0) goto cleanup_handler;

  (void)cf_math_unbind(&b);
  (void)cf_math_unbind(&c);

cleanup_handler:
  status = cf_math_handle_destroy(&handler);
  (void)app_status("handle_destroy", status);

cleanup_ctx:
  status = cf_math_cuda_context_destroy(&ctx);
  (void)app_status("cuda_context_destroy", status);

  return 0;
}
