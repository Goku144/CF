#include "MATH/cf_math.h"
#include "RUNTIME/cf_status.h"

#include <stdio.h>

static int app_status(const char *name, cf_status status)
{
  printf("%-32s %s\n", name, cf_status_as_char(status));
  return status == CF_OK ? 0 : 1;
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
  cf_status status = CF_OK;
  cf_usize first_offset = 0;

  status = cf_math_cuda_context_init(&ctx, 0);
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

  status = cf_math_bind(&a, &handler, &matrix_meta);
  if(app_status("bind a", status) != 0) goto cleanup_handler;
  first_offset = a.byte_offset;

  status = cf_math_bind(&b, &handler, &matrix_meta);
  if(app_status("bind b", status) != 0) goto cleanup_handler;

  status = cf_math_unbind(&a);
  if(app_status("unbind a", status) != 0) goto cleanup_handler;

  status = cf_math_bind(&c, &handler, &matrix_meta);
  if(app_status("bind c reuse", status) != 0) goto cleanup_handler;
  printf("%-32s %s\n", "reuse old slice", c.byte_offset == first_offset ? "yes" : "no");

  status = cf_math_rebind(&b, &handler, &vector_meta);
  if(app_status("rebind b vector", status) != 0) goto cleanup_handler;

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
