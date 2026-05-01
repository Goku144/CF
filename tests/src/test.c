/*
 * CF Framework
 * Copyright (C) 2026 Orion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "ALLOCATOR/cf_arena.h"
#include "ALLOCATOR/cf_pool.h"
#include "AI/cf_gradient.h"
#include "AI/cf_model.h"
#include "MATH/cf_math.h"
#include "RUNTIME/cf_status.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CF_TEST_CHECK(expr) \
  do \
  { \
    if(!(expr)) \
    { \
      printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
      return CF_ERR_STATE; \
    } \
  } while(0)

#define CF_TEST_STATUS(expr) \
  do \
  { \
    cf_status test_status__ = (expr); \
    if(test_status__ != CF_OK) \
    { \
      printf("FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr, cf_status_as_char(test_status__)); \
      return test_status__; \
    } \
  } while(0)

#define CF_TEST_CLOSE(a, b, eps) CF_TEST_CHECK(fabs((double)(a) - (double)(b)) <= (double)(eps))

static cf_status test_allocator_arena(void)
{
  cf_arena arena = {0};
  void *a = CF_NULL;
  void *b = CF_NULL;
  cf_u8 buffer[64] = {0};

  CF_TEST_STATUS(cf_arena_init(&arena, 128, CF_NULL));
  CF_TEST_STATUS(cf_arena_alloc(&arena, 1, 1, &a));
  CF_TEST_STATUS(cf_arena_alloc(&arena, 16, 16, &b));
  CF_TEST_CHECK(a != CF_NULL);
  CF_TEST_CHECK(b != CF_NULL);
  CF_TEST_CHECK((((cf_uptr)b) & 15U) == 0);
  CF_TEST_CHECK(arena.high_water >= arena.offset);
  cf_arena_reset(&arena);
  CF_TEST_CHECK(arena.offset == 0);
  cf_arena_destroy(&arena);
  CF_TEST_CHECK(arena.data == CF_NULL);

  CF_TEST_STATUS(cf_arena_init_with_buffer(&arena, buffer, sizeof(buffer)));
  CF_TEST_STATUS(cf_arena_alloc(&arena, 32, 8, &a));
  CF_TEST_CHECK(a == buffer);
  cf_arena_destroy(&arena);
  CF_TEST_CHECK(buffer[0] == 0);

  return CF_OK;
}

static cf_status test_allocator_pool(void)
{
  cf_pool pool = {0};
  void *a = CF_NULL;
  void *b = CF_NULL;
  void *c = CF_NULL;

  CF_TEST_STATUS(cf_pool_init(&pool, sizeof(cf_u32), 2, CF_NULL));
  CF_TEST_STATUS(cf_pool_alloc(&pool, &a));
  CF_TEST_STATUS(cf_pool_alloc(&pool, &b));
  CF_TEST_CHECK(a != CF_NULL);
  CF_TEST_CHECK(b != CF_NULL);
  CF_TEST_CHECK(a != b);
  CF_TEST_CHECK(cf_pool_alloc(&pool, &c) == CF_ERR_OOM);
  CF_TEST_STATUS(cf_pool_free(&pool, a));
  CF_TEST_STATUS(cf_pool_alloc(&pool, &c));
  CF_TEST_CHECK(c == a);
  CF_TEST_CHECK(pool.free_count == 0);
  cf_pool_reset(&pool);
  CF_TEST_CHECK(pool.free_count == pool.block_count);
  cf_pool_destroy(&pool);
  CF_TEST_CHECK(pool.data == CF_NULL);

  return CF_OK;
}

static cf_status test_math_cpu_pooled_storage(void)
{
  cf_math_handle_t handler = {0};
  cf_math_metadata metadata = {0};
  cf_math a = {0};
  cf_math b = {0};
  cf_math c = {0};
  cf_usize dims[CF_MATH_MAX_RANK] = {4};
  float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float output[4] = {0};
  cf_usize first_offset = 0;

  CF_TEST_STATUS(cf_math_handle_init(&handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_POOLED | CF_MATH_MEM_ALIGNED128, CF_MATH_HANDLE_OPT_NONE, 64));
  CF_TEST_STATUS(cf_math_metadata_init(&metadata, dims, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_bind(&a, &handler, &metadata));
  CF_TEST_STATUS(cf_math_bind(&b, &handler, &metadata));
  CF_TEST_CHECK((a.byte_offset & 127U) == 0);
  CF_TEST_CHECK((b.byte_offset & 127U) == 0);
  CF_TEST_CHECK(a.byte_offset != b.byte_offset);
  CF_TEST_STATUS(cf_math_cpy_h2d(&a, input, 4));
  CF_TEST_STATUS(cf_math_cpy_d2h(&a, output, 4));
  CF_TEST_CHECK(memcmp(input, output, sizeof(input)) == 0);

  first_offset = a.byte_offset;
  CF_TEST_STATUS(cf_math_unbind(&a));
  CF_TEST_STATUS(cf_math_bind(&c, &handler, &metadata));
  CF_TEST_CHECK(c.byte_offset == first_offset);

  CF_TEST_STATUS(cf_math_unbind(&b));
  CF_TEST_STATUS(cf_math_unbind(&c));
  cf_math_handle_reset(&handler);
  CF_TEST_CHECK(handler.storage.arena.offset == 0);
  CF_TEST_STATUS(cf_math_handle_destroy(&handler));

  return CF_OK;
}

static cf_status test_math_cpu_storage_flags(void)
{
  cf_math_handle_t handler = {0};
  cf_math_cuda_context fake_ctx = {0};

  CF_TEST_STATUS(cf_math_handle_init(&handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT | CF_MATH_MEM_READ_ONLY, CF_MATH_HANDLE_OPT_NONE, 32));
  CF_TEST_CHECK(handler.storage.allocator.backend != CF_NULL);
  CF_TEST_STATUS(cf_math_handle_destroy(&handler));

  CF_TEST_CHECK(cf_math_handle_init(&handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_MANAGED, CF_MATH_HANDLE_OPT_NONE, 0) == CF_ERR_INVALID);
  CF_TEST_CHECK(cf_math_handle_init(&handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_PEER_MAPPED, CF_MATH_HANDLE_OPT_NONE, 0) == CF_ERR_UNSUPPORTED);

#if !defined(CF_CUDA_AVAILABLE)
  CF_TEST_CHECK(cf_math_handle_init(&handler, &fake_ctx, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_PINNED, CF_MATH_HANDLE_OPT_NONE, 0) == CF_ERR_UNSUPPORTED);
#else
  CF_TEST_STATUS(cf_math_handle_init(&handler, &fake_ctx, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_PINNED, CF_MATH_HANDLE_OPT_NONE, 0));
  CF_TEST_STATUS(cf_math_handle_destroy(&handler));
#endif

  return CF_OK;
}

static cf_status test_math_cpu_ops_f32(void)
{
  cf_math_handle_t handler = {0};
  cf_math_metadata metadata = {0};
  cf_math a = {0};
  cf_math b = {0};
  cf_usize dims[CF_MATH_MAX_RANK] = {4};
  float a_data[4] = {8.0f, 12.0f, 20.0f, 24.0f};
  float b_data[4] = {2.0f, 3.0f, 4.0f, 6.0f};
  float out[4] = {0};

  CF_TEST_STATUS(cf_math_handle_init(&handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_POOLED | CF_MATH_MEM_ALIGNED128, CF_MATH_HANDLE_OPT_ELEMENTWISE, 512));
  CF_TEST_STATUS(cf_math_metadata_init(&metadata, dims, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_bind(&a, &handler, &metadata));
  CF_TEST_STATUS(cf_math_bind(&b, &handler, &metadata));

  CF_TEST_STATUS(cf_math_cpy_h2d(&a, a_data, 4));
  CF_TEST_STATUS(cf_math_cpy_h2d(&b, b_data, 4));
  CF_TEST_STATUS(cf_math_op(CF_MATH_OP_ADD, &a, &b));
  CF_TEST_STATUS(cf_math_cpy_d2h(&a, out, 4));
  CF_TEST_CHECK(out[0] == 10.0f && out[1] == 15.0f && out[2] == 24.0f && out[3] == 30.0f);

  CF_TEST_STATUS(cf_math_cpy_h2d(&a, a_data, 4));
  CF_TEST_STATUS(cf_math_op(CF_MATH_OP_SUB, &a, &b));
  CF_TEST_STATUS(cf_math_cpy_d2h(&a, out, 4));
  CF_TEST_CHECK(out[0] == 6.0f && out[1] == 9.0f && out[2] == 16.0f && out[3] == 18.0f);

  CF_TEST_STATUS(cf_math_cpy_h2d(&a, a_data, 4));
  CF_TEST_STATUS(cf_math_op(CF_MATH_OP_MUL, &a, &b));
  CF_TEST_STATUS(cf_math_cpy_d2h(&a, out, 4));
  CF_TEST_CHECK(out[0] == 16.0f && out[1] == 36.0f && out[2] == 80.0f && out[3] == 144.0f);

  CF_TEST_STATUS(cf_math_cpy_h2d(&a, a_data, 4));
  CF_TEST_STATUS(cf_math_op(CF_MATH_OP_DIV, &a, &b));
  CF_TEST_STATUS(cf_math_cpy_d2h(&a, out, 4));
  CF_TEST_CHECK(out[0] == 4.0f && out[1] == 4.0f && out[2] == 5.0f && out[3] == 4.0f);

  CF_TEST_CHECK(cf_math_op(CF_MATH_OP_MATMUL, &a, &b) == CF_ERR_UNSUPPORTED);
  CF_TEST_STATUS(cf_math_unbind(&a));
  CF_TEST_STATUS(cf_math_unbind(&b));
  CF_TEST_STATUS(cf_math_handle_destroy(&handler));
  return CF_OK;
}

static cf_status test_math_cpu_ops_f64_i32(void)
{
  cf_math_handle_t f64_handler = {0};
  cf_math_handle_t i32_handler = {0};
  cf_math_metadata metadata = {0};
  cf_math a = {0};
  cf_math b = {0};
  cf_usize dims[CF_MATH_MAX_RANK] = {3};
  double f64_a[3] = {1.5, 2.0, 4.0};
  double f64_b[3] = {2.0, 3.0, 0.5};
  double f64_out[3] = {0};
  cf_i32 i32_a[3] = {7, 9, 11};
  cf_i32 i32_b[3] = {2, 3, 4};
  cf_i32 i32_out[3] = {0};

  CF_TEST_STATUS(cf_math_metadata_init(&metadata, dims, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR));

  CF_TEST_STATUS(cf_math_handle_init(&f64_handler, CF_NULL, CF_MATH_DTYPE_F64, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_ELEMENTWISE, 256));
  CF_TEST_STATUS(cf_math_bind(&a, &f64_handler, &metadata));
  CF_TEST_STATUS(cf_math_bind(&b, &f64_handler, &metadata));
  CF_TEST_STATUS(cf_math_cpy_h2d(&a, f64_a, 3));
  CF_TEST_STATUS(cf_math_cpy_h2d(&b, f64_b, 3));
  CF_TEST_STATUS(cf_math_op(CF_MATH_OP_ADD, &a, &b));
  CF_TEST_STATUS(cf_math_cpy_d2h(&a, f64_out, 3));
  CF_TEST_CHECK(f64_out[0] == 3.5 && f64_out[1] == 5.0 && f64_out[2] == 4.5);
  CF_TEST_STATUS(cf_math_cpy_h2d(&a, f64_a, 3));
  CF_TEST_STATUS(cf_math_op(CF_MATH_OP_MUL, &a, &b));
  CF_TEST_STATUS(cf_math_cpy_d2h(&a, f64_out, 3));
  CF_TEST_CHECK(f64_out[0] == 3.0 && f64_out[1] == 6.0 && f64_out[2] == 2.0);
  CF_TEST_STATUS(cf_math_unbind(&a));
  CF_TEST_STATUS(cf_math_unbind(&b));
  CF_TEST_STATUS(cf_math_handle_destroy(&f64_handler));

  CF_TEST_STATUS(cf_math_handle_init(&i32_handler, CF_NULL, CF_MATH_DTYPE_I32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_ELEMENTWISE, 256));
  CF_TEST_STATUS(cf_math_bind(&a, &i32_handler, &metadata));
  CF_TEST_STATUS(cf_math_bind(&b, &i32_handler, &metadata));
  CF_TEST_STATUS(cf_math_cpy_h2d(&a, i32_a, 3));
  CF_TEST_STATUS(cf_math_cpy_h2d(&b, i32_b, 3));
  CF_TEST_STATUS(cf_math_op(CF_MATH_OP_ADD, &a, &b));
  CF_TEST_STATUS(cf_math_cpy_d2h(&a, i32_out, 3));
  CF_TEST_CHECK(i32_out[0] == 9 && i32_out[1] == 12 && i32_out[2] == 15);
  CF_TEST_STATUS(cf_math_cpy_h2d(&a, i32_a, 3));
  CF_TEST_STATUS(cf_math_op(CF_MATH_OP_SUB, &a, &b));
  CF_TEST_STATUS(cf_math_cpy_d2h(&a, i32_out, 3));
  CF_TEST_CHECK(i32_out[0] == 5 && i32_out[1] == 6 && i32_out[2] == 7);
  CF_TEST_STATUS(cf_math_cpy_h2d(&a, i32_a, 3));
  CF_TEST_STATUS(cf_math_op(CF_MATH_OP_MUL, &a, &b));
  CF_TEST_STATUS(cf_math_cpy_d2h(&a, i32_out, 3));
  CF_TEST_CHECK(i32_out[0] == 14 && i32_out[1] == 27 && i32_out[2] == 44);
  CF_TEST_STATUS(cf_math_unbind(&a));
  CF_TEST_STATUS(cf_math_unbind(&b));
  CF_TEST_STATUS(cf_math_handle_destroy(&i32_handler));

  return CF_OK;
}

static cf_status test_math_cpu_ops_unsupported_dtype(void)
{
  cf_math_handle_t handler = {0};
  cf_math_metadata metadata = {0};
  cf_math a = {0};
  cf_math b = {0};
  cf_usize dims[CF_MATH_MAX_RANK] = {2};
  cf_u8 data[2] = {1, 2};

  CF_TEST_STATUS(cf_math_handle_init(&handler, CF_NULL, CF_MATH_DTYPE_U8, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_ELEMENTWISE, 64));
  CF_TEST_STATUS(cf_math_metadata_init(&metadata, dims, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_bind(&a, &handler, &metadata));
  CF_TEST_STATUS(cf_math_bind(&b, &handler, &metadata));
  CF_TEST_STATUS(cf_math_cpy_h2d(&a, data, 2));
  CF_TEST_STATUS(cf_math_cpy_h2d(&b, data, 2));
  CF_TEST_CHECK(cf_math_op(CF_MATH_OP_ADD, &a, &b) == CF_ERR_UNSUPPORTED);
  CF_TEST_STATUS(cf_math_unbind(&a));
  CF_TEST_STATUS(cf_math_unbind(&b));
  CF_TEST_STATUS(cf_math_handle_destroy(&handler));
  return CF_OK;
}

static cf_status test_math_op_check_and_out(void)
{
  cf_math_handle_t f32_handler = {0};
  cf_math_handle_t f64_handler = {0};
  cf_math_metadata meta4 = {0};
  cf_math_metadata meta3 = {0};
  cf_math a = {0};
  cf_math b = {0};
  cf_math out = {0};
  cf_math short_view = {0};
  cf_math dtype_other = {0};
  cf_math unbound = {0};
  cf_usize dims4[CF_MATH_MAX_RANK] = {4};
  cf_usize dims3[CF_MATH_MAX_RANK] = {3};
  float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  float result[4] = {0};

  CF_TEST_STATUS(cf_math_metadata_init(&meta4, dims4, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_metadata_init(&meta3, dims3, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_handle_init(&f32_handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_ELEMENTWISE, 512));
  CF_TEST_STATUS(cf_math_handle_init(&f64_handler, CF_NULL, CF_MATH_DTYPE_F64, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_ELEMENTWISE, 128));
  CF_TEST_STATUS(cf_math_bind(&a, &f32_handler, &meta4));
  CF_TEST_STATUS(cf_math_bind(&b, &f32_handler, &meta4));
  CF_TEST_STATUS(cf_math_bind(&out, &f32_handler, &meta4));
  CF_TEST_STATUS(cf_math_bind(&short_view, &f32_handler, &meta3));
  CF_TEST_STATUS(cf_math_bind(&dtype_other, &f64_handler, &meta4));
  CF_TEST_STATUS(cf_math_cpy_h2d(&a, a_data, 4));
  CF_TEST_STATUS(cf_math_cpy_h2d(&b, b_data, 4));

  CF_TEST_STATUS(cf_math_op_check(CF_MATH_OP_ADD, &a, &b));
  CF_TEST_CHECK(cf_math_op_check(CF_MATH_OP_ADD, &a, &unbound) == CF_ERR_STATE);
  CF_TEST_CHECK(cf_math_op_check(CF_MATH_OP_ADD, &a, &short_view) == CF_ERR_INVALID);
  CF_TEST_CHECK(cf_math_op_check(CF_MATH_OP_ADD, &a, &dtype_other) == CF_ERR_INVALID);
  CF_TEST_CHECK(cf_math_op_check(CF_MATH_OP_MATMUL, &a, &b) == CF_ERR_UNSUPPORTED);

  CF_TEST_STATUS(cf_math_op_out(CF_MATH_OP_ADD, &out, &a, &b));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out, result, 4));
  CF_TEST_CLOSE(result[0], 11.0f, 0.00001);
  CF_TEST_CLOSE(result[1], 22.0f, 0.00001);
  CF_TEST_CLOSE(result[2], 33.0f, 0.00001);
  CF_TEST_CLOSE(result[3], 44.0f, 0.00001);
  CF_TEST_STATUS(cf_math_cpy_d2h(&a, result, 4));
  CF_TEST_CLOSE(result[0], 1.0f, 0.00001);

  CF_TEST_STATUS(cf_math_unbind(&a));
  CF_TEST_STATUS(cf_math_unbind(&b));
  CF_TEST_STATUS(cf_math_unbind(&out));
  CF_TEST_STATUS(cf_math_unbind(&short_view));
  CF_TEST_STATUS(cf_math_unbind(&dtype_other));
  CF_TEST_STATUS(cf_math_handle_destroy(&f32_handler));
  CF_TEST_STATUS(cf_math_handle_destroy(&f64_handler));
  return CF_OK;
}

static cf_status test_math_cpu_unary_scalar_reduce(void)
{
  cf_math_handle_t f32_handler = {0};
  cf_math_handle_t i32_handler = {0};
  cf_math_metadata meta4 = {0};
  cf_math_metadata scalar_meta = {0};
  cf_math x = {0};
  cf_math out = {0};
  cf_math scalar_out = {0};
  cf_math xi = {0};
  cf_math iout = {0};
  cf_usize dims4[CF_MATH_MAX_RANK] = {4};
  cf_usize dim1[CF_MATH_MAX_RANK] = {1};
  float values[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
  float positive[4] = {1.0f, 4.0f, 9.0f, 16.0f};
  float result[4] = {0};
  cf_i32 ivals[4] = {1, 2, 3, 4};
  cf_i32 iresult[4] = {0};
  float scalar_result[1] = {0};
  cf_i32 i_scalar_result[1] = {0};

  CF_TEST_STATUS(cf_math_metadata_init(&meta4, dims4, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_metadata_init(&scalar_meta, dim1, 1, CF_MATH_SHAPE_SCALAR, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_handle_init(&f32_handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_POOLED | CF_MATH_MEM_ALIGNED128, CF_MATH_HANDLE_OPT_ELEMENTWISE, 1024));
  CF_TEST_STATUS(cf_math_handle_init(&i32_handler, CF_NULL, CF_MATH_DTYPE_I32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_ELEMENTWISE, 256));
  CF_TEST_STATUS(cf_math_bind(&x, &f32_handler, &meta4));
  CF_TEST_STATUS(cf_math_bind(&out, &f32_handler, &meta4));
  CF_TEST_STATUS(cf_math_bind(&scalar_out, &f32_handler, &scalar_meta));

  CF_TEST_STATUS(cf_math_cpy_h2d(&x, values, 4));
  CF_TEST_STATUS(cf_math_unary(CF_MATH_OP_RELU, &x));
  CF_TEST_STATUS(cf_math_cpy_d2h(&x, result, 4));
  CF_TEST_CLOSE(result[0], 0.0f, 0.00001);
  CF_TEST_CLOSE(result[1], 0.0f, 0.00001);
  CF_TEST_CLOSE(result[2], 1.0f, 0.00001);
  CF_TEST_CLOSE(result[3], 2.0f, 0.00001);

  CF_TEST_STATUS(cf_math_cpy_h2d(&x, values, 4));
  CF_TEST_STATUS(cf_math_unary_out(CF_MATH_OP_SIGMOID, &out, &x));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out, result, 4));
  CF_TEST_CLOSE(result[0], 0.268941f, 0.0001);
  CF_TEST_CLOSE(result[2], 0.731059f, 0.0001);

  CF_TEST_STATUS(cf_math_cpy_h2d(&x, values, 4));
  CF_TEST_STATUS(cf_math_unary_out(CF_MATH_OP_TANH, &out, &x));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out, result, 4));
  CF_TEST_CLOSE(result[0], tanh(-1.0), 0.0001);
  CF_TEST_CLOSE(result[3], tanh(2.0), 0.0001);

  CF_TEST_STATUS(cf_math_cpy_h2d(&x, values, 4));
  CF_TEST_STATUS(cf_math_unary_out(CF_MATH_OP_GELU, &out, &x));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out, result, 4));
  CF_TEST_CHECK(result[0] < 0.0f && result[3] > 1.9f);

  CF_TEST_STATUS(cf_math_cpy_h2d(&x, positive, 4));
  CF_TEST_STATUS(cf_math_unary_out(CF_MATH_OP_LOG, &out, &x));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out, result, 4));
  CF_TEST_CLOSE(result[1], log(4.0), 0.0001);
  CF_TEST_STATUS(cf_math_unary_out(CF_MATH_OP_SQRT, &out, &x));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out, result, 4));
  CF_TEST_CLOSE(result[2], 3.0f, 0.0001);
  CF_TEST_STATUS(cf_math_unary_out(CF_MATH_OP_EXP, &out, &x));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out, result, 4));
  CF_TEST_CLOSE(result[0], exp(1.0), 0.0001);

  CF_TEST_STATUS(cf_math_scalar_out(CF_MATH_OP_MUL, &out, &x, 2.0));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out, result, 4));
  CF_TEST_CLOSE(result[0], 2.0f, 0.0001);
  CF_TEST_CLOSE(result[3], 32.0f, 0.0001);
  CF_TEST_STATUS(cf_math_scalar(CF_MATH_OP_DIV, &out, 2.0));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out, result, 4));
  CF_TEST_CLOSE(result[3], 16.0f, 0.0001);

  CF_TEST_STATUS(cf_math_reduce_sum(&scalar_out, &x));
  CF_TEST_STATUS(cf_math_cpy_d2h(&scalar_out, scalar_result, 1));
  CF_TEST_CLOSE(scalar_result[0], 30.0f, 0.0001);
  CF_TEST_STATUS(cf_math_reduce_mean(&scalar_out, &x));
  CF_TEST_STATUS(cf_math_cpy_d2h(&scalar_out, scalar_result, 1));
  CF_TEST_CLOSE(scalar_result[0], 7.5f, 0.0001);

  CF_TEST_STATUS(cf_math_bind(&xi, &i32_handler, &meta4));
  CF_TEST_STATUS(cf_math_bind(&iout, &i32_handler, &scalar_meta));
  CF_TEST_STATUS(cf_math_cpy_h2d(&xi, ivals, 4));
  CF_TEST_STATUS(cf_math_scalar(CF_MATH_OP_ADD, &xi, 3.0));
  CF_TEST_STATUS(cf_math_cpy_d2h(&xi, iresult, 4));
  CF_TEST_CHECK(iresult[0] == 4 && iresult[3] == 7);
  CF_TEST_STATUS(cf_math_reduce_sum(&iout, &xi));
  CF_TEST_STATUS(cf_math_cpy_d2h(&iout, i_scalar_result, 1));
  CF_TEST_CHECK(i_scalar_result[0] == 22);

  CF_TEST_STATUS(cf_math_unbind(&x));
  CF_TEST_STATUS(cf_math_unbind(&out));
  CF_TEST_STATUS(cf_math_unbind(&scalar_out));
  CF_TEST_STATUS(cf_math_unbind(&xi));
  CF_TEST_STATUS(cf_math_unbind(&iout));
  CF_TEST_STATUS(cf_math_handle_destroy(&f32_handler));
  CF_TEST_STATUS(cf_math_handle_destroy(&i32_handler));
  return CF_OK;
}

static cf_status test_math_cpu_matmul(void)
{
  cf_math_handle_t f32_handler = {0};
  cf_math_handle_t f64_handler = {0};
  cf_math_metadata a_meta = {0};
  cf_math_metadata b_meta = {0};
  cf_math_metadata out_meta = {0};
  cf_math_metadata bad_meta = {0};
  cf_math a = {0};
  cf_math b = {0};
  cf_math out = {0};
  cf_math bad = {0};
  cf_usize a_dims[CF_MATH_MAX_RANK] = {2, 3};
  cf_usize b_dims[CF_MATH_MAX_RANK] = {3, 2};
  cf_usize out_dims[CF_MATH_MAX_RANK] = {2, 2};
  cf_usize bad_dims[CF_MATH_MAX_RANK] = {4};
  float af[6] = {1, 2, 3, 4, 5, 6};
  float bf[6] = {7, 8, 9, 10, 11, 12};
  float of[4] = {0};
  double ad[6] = {1, 2, 3, 4, 5, 6};
  double bd[6] = {7, 8, 9, 10, 11, 12};
  double od[4] = {0};

  CF_TEST_STATUS(cf_math_metadata_init(&a_meta, a_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_metadata_init(&b_meta, b_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_metadata_init(&out_meta, out_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_metadata_init(&bad_meta, bad_dims, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR));

  CF_TEST_STATUS(cf_math_handle_init(&f32_handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_MATMUL, 512));
  CF_TEST_STATUS(cf_math_bind(&a, &f32_handler, &a_meta));
  CF_TEST_STATUS(cf_math_bind(&b, &f32_handler, &b_meta));
  CF_TEST_STATUS(cf_math_bind(&out, &f32_handler, &out_meta));
  CF_TEST_STATUS(cf_math_bind(&bad, &f32_handler, &bad_meta));
  CF_TEST_STATUS(cf_math_cpy_h2d(&a, af, 6));
  CF_TEST_STATUS(cf_math_cpy_h2d(&b, bf, 6));
  CF_TEST_STATUS(cf_math_matmul(&out, &a, &b));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out, of, 4));
  CF_TEST_CLOSE(of[0], 58.0f, 0.0001);
  CF_TEST_CLOSE(of[1], 64.0f, 0.0001);
  CF_TEST_CLOSE(of[2], 139.0f, 0.0001);
  CF_TEST_CLOSE(of[3], 154.0f, 0.0001);
  CF_TEST_CHECK(cf_math_matmul(&bad, &a, &b) == CF_ERR_INVALID);
  CF_TEST_STATUS(cf_math_unbind(&a));
  CF_TEST_STATUS(cf_math_unbind(&b));
  CF_TEST_STATUS(cf_math_unbind(&out));
  CF_TEST_STATUS(cf_math_unbind(&bad));
  CF_TEST_STATUS(cf_math_handle_destroy(&f32_handler));

  CF_TEST_STATUS(cf_math_handle_init(&f64_handler, CF_NULL, CF_MATH_DTYPE_F64, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_MATMUL, 1024));
  CF_TEST_STATUS(cf_math_bind(&a, &f64_handler, &a_meta));
  CF_TEST_STATUS(cf_math_bind(&b, &f64_handler, &b_meta));
  CF_TEST_STATUS(cf_math_bind(&out, &f64_handler, &out_meta));
  CF_TEST_STATUS(cf_math_cpy_h2d(&a, ad, 6));
  CF_TEST_STATUS(cf_math_cpy_h2d(&b, bd, 6));
  CF_TEST_STATUS(cf_math_matmul(&out, &a, &b));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out, od, 4));
  CF_TEST_CLOSE(od[0], 58.0, 0.0001);
  CF_TEST_CLOSE(od[3], 154.0, 0.0001);
  CF_TEST_STATUS(cf_math_unbind(&a));
  CF_TEST_STATUS(cf_math_unbind(&b));
  CF_TEST_STATUS(cf_math_unbind(&out));
  CF_TEST_STATUS(cf_math_handle_destroy(&f64_handler));
  return CF_OK;
}

static cf_status test_ai_dense_forward_cpu(void)
{
  cf_math_handle_t parameter_handler = {0};
  cf_math_handle_t activation_handler = {0};
  cf_ai_dense dense = {0};
  cf_math input = {0};
  cf_math_metadata input_meta = {0};
  cf_usize input_dims[CF_MATH_MAX_RANK] = {2, 3};
  float input_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float weights_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float bias_data[2] = {0.5f, -1.0f};
  float output_data[4] = {0};

  CF_TEST_STATUS(cf_math_handle_init(&parameter_handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_POOLED | CF_MATH_MEM_ALIGNED128, CF_MATH_HANDLE_OPT_MATMUL, 1024));
  CF_TEST_STATUS(cf_math_handle_init(&activation_handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_POOLED | CF_MATH_MEM_ALIGNED128, CF_MATH_HANDLE_OPT_MATMUL | CF_MATH_HANDLE_OPT_ELEMENTWISE, 1024));
  CF_TEST_STATUS(cf_ai_dense_init(&dense, &parameter_handler, &activation_handler, 2, 3, 2, CF_AI_ACT_NONE));
  CF_TEST_STATUS(cf_math_metadata_init(&input_meta, input_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_bind(&input, &activation_handler, &input_meta));
  CF_TEST_STATUS(cf_math_cpy_h2d(&dense.weights, weights_data, 6));
  CF_TEST_STATUS(cf_math_cpy_h2d(&dense.bias, bias_data, 2));
  CF_TEST_STATUS(cf_math_cpy_h2d(&input, input_data, 6));
  CF_TEST_STATUS(cf_ai_dense_forward(&dense, &input));
  CF_TEST_STATUS(cf_math_cpy_d2h(&dense.output, output_data, 4));
  CF_TEST_CLOSE(output_data[0], 22.5f, 0.0001);
  CF_TEST_CLOSE(output_data[1], 27.0f, 0.0001);
  CF_TEST_CLOSE(output_data[2], 49.5f, 0.0001);
  CF_TEST_CLOSE(output_data[3], 63.0f, 0.0001);

  CF_TEST_STATUS(cf_math_unbind(&input));
  CF_TEST_STATUS(cf_ai_dense_destroy(&dense));
  CF_TEST_STATUS(cf_math_handle_destroy(&activation_handler));
  CF_TEST_STATUS(cf_math_handle_destroy(&parameter_handler));
  return CF_OK;
}

static cf_status test_ai_dense_activation_cpu(void)
{
  cf_math_handle_t parameter_handler = {0};
  cf_math_handle_t activation_handler = {0};
  cf_ai_dense relu_dense = {0};
  cf_ai_dense sigmoid_dense = {0};
  cf_math input = {0};
  cf_math_metadata input_meta = {0};
  cf_usize input_dims[CF_MATH_MAX_RANK] = {1, 2};
  float input_data[2] = {0.0f, 2.0f};
  float relu_weights[4] = {-2.0f, 3.0f, -4.0f, 5.0f};
  float relu_bias[2] = {1.0f, -10.0f};
  float sigmoid_weights[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  float sigmoid_bias[2] = {0.0f, 0.0f};
  float output_data[2] = {0};

  CF_TEST_STATUS(cf_math_handle_init(&parameter_handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_MATMUL, 1024));
  CF_TEST_STATUS(cf_math_handle_init(&activation_handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_MATMUL | CF_MATH_HANDLE_OPT_ELEMENTWISE, 1024));
  CF_TEST_STATUS(cf_math_metadata_init(&input_meta, input_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_bind(&input, &activation_handler, &input_meta));
  CF_TEST_STATUS(cf_math_cpy_h2d(&input, input_data, 2));

  CF_TEST_STATUS(cf_ai_dense_init(&relu_dense, &parameter_handler, &activation_handler, 1, 2, 2, CF_AI_ACT_RELU));
  CF_TEST_STATUS(cf_math_cpy_h2d(&relu_dense.weights, relu_weights, 4));
  CF_TEST_STATUS(cf_math_cpy_h2d(&relu_dense.bias, relu_bias, 2));
  CF_TEST_STATUS(cf_ai_dense_forward(&relu_dense, &input));
  CF_TEST_STATUS(cf_math_cpy_d2h(&relu_dense.output, output_data, 2));
  CF_TEST_CLOSE(output_data[0], 0.0f, 0.0001);
  CF_TEST_CLOSE(output_data[1], 0.0f, 0.0001);
  CF_TEST_STATUS(cf_ai_dense_destroy(&relu_dense));

  CF_TEST_STATUS(cf_ai_dense_init(&sigmoid_dense, &parameter_handler, &activation_handler, 1, 2, 2, CF_AI_ACT_SIGMOID));
  CF_TEST_STATUS(cf_math_cpy_h2d(&sigmoid_dense.weights, sigmoid_weights, 4));
  CF_TEST_STATUS(cf_math_cpy_h2d(&sigmoid_dense.bias, sigmoid_bias, 2));
  CF_TEST_STATUS(cf_ai_dense_forward(&sigmoid_dense, &input));
  CF_TEST_STATUS(cf_math_cpy_d2h(&sigmoid_dense.output, output_data, 2));
  CF_TEST_CLOSE(output_data[0], 0.5f, 0.0001);
  CF_TEST_CLOSE(output_data[1], 0.880797f, 0.0001);

  CF_TEST_STATUS(cf_math_unbind(&input));
  CF_TEST_STATUS(cf_ai_dense_destroy(&sigmoid_dense));
  CF_TEST_STATUS(cf_math_handle_destroy(&activation_handler));
  CF_TEST_STATUS(cf_math_handle_destroy(&parameter_handler));
  return CF_OK;
}

static cf_status test_ai_model_forward_cpu(void)
{
  cf_math_handle_t parameter_handler = {0};
  cf_math_handle_t activation_handler = {0};
  cf_ai_dense layers[2] = {0};
  cf_ai_model model = {0};
  cf_math input = {0};
  cf_math *output = CF_NULL;
  cf_math_metadata input_meta = {0};
  cf_usize input_dims[CF_MATH_MAX_RANK] = {1, 2};
  float input_data[2] = {2.0f, -1.0f};
  float w0[4] = {1.0f, -1.0f, 2.0f, 3.0f};
  float b0[2] = {0.0f, 1.0f};
  float w1[2] = {1.0f, 2.0f};
  float b1[1] = {0.5f};
  float output_data[1] = {0};

  CF_TEST_STATUS(cf_math_handle_init(&parameter_handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_MATMUL, 2048));
  CF_TEST_STATUS(cf_math_handle_init(&activation_handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_MATMUL | CF_MATH_HANDLE_OPT_ELEMENTWISE, 2048));
  CF_TEST_STATUS(cf_ai_dense_init(&layers[0], &parameter_handler, &activation_handler, 1, 2, 2, CF_AI_ACT_RELU));
  CF_TEST_STATUS(cf_ai_dense_init(&layers[1], &parameter_handler, &activation_handler, 1, 2, 1, CF_AI_ACT_NONE));
  CF_TEST_STATUS(cf_ai_model_init(&model, layers, 2, &parameter_handler, &activation_handler, CF_MATH_DEVICE_CPU));
  CF_TEST_STATUS(cf_math_metadata_init(&input_meta, input_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_bind(&input, &activation_handler, &input_meta));
  CF_TEST_STATUS(cf_math_cpy_h2d(&input, input_data, 2));
  CF_TEST_STATUS(cf_math_cpy_h2d(&layers[0].weights, w0, 4));
  CF_TEST_STATUS(cf_math_cpy_h2d(&layers[0].bias, b0, 2));
  CF_TEST_STATUS(cf_math_cpy_h2d(&layers[1].weights, w1, 2));
  CF_TEST_STATUS(cf_math_cpy_h2d(&layers[1].bias, b1, 1));
  CF_TEST_STATUS(cf_ai_model_forward(&model, &input, &output));
  CF_TEST_CHECK(output == &layers[1].output);
  CF_TEST_STATUS(cf_math_cpy_d2h(output, output_data, 1));
  CF_TEST_CLOSE(output_data[0], 0.5f, 0.0001);

  CF_TEST_STATUS(cf_math_unbind(&input));
  CF_TEST_STATUS(cf_ai_model_destroy(&model));
  CF_TEST_STATUS(cf_math_handle_destroy(&activation_handler));
  CF_TEST_STATUS(cf_math_handle_destroy(&parameter_handler));
  return CF_OK;
}

static cf_status test_ai_loss_cpu(void)
{
  cf_math_handle_t handler = {0};
  cf_math_metadata vector_meta = {0};
  cf_math_metadata scalar_meta = {0};
  cf_math prediction = {0};
  cf_math target = {0};
  cf_math out = {0};
  cf_usize vector_dims[CF_MATH_MAX_RANK] = {4};
  cf_usize scalar_dims[CF_MATH_MAX_RANK] = {1};
  float pred_mse[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float target_mse[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float pred_bce[4] = {0.25f, 0.75f, 0.75f, 0.25f};
  float target_bce[4] = {0.0f, 1.0f, 1.0f, 0.0f};
  float out_data[1] = {0};

  CF_TEST_STATUS(cf_math_handle_init(&handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_REDUCTION, 1024));
  CF_TEST_STATUS(cf_math_metadata_init(&vector_meta, vector_dims, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_metadata_init(&scalar_meta, scalar_dims, 1, CF_MATH_SHAPE_SCALAR, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_bind(&prediction, &handler, &vector_meta));
  CF_TEST_STATUS(cf_math_bind(&target, &handler, &vector_meta));
  CF_TEST_STATUS(cf_math_bind(&out, &handler, &scalar_meta));

  CF_TEST_STATUS(cf_math_cpy_h2d(&prediction, pred_mse, 4));
  CF_TEST_STATUS(cf_math_cpy_h2d(&target, target_mse, 4));
  CF_TEST_STATUS(cf_ai_loss_forward(CF_AI_LOSS_MSE, &out, &prediction, &target));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out, out_data, 1));
  CF_TEST_CLOSE(out_data[0], 3.5f, 0.0001);

  CF_TEST_STATUS(cf_math_cpy_h2d(&prediction, pred_bce, 4));
  CF_TEST_STATUS(cf_math_cpy_h2d(&target, target_bce, 4));
  CF_TEST_STATUS(cf_ai_loss_forward(CF_AI_LOSS_BINARY_CROSS_ENTROPY, &out, &prediction, &target));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out, out_data, 1));
  CF_TEST_CLOSE(out_data[0], 0.287682f, 0.0001);

  CF_TEST_STATUS(cf_math_unbind(&prediction));
  CF_TEST_STATUS(cf_math_unbind(&target));
  CF_TEST_STATUS(cf_math_unbind(&out));
  CF_TEST_STATUS(cf_math_handle_destroy(&handler));
  return CF_OK;
}

static cf_status test_ai_invalid_and_gradient_boundary_cpu(void)
{
  cf_math_handle_t parameter_handler = {0};
  cf_math_handle_t activation_handler = {0};
  cf_ai_dense dense = {0};
  cf_math input = {0};
  cf_math_metadata input_meta = {0};
  cf_usize input_dims[CF_MATH_MAX_RANK] = {1, 3};

  CF_TEST_STATUS(cf_math_handle_init(&parameter_handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_MATMUL, 1024));
  CF_TEST_STATUS(cf_math_handle_init(&activation_handler, CF_NULL, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CPU, CF_MATH_MEM_DEFAULT, CF_MATH_HANDLE_OPT_MATMUL, 1024));
  CF_TEST_STATUS(cf_ai_dense_init(&dense, &parameter_handler, &activation_handler, 1, 2, 2, CF_AI_ACT_NONE));
  CF_TEST_STATUS(cf_math_metadata_init(&input_meta, input_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_bind(&input, &activation_handler, &input_meta));
  CF_TEST_CHECK(cf_ai_dense_forward(&dense, &input) == CF_ERR_INVALID);
  CF_TEST_CHECK(cf_ai_dense_backward(&dense, &input, &dense.output) == CF_ERR_UNSUPPORTED);
  CF_TEST_CHECK(cf_ai_loss_backward(CF_AI_LOSS_MSE, &input, &dense.output, &dense.output) == CF_ERR_UNSUPPORTED);

  CF_TEST_STATUS(cf_math_unbind(&input));
  CF_TEST_STATUS(cf_ai_dense_destroy(&dense));
  CF_TEST_STATUS(cf_math_handle_destroy(&activation_handler));
  CF_TEST_STATUS(cf_math_handle_destroy(&parameter_handler));
  return CF_OK;
}

#if defined(CF_CUDA_AVAILABLE)
static cf_status test_math_cuda_ops_f32(void)
{
  cf_math_cuda_context ctx = {0};
  cf_math_handle_t handler = {0};
  cf_math_metadata metadata = {0};
  cf_math_metadata scalar_meta = {0};
  cf_math_metadata a_meta = {0};
  cf_math_metadata b_meta = {0};
  cf_math_metadata out_meta = {0};
  cf_math a = {0};
  cf_math b = {0};
  cf_math out_view = {0};
  cf_math scalar_out = {0};
  cf_math ma = {0};
  cf_math mb = {0};
  cf_math mout = {0};
  cf_math_handle_t f64_handler = {0};
  cf_math da = {0};
  cf_math db = {0};
  cf_math dout = {0};
  cf_usize dims[CF_MATH_MAX_RANK] = {4};
  cf_usize dim1[CF_MATH_MAX_RANK] = {1};
  cf_usize a_dims[CF_MATH_MAX_RANK] = {2, 3};
  cf_usize b_dims[CF_MATH_MAX_RANK] = {3, 2};
  cf_usize out_dims[CF_MATH_MAX_RANK] = {2, 2};
  float a_data[4] = {8.0f, 12.0f, 20.0f, 24.0f};
  float b_data[4] = {2.0f, 3.0f, 4.0f, 6.0f};
  float unary_data[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
  float out[4] = {0};
  float scalar_value[1] = {0};
  float af[6] = {1, 2, 3, 4, 5, 6};
  float bf[6] = {7, 8, 9, 10, 11, 12};
  float mf[4] = {0};
  double ad[6] = {1, 2, 3, 4, 5, 6};
  double bd[6] = {7, 8, 9, 10, 11, 12};
  double md[4] = {0};

  CF_TEST_STATUS(cf_math_cuda_context_init(&ctx, 0, 0));
  CF_TEST_STATUS(cf_math_handle_init(&handler, &ctx, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CUDA, CF_MATH_MEM_POOLED, CF_MATH_HANDLE_OPT_ELEMENTWISE, 512));
  CF_TEST_STATUS(cf_math_handle_init(&f64_handler, &ctx, CF_MATH_DTYPE_F64, CF_MATH_DEVICE_CUDA, CF_MATH_MEM_POOLED, CF_MATH_HANDLE_OPT_MATMUL, 1024));
  CF_TEST_STATUS(cf_math_metadata_init(&metadata, dims, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_metadata_init(&scalar_meta, dim1, 1, CF_MATH_SHAPE_SCALAR, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_metadata_init(&a_meta, a_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_metadata_init(&b_meta, b_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_metadata_init(&out_meta, out_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_bind(&a, &handler, &metadata));
  CF_TEST_STATUS(cf_math_bind(&b, &handler, &metadata));
  CF_TEST_STATUS(cf_math_bind(&out_view, &handler, &metadata));
  CF_TEST_STATUS(cf_math_bind(&scalar_out, &handler, &scalar_meta));
  CF_TEST_STATUS(cf_math_bind(&ma, &handler, &a_meta));
  CF_TEST_STATUS(cf_math_bind(&mb, &handler, &b_meta));
  CF_TEST_STATUS(cf_math_bind(&mout, &handler, &out_meta));
  CF_TEST_STATUS(cf_math_bind(&da, &f64_handler, &a_meta));
  CF_TEST_STATUS(cf_math_bind(&db, &f64_handler, &b_meta));
  CF_TEST_STATUS(cf_math_bind(&dout, &f64_handler, &out_meta));

  CF_TEST_STATUS(cf_math_cpy_h2d(&a, a_data, 4));
  CF_TEST_STATUS(cf_math_cpy_h2d(&b, b_data, 4));
  CF_TEST_STATUS(cf_math_op(CF_MATH_OP_ADD, &a, &b));
  CF_TEST_STATUS(cf_math_cpy_d2h(&a, out, 4));
  CF_TEST_CHECK(out[0] == 10.0f && out[1] == 15.0f && out[2] == 24.0f && out[3] == 30.0f);

  CF_TEST_STATUS(cf_math_cpy_h2d(&a, unary_data, 4));
  CF_TEST_STATUS(cf_math_unary_out(CF_MATH_OP_RELU, &out_view, &a));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out_view, out, 4));
  CF_TEST_CHECK(out[0] == 0.0f && out[2] == 1.0f && out[3] == 2.0f);

  CF_TEST_STATUS(cf_math_scalar(CF_MATH_OP_MUL, &out_view, 3.0));
  CF_TEST_STATUS(cf_math_cpy_d2h(&out_view, out, 4));
  CF_TEST_CHECK(out[3] == 6.0f);

  CF_TEST_STATUS(cf_math_reduce_sum(&scalar_out, &out_view));
  CF_TEST_STATUS(cf_math_cpy_d2h(&scalar_out, scalar_value, 1));
  CF_TEST_CHECK(scalar_value[0] == 9.0f);

  CF_TEST_STATUS(cf_math_cpy_h2d(&ma, af, 6));
  CF_TEST_STATUS(cf_math_cpy_h2d(&mb, bf, 6));
  CF_TEST_STATUS(cf_math_matmul(&mout, &ma, &mb));
  CF_TEST_STATUS(cf_math_cpy_d2h(&mout, mf, 4));
  CF_TEST_CLOSE(mf[0], 58.0f, 0.0001);
  CF_TEST_CLOSE(mf[3], 154.0f, 0.0001);

  CF_TEST_STATUS(cf_math_cpy_h2d(&da, ad, 6));
  CF_TEST_STATUS(cf_math_cpy_h2d(&db, bd, 6));
  CF_TEST_STATUS(cf_math_matmul(&dout, &da, &db));
  CF_TEST_STATUS(cf_math_cpy_d2h(&dout, md, 4));
  CF_TEST_CLOSE(md[0], 58.0, 0.0001);
  CF_TEST_CLOSE(md[3], 154.0, 0.0001);

  CF_TEST_STATUS(cf_math_unbind(&a));
  CF_TEST_STATUS(cf_math_unbind(&b));
  CF_TEST_STATUS(cf_math_unbind(&out_view));
  CF_TEST_STATUS(cf_math_unbind(&scalar_out));
  CF_TEST_STATUS(cf_math_unbind(&ma));
  CF_TEST_STATUS(cf_math_unbind(&mb));
  CF_TEST_STATUS(cf_math_unbind(&mout));
  CF_TEST_STATUS(cf_math_unbind(&da));
  CF_TEST_STATUS(cf_math_unbind(&db));
  CF_TEST_STATUS(cf_math_unbind(&dout));
  CF_TEST_STATUS(cf_math_handle_destroy(&handler));
  CF_TEST_STATUS(cf_math_handle_destroy(&f64_handler));
  CF_TEST_STATUS(cf_math_cuda_context_destroy(&ctx));
  return CF_OK;
}

static cf_status test_ai_cuda_core(void)
{
  cf_math_cuda_context ctx = {0};
  cf_math_handle_t parameter_handler = {0};
  cf_math_handle_t activation_handler = {0};
  cf_ai_dense dense = {0};
  cf_ai_dense layers[2] = {0};
  cf_ai_model model = {0};
  cf_math input = {0};
  cf_math model_input = {0};
  cf_math prediction = {0};
  cf_math target = {0};
  cf_math loss_out = {0};
  cf_math *model_output = CF_NULL;
  cf_math_metadata input_meta = {0};
  cf_math_metadata model_input_meta = {0};
  cf_math_metadata loss_vec_meta = {0};
  cf_math_metadata loss_scalar_meta = {0};
  cf_usize input_dims[CF_MATH_MAX_RANK] = {2, 3};
  cf_usize model_input_dims[CF_MATH_MAX_RANK] = {1, 2};
  cf_usize loss_vec_dims[CF_MATH_MAX_RANK] = {4};
  cf_usize loss_scalar_dims[CF_MATH_MAX_RANK] = {1};
  float input_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float weights_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float bias_data[2] = {0.5f, -1.0f};
  float dense_output[4] = {0};
  float model_input_data[2] = {2.0f, -1.0f};
  float w0[4] = {1.0f, -1.0f, 2.0f, 3.0f};
  float b0[2] = {0.0f, 1.0f};
  float w1[2] = {1.0f, 2.0f};
  float b1[1] = {0.5f};
  float model_output_data[1] = {0};
  float pred_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float target_data[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float loss_data[1] = {0};

  CF_TEST_STATUS(cf_math_cuda_context_init(&ctx, 0, 0));
  CF_TEST_STATUS(cf_math_handle_init(&parameter_handler, &ctx, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CUDA, CF_MATH_MEM_POOLED, CF_MATH_HANDLE_OPT_MATMUL, 4096));
  CF_TEST_STATUS(cf_math_handle_init(&activation_handler, &ctx, CF_MATH_DTYPE_F32, CF_MATH_DEVICE_CUDA, CF_MATH_MEM_POOLED, CF_MATH_HANDLE_OPT_MATMUL | CF_MATH_HANDLE_OPT_ELEMENTWISE | CF_MATH_HANDLE_OPT_REDUCTION, 4096));

  CF_TEST_STATUS(cf_ai_dense_init(&dense, &parameter_handler, &activation_handler, 2, 3, 2, CF_AI_ACT_NONE));
  CF_TEST_STATUS(cf_math_metadata_init(&input_meta, input_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_bind(&input, &activation_handler, &input_meta));
  CF_TEST_STATUS(cf_math_cpy_h2d(&dense.weights, weights_data, 6));
  CF_TEST_STATUS(cf_math_cpy_h2d(&dense.bias, bias_data, 2));
  CF_TEST_STATUS(cf_math_cpy_h2d(&input, input_data, 6));
  CF_TEST_STATUS(cf_ai_dense_forward(&dense, &input));
  CF_TEST_STATUS(cf_math_cpy_d2h(&dense.output, dense_output, 4));
  CF_TEST_CLOSE(dense_output[0], 22.5f, 0.0001);
  CF_TEST_CLOSE(dense_output[3], 63.0f, 0.0001);
  CF_TEST_STATUS(cf_math_unbind(&input));
  CF_TEST_STATUS(cf_ai_dense_destroy(&dense));

  CF_TEST_STATUS(cf_ai_dense_init(&layers[0], &parameter_handler, &activation_handler, 1, 2, 2, CF_AI_ACT_RELU));
  CF_TEST_STATUS(cf_ai_dense_init(&layers[1], &parameter_handler, &activation_handler, 1, 2, 1, CF_AI_ACT_NONE));
  CF_TEST_STATUS(cf_ai_model_init(&model, layers, 2, &parameter_handler, &activation_handler, CF_MATH_DEVICE_CUDA));
  CF_TEST_STATUS(cf_math_metadata_init(&model_input_meta, model_input_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_bind(&model_input, &activation_handler, &model_input_meta));
  CF_TEST_STATUS(cf_math_cpy_h2d(&model_input, model_input_data, 2));
  CF_TEST_STATUS(cf_math_cpy_h2d(&layers[0].weights, w0, 4));
  CF_TEST_STATUS(cf_math_cpy_h2d(&layers[0].bias, b0, 2));
  CF_TEST_STATUS(cf_math_cpy_h2d(&layers[1].weights, w1, 2));
  CF_TEST_STATUS(cf_math_cpy_h2d(&layers[1].bias, b1, 1));
  CF_TEST_STATUS(cf_ai_model_forward(&model, &model_input, &model_output));
  CF_TEST_STATUS(cf_math_cpy_d2h(model_output, model_output_data, 1));
  CF_TEST_CLOSE(model_output_data[0], 0.5f, 0.0001);
  CF_TEST_STATUS(cf_math_unbind(&model_input));
  CF_TEST_STATUS(cf_ai_model_destroy(&model));

  CF_TEST_STATUS(cf_math_metadata_init(&loss_vec_meta, loss_vec_dims, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_metadata_init(&loss_scalar_meta, loss_scalar_dims, 1, CF_MATH_SHAPE_SCALAR, CF_MATH_LAYOUT_ROW_MAJOR));
  CF_TEST_STATUS(cf_math_bind(&prediction, &activation_handler, &loss_vec_meta));
  CF_TEST_STATUS(cf_math_bind(&target, &activation_handler, &loss_vec_meta));
  CF_TEST_STATUS(cf_math_bind(&loss_out, &activation_handler, &loss_scalar_meta));
  CF_TEST_STATUS(cf_math_cpy_h2d(&prediction, pred_data, 4));
  CF_TEST_STATUS(cf_math_cpy_h2d(&target, target_data, 4));
  CF_TEST_STATUS(cf_ai_loss_forward(CF_AI_LOSS_MSE, &loss_out, &prediction, &target));
  CF_TEST_STATUS(cf_math_cpy_d2h(&loss_out, loss_data, 1));
  CF_TEST_CLOSE(loss_data[0], 3.5f, 0.0001);
  CF_TEST_STATUS(cf_math_unbind(&prediction));
  CF_TEST_STATUS(cf_math_unbind(&target));
  CF_TEST_STATUS(cf_math_unbind(&loss_out));

  CF_TEST_STATUS(cf_math_handle_destroy(&activation_handler));
  CF_TEST_STATUS(cf_math_handle_destroy(&parameter_handler));
  CF_TEST_STATUS(cf_math_cuda_context_destroy(&ctx));
  return CF_OK;
}
#endif

int main(void)
{
  CF_TEST_STATUS(test_allocator_arena());
  CF_TEST_STATUS(test_allocator_pool());
  CF_TEST_STATUS(test_math_cpu_pooled_storage());
  CF_TEST_STATUS(test_math_cpu_storage_flags());
  CF_TEST_STATUS(test_math_cpu_ops_f32());
  CF_TEST_STATUS(test_math_cpu_ops_f64_i32());
  CF_TEST_STATUS(test_math_cpu_ops_unsupported_dtype());
  CF_TEST_STATUS(test_math_op_check_and_out());
  CF_TEST_STATUS(test_math_cpu_unary_scalar_reduce());
  CF_TEST_STATUS(test_math_cpu_matmul());
  CF_TEST_STATUS(test_ai_dense_forward_cpu());
  CF_TEST_STATUS(test_ai_dense_activation_cpu());
  CF_TEST_STATUS(test_ai_model_forward_cpu());
  CF_TEST_STATUS(test_ai_loss_cpu());
  CF_TEST_STATUS(test_ai_invalid_and_gradient_boundary_cpu());
#if defined(CF_CUDA_AVAILABLE)
  CF_TEST_STATUS(test_math_cuda_ops_f32());
  CF_TEST_STATUS(test_ai_cuda_core());
#endif
  CF_TEST_CHECK(cf_math_rotl8(0x12U, 4U) == 0x21U);

  printf("allocator arena: ok\n");
  printf("allocator pool: ok\n");
  printf("math cpu pooled storage: ok\n");
  printf("math cpu storage flags: ok\n");
  printf("math cpu ops: ok\n");
  printf("math cpu op families: ok\n");
  printf("math cpu matmul: ok\n");
  printf("ai cpu dense forward: ok\n");
  printf("ai cpu model/loss: ok\n");
  printf("ai gradient boundary: ok\n");
#if defined(CF_CUDA_AVAILABLE)
  printf("math cuda ops: ok\n");
  printf("ai cuda core: ok\n");
#endif
  printf("basic helper: rotl8(0x12, 4) = 0x%02x\n", cf_math_rotl8(0x12U, 4U));
  return 0;
}
