/*
 * CF Framework
 * Copyright (C) 2026 Orion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "AI/cf_model.h"

#include <math.h>
#include <string.h>

static cf_bool cf_ai_math_is_bound(const cf_math *x)
{
  return x != CF_NULL && x->handler != CF_NULL && x->metadata != CF_NULL;
}

static cf_bool cf_ai_is_float_dtype(cf_math_dtype dtype)
{
  return dtype == CF_MATH_DTYPE_F32 || dtype == CF_MATH_DTYPE_F64;
}

static cf_status cf_ai_resolve_ptr(const cf_math *x, void **ptr)
{
  if(x == CF_NULL || ptr == CF_NULL) return CF_ERR_NULL;
  *ptr = CF_NULL;
  if(x->handler == CF_NULL || x->metadata == CF_NULL) return CF_ERR_STATE;
  if(x->byte_size == 0) return CF_OK;
  if(x->handler->storage.allocator.backend == CF_NULL) return CF_ERR_STATE;
  if(x->byte_offset > x->handler->storage.arena.capacity) return CF_ERR_BOUNDS;
  if(x->byte_size > x->handler->storage.arena.capacity - x->byte_offset) return CF_ERR_BOUNDS;

  *ptr = (void *)((cf_u8 *)x->handler->storage.allocator.backend + x->byte_offset);
  return CF_OK;
}

static cf_status cf_ai_activation_to_math(cf_ai_activation activation, cf_math_op_kind *op)
{
  if(op == CF_NULL) return CF_ERR_NULL;
  switch(activation)
  {
    case CF_AI_ACT_NONE: *op = CF_MATH_OP_NONE; return CF_OK;
    case CF_AI_ACT_RELU: *op = CF_MATH_OP_RELU; return CF_OK;
    case CF_AI_ACT_GELU: *op = CF_MATH_OP_GELU; return CF_OK;
    case CF_AI_ACT_SIGMOID: *op = CF_MATH_OP_SIGMOID; return CF_OK;
    case CF_AI_ACT_TANH: *op = CF_MATH_OP_TANH; return CF_OK;
  }
  return CF_ERR_UNSUPPORTED;
}

static cf_status cf_ai_check_dense_input(const cf_ai_dense *layer, const cf_math *input)
{
  if(layer == CF_NULL || input == CF_NULL) return CF_ERR_NULL;
  if(cf_ai_math_is_bound(input) == CF_FALSE || cf_ai_math_is_bound(&layer->weights) == CF_FALSE || cf_ai_math_is_bound(&layer->bias) == CF_FALSE || cf_ai_math_is_bound(&layer->output) == CF_FALSE)
    return CF_ERR_STATE;
  if(input->handler->storage.device != layer->weights.handler->storage.device || input->handler->storage.device != layer->bias.handler->storage.device || input->handler->storage.device != layer->output.handler->storage.device)
    return CF_ERR_INVALID;
  if(input->handler->storage.dtype != layer->weights.handler->storage.dtype || input->handler->storage.dtype != layer->bias.handler->storage.dtype || input->handler->storage.dtype != layer->output.handler->storage.dtype)
    return CF_ERR_INVALID;
  if(cf_ai_is_float_dtype(input->handler->storage.dtype) == CF_FALSE) return CF_ERR_UNSUPPORTED;
  if(input->metadata->rank != 2 || layer->weights.metadata->rank != 2 || layer->bias.metadata->rank != 1 || layer->output.metadata->rank != 2)
    return CF_ERR_INVALID;
  if(input->metadata->layout != CF_MATH_LAYOUT_ROW_MAJOR || layer->weights.metadata->layout != CF_MATH_LAYOUT_ROW_MAJOR || layer->output.metadata->layout != CF_MATH_LAYOUT_ROW_MAJOR)
    return CF_ERR_UNSUPPORTED;
  if(input->metadata->dim[1] != layer->weights.metadata->dim[0]) return CF_ERR_INVALID;
  if(layer->weights.metadata->dim[1] != layer->bias.metadata->dim[0]) return CF_ERR_INVALID;
  if(layer->output.metadata->dim[0] != input->metadata->dim[0]) return CF_ERR_INVALID;
  if(layer->output.metadata->dim[1] != layer->weights.metadata->dim[1]) return CF_ERR_INVALID;
  return CF_OK;
}

static cf_status cf_ai_check_loss_views(cf_math *out, const cf_math *prediction, const cf_math *target)
{
  if(out == CF_NULL || prediction == CF_NULL || target == CF_NULL) return CF_ERR_NULL;
  if(cf_ai_math_is_bound(out) == CF_FALSE || cf_ai_math_is_bound(prediction) == CF_FALSE || cf_ai_math_is_bound(target) == CF_FALSE)
    return CF_ERR_STATE;
  if(out->handler->storage.device != prediction->handler->storage.device || prediction->handler->storage.device != target->handler->storage.device)
    return CF_ERR_INVALID;
  if(out->handler->storage.dtype != prediction->handler->storage.dtype || prediction->handler->storage.dtype != target->handler->storage.dtype)
    return CF_ERR_INVALID;
  if(cf_ai_is_float_dtype(prediction->handler->storage.dtype) == CF_FALSE) return CF_ERR_UNSUPPORTED;
  if(prediction->metadata->len != target->metadata->len) return CF_ERR_INVALID;
  if(out->metadata->len != 1 || prediction->metadata->len == 0) return CF_ERR_INVALID;
  return CF_OK;
}

static cf_status cf_ai_bias_add_cpu(cf_math_dtype dtype, void *out_ptr, const void *bias_ptr, cf_usize batch, cf_usize out_features)
{
  if(dtype == CF_MATH_DTYPE_F32)
  {
    float *out = (float *)out_ptr;
    const float *bias = (const float *)bias_ptr;
    for(cf_usize row = 0; row < batch; ++row)
    {
      for(cf_usize col = 0; col < out_features; ++col)
      {
        out[row * out_features + col] += bias[col];
      }
    }
    return CF_OK;
  }
  if(dtype == CF_MATH_DTYPE_F64)
  {
    double *out = (double *)out_ptr;
    const double *bias = (const double *)bias_ptr;
    for(cf_usize row = 0; row < batch; ++row)
    {
      for(cf_usize col = 0; col < out_features; ++col)
      {
        out[row * out_features + col] += bias[col];
      }
    }
    return CF_OK;
  }
  return CF_ERR_UNSUPPORTED;
}

static cf_status cf_ai_loss_cpu(cf_ai_loss_kind loss, cf_math_dtype dtype, void *out_ptr, const void *prediction_ptr, const void *target_ptr, cf_usize len)
{
  double total = 0.0;

  if(loss != CF_AI_LOSS_MSE && loss != CF_AI_LOSS_BINARY_CROSS_ENTROPY) return CF_ERR_UNSUPPORTED;

  if(dtype == CF_MATH_DTYPE_F32)
  {
    const float *prediction = (const float *)prediction_ptr;
    const float *target = (const float *)target_ptr;
    float *out = (float *)out_ptr;
    for(cf_usize i = 0; i < len; ++i)
    {
      double p = (double)prediction[i];
      double t = (double)target[i];
      if(loss == CF_AI_LOSS_MSE)
      {
        double d = p - t;
        total += d * d;
      }
      else
      {
        if(p < 1.0e-7) p = 1.0e-7;
        if(p > 1.0 - 1.0e-7) p = 1.0 - 1.0e-7;
        total += -(t * log(p) + (1.0 - t) * log(1.0 - p));
      }
    }
    out[0] = (float)(total / (double)len);
    return CF_OK;
  }

  if(dtype == CF_MATH_DTYPE_F64)
  {
    const double *prediction = (const double *)prediction_ptr;
    const double *target = (const double *)target_ptr;
    double *out = (double *)out_ptr;
    for(cf_usize i = 0; i < len; ++i)
    {
      double p = prediction[i];
      double t = target[i];
      if(loss == CF_AI_LOSS_MSE)
      {
        double d = p - t;
        total += d * d;
      }
      else
      {
        if(p < 1.0e-12) p = 1.0e-12;
        if(p > 1.0 - 1.0e-12) p = 1.0 - 1.0e-12;
        total += -(t * log(p) + (1.0 - t) * log(1.0 - p));
      }
    }
    out[0] = total / (double)len;
    return CF_OK;
  }

  return CF_ERR_UNSUPPORTED;
}

#if defined(CF_CUDA_AVAILABLE)
static cf_status cf_ai_cuda_sync(const cf_math_handle_t *handler)
{
  if(handler != CF_NULL && handler->cuda_ctx != CF_NULL && handler->cuda_ctx->stream != CF_NULL)
    return cudaStreamSynchronize(handler->cuda_ctx->stream) == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
  return cudaDeviceSynchronize() == cudaSuccess ? CF_OK : CF_ERR_CUDA_SYNC;
}

static __global__ void cf_ai_bias_add_kernel_f32(float *out, const float *bias, cf_usize batch, cf_usize out_features)
{
  cf_usize i = (cf_usize)blockIdx.x * (cf_usize)blockDim.x + (cf_usize)threadIdx.x;
  cf_usize len = batch * out_features;
  if(i < len) out[i] += bias[i % out_features];
}

static __global__ void cf_ai_bias_add_kernel_f64(double *out, const double *bias, cf_usize batch, cf_usize out_features)
{
  cf_usize i = (cf_usize)blockIdx.x * (cf_usize)blockDim.x + (cf_usize)threadIdx.x;
  cf_usize len = batch * out_features;
  if(i < len) out[i] += bias[i % out_features];
}

static __global__ void cf_ai_loss_kernel_f32(cf_ai_loss_kind loss, float *out, const float *prediction, const float *target, cf_usize len)
{
  double total = 0.0;
  for(cf_usize i = 0; i < len; ++i)
  {
    double p = (double)prediction[i];
    double t = (double)target[i];
    if(loss == CF_AI_LOSS_MSE)
    {
      double d = p - t;
      total += d * d;
    }
    else
    {
      if(p < 1.0e-7) p = 1.0e-7;
      if(p > 1.0 - 1.0e-7) p = 1.0 - 1.0e-7;
      total += -(t * log(p) + (1.0 - t) * log(1.0 - p));
    }
  }
  out[0] = (float)(total / (double)len);
}

static __global__ void cf_ai_loss_kernel_f64(cf_ai_loss_kind loss, double *out, const double *prediction, const double *target, cf_usize len)
{
  double total = 0.0;
  for(cf_usize i = 0; i < len; ++i)
  {
    double p = prediction[i];
    double t = target[i];
    if(loss == CF_AI_LOSS_MSE)
    {
      double d = p - t;
      total += d * d;
    }
    else
    {
      if(p < 1.0e-12) p = 1.0e-12;
      if(p > 1.0 - 1.0e-12) p = 1.0 - 1.0e-12;
      total += -(t * log(p) + (1.0 - t) * log(1.0 - p));
    }
  }
  out[0] = total / (double)len;
}
#endif

static cf_status cf_ai_bias_add(cf_ai_dense *layer)
{
  void *out_ptr = CF_NULL;
  void *bias_ptr = CF_NULL;
  cf_status status = CF_OK;
  cf_math_dtype dtype = layer->output.handler->storage.dtype;
  cf_math_device device = layer->output.handler->storage.device;
  cf_usize batch = layer->output.metadata->dim[0];
  cf_usize out_features = layer->output.metadata->dim[1];

  status = cf_ai_resolve_ptr(&layer->output, &out_ptr);
  if(status != CF_OK) return status;
  status = cf_ai_resolve_ptr(&layer->bias, &bias_ptr);
  if(status != CF_OK) return status;
  if(out_ptr == CF_NULL || bias_ptr == CF_NULL) return CF_ERR_STATE;

  if(device == CF_MATH_DEVICE_CPU || (layer->output.handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0)
    return cf_ai_bias_add_cpu(dtype, out_ptr, bias_ptr, batch, out_features);

#if defined(CF_CUDA_AVAILABLE)
  {
    const cf_usize len = batch * out_features;
    const unsigned int block = 256U;
    const unsigned int grid = (unsigned int)((len + block - 1U) / block);
    cudaStream_t stream = layer->output.handler->cuda_ctx != CF_NULL ? layer->output.handler->cuda_ctx->stream : CF_NULL;
    if(dtype == CF_MATH_DTYPE_F32)
      cf_ai_bias_add_kernel_f32<<<grid, block, 0, stream>>>((float *)out_ptr, (const float *)bias_ptr, batch, out_features);
    else if(dtype == CF_MATH_DTYPE_F64)
      cf_ai_bias_add_kernel_f64<<<grid, block, 0, stream>>>((double *)out_ptr, (const double *)bias_ptr, batch, out_features);
    else
      return CF_ERR_UNSUPPORTED;
    if(cudaGetLastError() != cudaSuccess) return CF_ERR_CUDA_LAUNCH;
    return cf_ai_cuda_sync(layer->output.handler);
  }
#else
  return CF_ERR_UNSUPPORTED;
#endif
}

cf_status cf_ai_dense_init(cf_ai_dense *layer, cf_math_handle_t *parameter_handler, cf_math_handle_t *activation_handler, cf_usize batch, cf_usize in_features, cf_usize out_features, cf_ai_activation activation)
{
  cf_usize weights_dims[CF_MATH_MAX_RANK] = {0};
  cf_usize bias_dims[CF_MATH_MAX_RANK] = {0};
  cf_usize output_dims[CF_MATH_MAX_RANK] = {0};
  cf_status status = CF_OK;

  if(layer == CF_NULL || parameter_handler == CF_NULL || activation_handler == CF_NULL) return CF_ERR_NULL;
  if(batch == 0 || in_features == 0 || out_features == 0) return CF_ERR_INVALID;
  if(parameter_handler->storage.device != activation_handler->storage.device) return CF_ERR_INVALID;
  if(parameter_handler->storage.dtype != activation_handler->storage.dtype) return CF_ERR_INVALID;
  if(cf_ai_is_float_dtype(parameter_handler->storage.dtype) == CF_FALSE) return CF_ERR_UNSUPPORTED;

  memset(layer, 0, sizeof(*layer));
  layer->activation = activation;
  weights_dims[0] = in_features;
  weights_dims[1] = out_features;
  bias_dims[0] = out_features;
  output_dims[0] = batch;
  output_dims[1] = out_features;

  status = cf_math_metadata_init(&layer->weights_meta, weights_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR);
  if(status != CF_OK) return status;
  status = cf_math_metadata_init(&layer->bias_meta, bias_dims, 1, CF_MATH_SHAPE_VECTOR, CF_MATH_LAYOUT_ROW_MAJOR);
  if(status != CF_OK) return status;
  status = cf_math_metadata_init(&layer->output_meta, output_dims, 2, CF_MATH_SHAPE_MATRIX, CF_MATH_LAYOUT_ROW_MAJOR);
  if(status != CF_OK) return status;

  status = cf_math_bind(&layer->weights, parameter_handler, &layer->weights_meta);
  if(status != CF_OK) return status;
  status = cf_math_bind(&layer->bias, parameter_handler, &layer->bias_meta);
  if(status != CF_OK)
  {
    CF_UNUSED(cf_math_unbind(&layer->weights));
    return status;
  }
  status = cf_math_bind(&layer->output, activation_handler, &layer->output_meta);
  if(status != CF_OK)
  {
    CF_UNUSED(cf_math_unbind(&layer->bias));
    CF_UNUSED(cf_math_unbind(&layer->weights));
    return status;
  }

  return CF_OK;
}

cf_status cf_ai_dense_forward(cf_ai_dense *layer, const cf_math *input)
{
  cf_math_op_kind activation_op = CF_MATH_OP_NONE;
  cf_status status = CF_OK;

  status = cf_ai_check_dense_input(layer, input);
  if(status != CF_OK) return status;
  status = cf_ai_activation_to_math(layer->activation, &activation_op);
  if(status != CF_OK) return status;

  status = cf_math_matmul(&layer->output, input, &layer->weights);
  if(status != CF_OK) return status;
  status = cf_ai_bias_add(layer);
  if(status != CF_OK) return status;
  if(activation_op != CF_MATH_OP_NONE) return cf_math_unary(activation_op, &layer->output);
  return CF_OK;
}

cf_status cf_ai_dense_destroy(cf_ai_dense *layer)
{
  cf_status status = CF_OK;
  cf_status cleanup_status = CF_OK;

  if(layer == CF_NULL) return CF_ERR_NULL;
  if(layer->weights.handler != CF_NULL)
  {
    cleanup_status = cf_math_unbind(&layer->weights);
    if(status == CF_OK) status = cleanup_status;
  }
  if(layer->bias.handler != CF_NULL)
  {
    cleanup_status = cf_math_unbind(&layer->bias);
    if(status == CF_OK) status = cleanup_status;
  }
  if(layer->output.handler != CF_NULL)
  {
    cleanup_status = cf_math_unbind(&layer->output);
    if(status == CF_OK) status = cleanup_status;
  }

  memset(layer, 0, sizeof(*layer));
  return status;
}

cf_status cf_ai_model_init(cf_ai_model *model, cf_ai_dense *layers, cf_usize layer_count, cf_math_handle_t *parameter_handler, cf_math_handle_t *activation_handler, cf_math_device device)
{
  if(model == CF_NULL || layers == CF_NULL || parameter_handler == CF_NULL || activation_handler == CF_NULL) return CF_ERR_NULL;
  if(layer_count == 0) return CF_ERR_INVALID;
  if(parameter_handler->storage.device != device || activation_handler->storage.device != device) return CF_ERR_INVALID;

  model->layers = layers;
  model->layer_count = layer_count;
  model->parameter_handler = parameter_handler;
  model->activation_handler = activation_handler;
  model->device = device;
  return CF_OK;
}

cf_status cf_ai_model_forward(cf_ai_model *model, const cf_math *input, cf_math **output)
{
  const cf_math *current = input;
  cf_status status = CF_OK;

  if(model == CF_NULL || input == CF_NULL || output == CF_NULL) return CF_ERR_NULL;
  if(model->layers == CF_NULL || model->layer_count == 0) return CF_ERR_STATE;
  if(cf_ai_math_is_bound(input) == CF_FALSE) return CF_ERR_STATE;
  if(input->handler->storage.device != model->device) return CF_ERR_INVALID;

  for(cf_usize i = 0; i < model->layer_count; ++i)
  {
    status = cf_ai_dense_forward(&model->layers[i], current);
    if(status != CF_OK) return status;
    current = &model->layers[i].output;
  }

  *output = &model->layers[model->layer_count - 1U].output;
  return CF_OK;
}

cf_status cf_ai_model_destroy(cf_ai_model *model)
{
  cf_status status = CF_OK;
  cf_status cleanup_status = CF_OK;

  if(model == CF_NULL) return CF_ERR_NULL;
  if(model->layers != CF_NULL)
  {
    for(cf_usize i = 0; i < model->layer_count; ++i)
    {
      cleanup_status = cf_ai_dense_destroy(&model->layers[i]);
      if(status == CF_OK) status = cleanup_status;
    }
  }
  memset(model, 0, sizeof(*model));
  return status;
}

cf_status cf_ai_loss_forward(cf_ai_loss_kind loss, cf_math *out, const cf_math *prediction, const cf_math *target)
{
  void *out_ptr = CF_NULL;
  void *prediction_ptr = CF_NULL;
  void *target_ptr = CF_NULL;
  cf_status status = CF_OK;
  cf_math_dtype dtype = CF_MATH_DTYPE_BOOL;
  cf_math_device device = CF_MATH_DEVICE_CPU;

  status = cf_ai_check_loss_views(out, prediction, target);
  if(status != CF_OK) return status;
  if(loss != CF_AI_LOSS_MSE && loss != CF_AI_LOSS_BINARY_CROSS_ENTROPY) return CF_ERR_UNSUPPORTED;

  status = cf_ai_resolve_ptr(out, &out_ptr);
  if(status != CF_OK) return status;
  status = cf_ai_resolve_ptr(prediction, &prediction_ptr);
  if(status != CF_OK) return status;
  status = cf_ai_resolve_ptr(target, &target_ptr);
  if(status != CF_OK) return status;
  if(out_ptr == CF_NULL || prediction_ptr == CF_NULL || target_ptr == CF_NULL) return CF_ERR_STATE;

  dtype = prediction->handler->storage.dtype;
  device = prediction->handler->storage.device;
  if(device == CF_MATH_DEVICE_CPU || (prediction->handler->storage.allocator.mem_flag & CF_MATH_MEM_PINNED) != 0)
    return cf_ai_loss_cpu(loss, dtype, out_ptr, prediction_ptr, target_ptr, prediction->metadata->len);

#if defined(CF_CUDA_AVAILABLE)
  {
    cudaStream_t stream = out->handler->cuda_ctx != CF_NULL ? out->handler->cuda_ctx->stream : CF_NULL;
    if(dtype == CF_MATH_DTYPE_F32)
      cf_ai_loss_kernel_f32<<<1, 1, 0, stream>>>(loss, (float *)out_ptr, (const float *)prediction_ptr, (const float *)target_ptr, prediction->metadata->len);
    else if(dtype == CF_MATH_DTYPE_F64)
      cf_ai_loss_kernel_f64<<<1, 1, 0, stream>>>(loss, (double *)out_ptr, (const double *)prediction_ptr, (const double *)target_ptr, prediction->metadata->len);
    else
      return CF_ERR_UNSUPPORTED;
    if(cudaGetLastError() != cudaSuccess) return CF_ERR_CUDA_LAUNCH;
    return cf_ai_cuda_sync(out->handler);
  }
#else
  return CF_ERR_UNSUPPORTED;
#endif
}
