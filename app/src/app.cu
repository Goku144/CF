#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "MATH/cf_math.h"
#include "RUNTIME/stb_image.h"

#define DIGIT_BATCH_SIZE 64
#define DIGIT_IMAGE_H 28
#define DIGIT_IMAGE_W 28
#define DIGIT_IMAGE_PIXELS (DIGIT_IMAGE_H * DIGIT_IMAGE_W)
#define DIGIT_CONV_CHANNELS 16
#define DIGIT_PADDED_CLASSES 16
#define DIGIT_REAL_CLASSES 10
#define DIGIT_FLATTENED_FEATURES (DIGIT_CONV_CHANNELS * 14 * 14)

typedef struct {
  char path[256];
  uint8_t label;
} sample_t;

typedef struct {
  cf_math input_raw;
  cf_math input;
  cf_math labels;
  cf_math conv_w;
  cf_math conv_out;
  cf_math relu_out;
  cf_math pool_out;
  cf_math flat;
  cf_math dense_w;
  cf_math dense_b;
  cf_math logits;
  cf_math probs;
  cf_math dY;
  cf_math batch_loss;
  cf_math loss;
  cf_math d_flat;
  cf_math d_pool;
  cf_math d_relu;
  cf_math d_conv_out;
  cf_math d_input;
  cf_math d_conv_w;
  cf_math d_dense_w;
  cf_math d_dense_b;
} digit_tensors;

typedef struct {
  cf_math_desc raw_desc;
  cf_math_desc input_desc;
  cf_math_desc label_desc;
  cf_math_desc conv_w_desc;
  cf_math_desc conv_out_desc;
  cf_math_desc pool_desc;
  cf_math_desc flat_desc;
  cf_math_desc dense_w_desc;
  cf_math_desc dense_b_desc;
  cf_math_desc logits_desc;
  cf_math_desc batch_loss_desc;
  cf_math_desc scalar_desc;
} digit_descs;

static const char *g_train_csv = "public/train.csv";
static const char *g_test_csv = "public/test.csv";
static const char *g_dataset_root = "public";
static int g_batch_cursor = 0;

static int app_check_cf_status(const char *step, cf_status state)
{
  if(state == CF_OK) return 0;
  printf("%s failed: %s\n", step, cf_status_as_char(state));
  return 1;
}

static int app_check_cuda_status(const char *step, cudaError_t state)
{
  if(state == cudaSuccess) return 0;
  printf("%s failed: %s\n", step, cudaGetErrorString(state));
  return 1;
}

static void *app_device_ptr(const cf_math_handle *handle, const cf_math *math)
{
  return (void *)((cf_uptr)handle->storage.backend + (cf_uptr)math->byte_offset);
}

__global__ static void app_normalize_u16_to_f16_kernel(uint4 *out, const uint4 *in, int chunks)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= chunks) return;

  uint4 src = in[idx];
  uint4 dst;
  const uint16_t *s = (const uint16_t *)&src;
  __half *d = (__half *)&dst;

  #pragma unroll
  for(int i = 0; i < 8; ++i) {
    d[i] = __float2half_rn((float)s[i] * (1.0f / 65535.0f));
  }

  out[idx] = dst;
}

__global__ static void app_normalize_u16_to_f16_tail_kernel(__half *out, const uint16_t *in, int start, int count)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= count) return;
  out[start + idx] = __float2half_rn((float)in[start + idx] * (1.0f / 65535.0f));
}

static void app_normalize_u16_to_f16(cf_math_handle *handle, cf_math *dst, cf_math *src)
{
  int n = (int)dst->elem_len;
  int chunks = n / 8;
  int threads = 256;
  __half *dst_d = (__half *)app_device_ptr(handle, dst);
  uint16_t *src_d = (uint16_t *)app_device_ptr(handle, src);

  if(chunks > 0) {
    int blocks = (chunks + threads - 1) / threads;
    app_normalize_u16_to_f16_kernel<<<blocks, threads, 0, handle->workspace->stream>>>((uint4 *)dst_d, (const uint4 *)src_d, chunks);
  }

  int tail_start = chunks * 8;
  if(tail_start < n) {
    int tail = n - tail_start;
    int blocks = (tail + threads - 1) / threads;
    app_normalize_u16_to_f16_tail_kernel<<<blocks, threads, 0, handle->workspace->stream>>>(dst_d, src_d, tail_start, tail);
  }
}

static int app_join_path(char *dst, size_t dst_size, const char *root, const char *rel)
{
  int n = snprintf(dst, dst_size, "%s/%s", root, rel);
  return n > 0 && (size_t)n < dst_size;
}

int load_csv(const char *csv_path, sample_t **out_samples, int *out_count)
{
  FILE *file = fopen(csv_path, "rb");
  char line[512];
  sample_t *samples = NULL;
  int count = 0;
  int cap = 0;

  if(file == NULL || out_samples == NULL || out_count == NULL) {
    if(file != NULL) fclose(file);
    return 0;
  }

  if(fgets(line, sizeof(line), file) == NULL) {
    fclose(file);
    return 0;
  }

  while(fgets(line, sizeof(line), file) != NULL) {
    char path[256];
    int label = 0;

    if(sscanf(line, "%255[^,],%d", path, &label) != 2) continue;
    if(label < 0 || label >= DIGIT_REAL_CLASSES) continue;

    if(count == cap) {
      int next_cap = cap == 0 ? 1024 : cap * 2;
      sample_t *next = (sample_t *)realloc(samples, (size_t)next_cap * sizeof(sample_t));
      if(next == NULL) {
        free(samples);
        fclose(file);
        return 0;
      }
      samples = next;
      cap = next_cap;
    }

    snprintf(samples[count].path, sizeof(samples[count].path), "%s", path);
    samples[count].label = (uint8_t)label;
    ++count;
  }

  fclose(file);
  *out_samples = samples;
  *out_count = count;
  return count > 0;
}

static void app_resize_or_copy_u16(uint16_t *dst, const uint16_t *src, int width, int height)
{
  if(width == DIGIT_IMAGE_W && height == DIGIT_IMAGE_H) {
    memcpy(dst, src, DIGIT_IMAGE_PIXELS * sizeof(uint16_t));
    return;
  }

  for(int y = 0; y < DIGIT_IMAGE_H; ++y) {
    int sy = (y * height) / DIGIT_IMAGE_H;
    for(int x = 0; x < DIGIT_IMAGE_W; ++x) {
      int sx = (x * width) / DIGIT_IMAGE_W;
      dst[y * DIGIT_IMAGE_W + x] = src[sy * width + sx];
    }
  }
}

void get_next_batch(sample_t *dataset, int dataset_size, uint16_t *images_cpu, uint8_t *labels_cpu, int batch_size)
{
  for(int i = 0; i < batch_size; ++i) {
    int index = g_batch_cursor++ % dataset_size;
    char full_path[512];
    int width = 0;
    int height = 0;
    int channels = 0;
    uint16_t *pixels = NULL;
    uint16_t *dst = images_cpu + (size_t)i * DIGIT_IMAGE_PIXELS;

    labels_cpu[i] = dataset[index].label;
    memset(dst, 0, DIGIT_IMAGE_PIXELS * sizeof(uint16_t));

    if(!app_join_path(full_path, sizeof(full_path), g_dataset_root, dataset[index].path)) {
      continue;
    }

    pixels = stbi_load_16(full_path, &width, &height, &channels, 1);
    if(pixels == NULL || width <= 0 || height <= 0) {
      if(pixels != NULL) stbi_image_free(pixels);
      continue;
    }

    app_resize_or_copy_u16(dst, pixels, width, height);
    stbi_image_free(pixels);
    (void)channels;
  }
}

static int app_desc_create(digit_descs *d)
{
  int raw_dim[4] = {DIGIT_BATCH_SIZE, 1, DIGIT_IMAGE_H, DIGIT_IMAGE_W};
  int label_dim[1] = {DIGIT_BATCH_SIZE};
  int conv_w_dim[4] = {DIGIT_CONV_CHANNELS, 1, 3, 3};
  int conv_out_dim[4] = {DIGIT_BATCH_SIZE, DIGIT_CONV_CHANNELS, DIGIT_IMAGE_H, DIGIT_IMAGE_W};
  int pool_dim[4] = {DIGIT_BATCH_SIZE, DIGIT_CONV_CHANNELS, 14, 14};
  int flat_dim[2] = {DIGIT_BATCH_SIZE, DIGIT_FLATTENED_FEATURES};
  int dense_w_dim[2] = {DIGIT_FLATTENED_FEATURES, DIGIT_PADDED_CLASSES};
  int dense_b_dim[1] = {DIGIT_PADDED_CLASSES};
  int logits_dim[2] = {DIGIT_BATCH_SIZE, DIGIT_PADDED_CLASSES};
  int batch_loss_dim[1] = {DIGIT_BATCH_SIZE};
  int scalar_dim[1] = {1};

  memset(d, 0, sizeof(*d));
  if(app_check_cf_status("cf_math_desc_create(raw)", cf_math_desc_create(&d->raw_desc, 4, raw_dim, CF_MATH_DTYPE_F16))) return 0;
  if(app_check_cf_status("cf_math_desc_create(input)", cf_math_desc_create(&d->input_desc, 4, raw_dim, CF_MATH_DTYPE_F16))) return 0;
  if(app_check_cf_status("cf_math_desc_create(labels)", cf_math_desc_create(&d->label_desc, 1, label_dim, CF_MATH_DTYPE_U8))) return 0;
  if(app_check_cf_status("cf_math_desc_create(conv_w)", cf_math_desc_create(&d->conv_w_desc, 4, conv_w_dim, CF_MATH_DTYPE_F16))) return 0;
  if(app_check_cf_status("cf_math_desc_create(conv_out)", cf_math_desc_create(&d->conv_out_desc, 4, conv_out_dim, CF_MATH_DTYPE_F16))) return 0;
  if(app_check_cf_status("cf_math_desc_create(pool)", cf_math_desc_create(&d->pool_desc, 4, pool_dim, CF_MATH_DTYPE_F16))) return 0;
  if(app_check_cf_status("cf_math_desc_create(flat)", cf_math_desc_create(&d->flat_desc, 2, flat_dim, CF_MATH_DTYPE_F16))) return 0;
  if(app_check_cf_status("cf_math_desc_create(dense_w)", cf_math_desc_create(&d->dense_w_desc, 2, dense_w_dim, CF_MATH_DTYPE_F16))) return 0;
  if(app_check_cf_status("cf_math_desc_create(dense_b)", cf_math_desc_create(&d->dense_b_desc, 1, dense_b_dim, CF_MATH_DTYPE_F16))) return 0;
  if(app_check_cf_status("cf_math_desc_create(logits)", cf_math_desc_create(&d->logits_desc, 2, logits_dim, CF_MATH_DTYPE_F16))) return 0;
  if(app_check_cf_status("cf_math_desc_create(batch_loss)", cf_math_desc_create(&d->batch_loss_desc, 1, batch_loss_dim, CF_MATH_DTYPE_F32))) return 0;
  if(app_check_cf_status("cf_math_desc_create(loss)", cf_math_desc_create(&d->scalar_desc, 1, scalar_dim, CF_MATH_DTYPE_F32))) return 0;
  return 1;
}

static void app_desc_destroy(digit_descs *d)
{
  cf_math_desc_destroy(&d->raw_desc);
  cf_math_desc_destroy(&d->input_desc);
  cf_math_desc_destroy(&d->label_desc);
  cf_math_desc_destroy(&d->conv_w_desc);
  cf_math_desc_destroy(&d->conv_out_desc);
  cf_math_desc_destroy(&d->pool_desc);
  cf_math_desc_destroy(&d->flat_desc);
  cf_math_desc_destroy(&d->dense_w_desc);
  cf_math_desc_destroy(&d->dense_b_desc);
  cf_math_desc_destroy(&d->logits_desc);
  cf_math_desc_destroy(&d->batch_loss_desc);
  cf_math_desc_destroy(&d->scalar_desc);
}

static int app_bind_tensors(cf_math_handle *handle, digit_tensors *t, digit_descs *d)
{
  memset(t, 0, sizeof(*t));
  if(app_check_cf_status("cf_math_bind(input_raw)", cf_math_bind(handle, &t->input_raw, &d->raw_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(input)", cf_math_bind(handle, &t->input, &d->input_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(labels)", cf_math_bind(handle, &t->labels, &d->label_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(conv_w)", cf_math_bind(handle, &t->conv_w, &d->conv_w_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(conv_out)", cf_math_bind(handle, &t->conv_out, &d->conv_out_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(relu_out)", cf_math_bind(handle, &t->relu_out, &d->conv_out_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(pool_out)", cf_math_bind(handle, &t->pool_out, &d->pool_desc))) return 0;
  t->flat = t->pool_out;
  t->flat.desc = &d->flat_desc;
  if(app_check_cf_status("cf_math_bind(dense_w)", cf_math_bind(handle, &t->dense_w, &d->dense_w_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(dense_b)", cf_math_bind(handle, &t->dense_b, &d->dense_b_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(logits)", cf_math_bind(handle, &t->logits, &d->logits_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(probs)", cf_math_bind(handle, &t->probs, &d->logits_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(dY)", cf_math_bind(handle, &t->dY, &d->logits_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(batch_loss)", cf_math_bind(handle, &t->batch_loss, &d->batch_loss_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(loss)", cf_math_bind(handle, &t->loss, &d->scalar_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(d_flat)", cf_math_bind(handle, &t->d_flat, &d->flat_desc))) return 0;
  t->d_pool = t->d_flat;
  t->d_pool.desc = &d->pool_desc;
  if(app_check_cf_status("cf_math_bind(d_relu)", cf_math_bind(handle, &t->d_relu, &d->conv_out_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(d_conv_out)", cf_math_bind(handle, &t->d_conv_out, &d->conv_out_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(d_input)", cf_math_bind(handle, &t->d_input, &d->input_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(d_conv_w)", cf_math_bind(handle, &t->d_conv_w, &d->conv_w_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(d_dense_w)", cf_math_bind(handle, &t->d_dense_w, &d->dense_w_desc))) return 0;
  if(app_check_cf_status("cf_math_bind(d_dense_b)", cf_math_bind(handle, &t->d_dense_b, &d->dense_b_desc))) return 0;
  return 1;
}

static int app_init_param(cf_math_handle *handle, cf_math *param, float scale)
{
  __half *host = (__half *)malloc(param->elem_len * sizeof(__half));
  if(host == NULL) return 0;

  for(cf_usize i = 0; i < param->elem_len; ++i) {
    int centered = (int)(i % 17) - 8;
    host[i] = __float2half_rn((float)centered * scale);
  }

  cudaError_t state = cudaMemcpyAsync(app_device_ptr(handle, param), host, param->elem_len * sizeof(__half), cudaMemcpyHostToDevice, handle->workspace->stream);
  free(host);
  return app_check_cuda_status("cudaMemcpyAsync(init_param)", state) == 0;
}

static int app_init_zero_param(cf_math_handle *handle, cf_math *param)
{
  return app_check_cuda_status("cudaMemsetAsync(init_zero)", cudaMemsetAsync(app_device_ptr(handle, param), 0, param->elem_len * sizeof(__half), handle->workspace->stream)) == 0;
}

static void app_attach_grads(digit_tensors *t, cf_math_grad_node *conv_node, cf_math_grad_node *dense_w_node, cf_math_grad_node *dense_b_node)
{
  *conv_node = (cf_math_grad_node){.grad = &t->d_conv_w, .grad_state = CF_MATH_GRAD_LEAF};
  *dense_w_node = (cf_math_grad_node){.grad = &t->d_dense_w, .grad_state = CF_MATH_GRAD_LEAF};
  *dense_b_node = (cf_math_grad_node){.grad = &t->d_dense_b, .grad_state = CF_MATH_GRAD_LEAF};
  t->conv_w.grad_fn = conv_node;
  t->dense_w.grad_fn = dense_w_node;
  t->dense_b.grad_fn = dense_b_node;
}

static void digit_forward(cf_math_handle *handle, digit_tensors *t)
{
  cf_math_conv2d_f16(handle, &t->conv_out, &t->input, &t->conv_w, 1, 1, 1, 1, 1, 1);
  cf_math_relu_f16(handle, &t->relu_out, &t->conv_out);
  cf_math_pooling_f16(handle, &t->pool_out, &t->relu_out, CF_MATH_POOLING_MAX, 2, 2, 0, 0, 2, 2);
  cf_math_linear_bias_f16(handle, &t->logits, &t->flat, &t->dense_w, &t->dense_b);
  cf_math_softmax_f16(handle, &t->probs, &t->logits, 1);
}

static int cf_math_save_weights(cf_math_handle *handle, const cf_math *weights, const char *path)
{
  FILE *file = NULL;
  __half *host = NULL;
  size_t bytes = weights->elem_len * sizeof(__half);

  if(app_check_cuda_status("cudaStreamSynchronize(save)", cudaStreamSynchronize(handle->workspace->stream))) return 0;

  host = (__half *)malloc(bytes);
  if(host == NULL) return 0;

  if(app_check_cf_status("cf_math_copy_to_host(save)", cf_math_copy_to_host(handle, weights, host, bytes))) {
    free(host);
    return 0;
  }

  file = fopen(path, "wb");
  if(file == NULL) {
    free(host);
    return 0;
  }

  if(fwrite(host, 1, bytes, file) != bytes) {
    fclose(file);
    free(host);
    return 0;
  }

  fclose(file);
  free(host);
  return 1;
}

static void app_save_checkpoints(cf_math_handle *handle, digit_tensors *t, const char *save_dir, int step)
{
  char path[512];

  snprintf(path, sizeof(path), "%s/layer1_weights_step%d.bin", save_dir, step);
  if(!cf_math_save_weights(handle, &t->conv_w, path)) printf("checkpoint failed: %s\n", path);

  snprintf(path, sizeof(path), "%s/dense_weights_step%d.bin", save_dir, step);
  if(!cf_math_save_weights(handle, &t->dense_w, path)) printf("checkpoint failed: %s\n", path);

  snprintf(path, sizeof(path), "%s/dense_bias_step%d.bin", save_dir, step);
  if(!cf_math_save_weights(handle, &t->dense_b, path)) printf("checkpoint failed: %s\n", path);
}

int predict_digit(cf_math_handle *handle, digit_tensors *t, float *confidence)
{
  __half probs[DIGIT_PADDED_CLASSES];
  int best = 0;
  float best_p = -1.0f;

  digit_forward(handle, t);

  cf_math first = t->probs;
  first.elem_len = DIGIT_PADDED_CLASSES;
  if(app_check_cf_status("cf_math_copy_to_host(predict)", cf_math_copy_to_host(handle, &first, probs, sizeof(probs)))) {
    if(confidence != NULL) *confidence = 0.0f;
    return -1;
  }

  for(int i = 0; i < DIGIT_REAL_CLASSES; ++i) {
    float p = __half2float(probs[i]);
    if(p > best_p) {
      best_p = p;
      best = i;
    }
  }

  if(confidence != NULL) *confidence = best_p;
  return best;
}

void train_digit_recognizer(cf_math_handle *handle, int total_steps, const char *save_dir)
{
  sample_t *train = NULL;
  sample_t *test = NULL;
  int train_count = 0;
  int test_count = 0;
  digit_descs descs;
  digit_tensors tensors;
  cf_math_grad_node conv_node;
  cf_math_grad_node dense_w_node;
  cf_math_grad_node dense_b_node;
  uint16_t *images_cpu = NULL;
  uint8_t *labels_cpu = NULL;
  float loss_host = 0.0f;
  const float lr = 0.01f;

  if(total_steps <= 0) total_steps = 1;
  mkdir(save_dir, 0755);

  if(!load_csv(g_train_csv, &train, &train_count)) {
    printf("Failed to load training CSV: %s\n", g_train_csv);
    return;
  }
  if(!load_csv(g_test_csv, &test, &test_count)) {
    printf("Warning: failed to load test CSV: %s\n", g_test_csv);
  }

  if(!app_desc_create(&descs)) goto done;
  if(!app_bind_tensors(handle, &tensors, &descs)) goto done;

  app_attach_grads(&tensors, &conv_node, &dense_w_node, &dense_b_node);

  if(!app_init_param(handle, &tensors.conv_w, 0.01f)) goto done;
  if(!app_init_param(handle, &tensors.dense_w, 0.001f)) goto done;
  if(!app_init_zero_param(handle, &tensors.dense_b)) goto done;

  if(app_check_cuda_status("cudaMallocHost(images)", cudaMallocHost((void **)&images_cpu, DIGIT_BATCH_SIZE * DIGIT_IMAGE_PIXELS * sizeof(uint16_t)))) goto done;
  if(app_check_cuda_status("cudaMallocHost(labels)", cudaMallocHost((void **)&labels_cpu, DIGIT_BATCH_SIZE * sizeof(uint8_t)))) goto done;

  for(int step = 1; step <= total_steps; ++step) {
    get_next_batch(train, train_count, images_cpu, labels_cpu, DIGIT_BATCH_SIZE);

    cf_math_zero_grad_f16(handle, &tensors.conv_w);
    cf_math_zero_grad_f16(handle, &tensors.dense_w);
    cf_math_zero_grad_f16(handle, &tensors.dense_b);

    if(app_check_cuda_status("cudaMemcpyAsync(images)", cudaMemcpyAsync(app_device_ptr(handle, &tensors.input_raw), images_cpu, DIGIT_BATCH_SIZE * DIGIT_IMAGE_PIXELS * sizeof(uint16_t), cudaMemcpyHostToDevice, handle->workspace->stream))) goto done;
    if(app_check_cuda_status("cudaMemcpyAsync(labels)", cudaMemcpyAsync(app_device_ptr(handle, &tensors.labels), labels_cpu, DIGIT_BATCH_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice, handle->workspace->stream))) goto done;

    app_normalize_u16_to_f16(handle, &tensors.input, &tensors.input_raw);
    digit_forward(handle, &tensors);
    cf_math_fused_cross_entropy(handle, &tensors.dY, &tensors.batch_loss, &tensors.loss, &tensors.probs, &tensors.labels);

    cf_math_matmul_trans_b_f16(handle, &tensors.d_flat, &tensors.dY, &tensors.dense_w);
    cf_math_matmul_trans_a_f16(handle, &tensors.d_dense_w, &tensors.flat, &tensors.dY);
    cf_math_reduce_sum_rows_f16(handle, &tensors.d_dense_b, &tensors.dY);
    cf_math_pooling_backward_f16(handle, &tensors.d_relu, &tensors.d_pool, &tensors.pool_out, &tensors.relu_out, CF_MATH_POOLING_MAX, 2, 2, 0, 0, 2, 2);
    cf_math_relu_backward_f16(handle, &tensors.d_conv_out, &tensors.d_relu, &tensors.conv_out);
    cf_math_conv2d_backward_data_f16(handle, &tensors.d_input, &tensors.d_conv_out, &tensors.conv_w, 1, 1, 1, 1, 1, 1);
    cf_math_conv2d_backward_filter_f16(handle, &tensors.d_conv_w, &tensors.d_conv_out, &tensors.input, 1, 1, 1, 1, 1, 1);

    cf_math_sgd_update_f16(handle, &tensors.conv_w, &tensors.d_conv_w, lr);
    cf_math_sgd_update_f16(handle, &tensors.dense_w, &tensors.d_dense_w, lr);
    cf_math_sgd_update_f16(handle, &tensors.dense_b, &tensors.d_dense_b, lr);

    if(app_check_cf_status("cf_math_copy_to_host(loss)", cf_math_copy_to_host(handle, &tensors.loss, &loss_host, sizeof(loss_host)))) goto done;
    printf("step %d/%d loss %.6f\n", step, total_steps, loss_host);

    if((step % 512) == 0 || step == total_steps) {
      app_save_checkpoints(handle, &tensors, save_dir, step);
    }
  }

  if(test_count > 0) {
    float confidence = 0.0f;
    get_next_batch(test, test_count, images_cpu, labels_cpu, DIGIT_BATCH_SIZE);
    if(app_check_cuda_status("cudaMemcpyAsync(test_images)", cudaMemcpyAsync(app_device_ptr(handle, &tensors.input_raw), images_cpu, DIGIT_BATCH_SIZE * DIGIT_IMAGE_PIXELS * sizeof(uint16_t), cudaMemcpyHostToDevice, handle->workspace->stream))) goto done;
    app_normalize_u16_to_f16(handle, &tensors.input, &tensors.input_raw);
    int digit = predict_digit(handle, &tensors, &confidence);
    printf("test prediction sample: predicted=%d label=%u confidence=%.6f\n", digit, (unsigned)labels_cpu[0], confidence);
  }

done:
  if(images_cpu != NULL) cudaFreeHost(images_cpu);
  if(labels_cpu != NULL) cudaFreeHost(labels_cpu);
  app_desc_destroy(&descs);
  free(test);
  free(train);
}

int main(int argc, char **argv)
{
  int total_steps = argc > 1 ? atoi(argv[1]) : 2;
  const char *save_dir = argc > 2 ? argv[2] : "checkpoints";
  const cf_usize workspace_capacity = 128 * 1024 * 1024;
  const cf_usize storage_capacity = 256 * 1024 * 1024;
  cf_math_context ctx = {0};
  cf_math_workspace workspace = {0};
  cf_math_handle handle = {0};
  int exit_code = 1;

  if(argc > 3) g_train_csv = argv[3];
  if(argc > 4) g_test_csv = argv[4];
  if(argc > 5) g_dataset_root = argv[5];

  printf("Digit trainer: steps=%d save_dir=%s train_csv=%s test_csv=%s dataset_root=%s\n", total_steps, save_dir, g_train_csv, g_test_csv, g_dataset_root);
  if(app_check_cf_status("cf_math_context_create", cf_math_context_create(&ctx, 0, CF_MATH_DEVICE_CUDA))) goto done;
  if(app_check_cf_status("cf_math_workspace_create", cf_math_workspace_create(&workspace, workspace_capacity, CF_MATH_DEVICE_CUDA))) goto done;
  if(app_check_cf_status("cf_math_handle_create", cf_math_handle_create(&handle, &ctx, &workspace, storage_capacity, CF_MATH_DEVICE_CUDA))) goto done;

  train_digit_recognizer(&handle, total_steps, save_dir);

  if(app_check_cuda_status("cudaStreamSynchronize(final)", cudaStreamSynchronize(handle.workspace->stream))) goto done;
  exit_code = 0;

done:
  cf_math_handle_destroy(&handle);
  cf_math_workspace_destroy(&workspace);
  cf_math_context_destroy(&ctx);
  return exit_code;
}
