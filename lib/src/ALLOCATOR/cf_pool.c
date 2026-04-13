#include "ALLOCATOR/cf_pool.h"
#include <stdlib.h>

static void *cf_pool_alloc(void *ctx, cf_usize size)
{
  CF_UNUSED(size);
  if(ctx == CF_NULL) return CF_NULL;
  cf_pool *pool = (cf_pool *) ctx;
  if(!cf_pool_is_valid(pool)) return CF_NULL;
  if(pool->data == CF_NULL) return CF_NULL;
  if(pool->slot_used == pool->slot_total) return CF_NULL;
  void *ptr = pool->list;
  pool->list = *(void **) pool->list;
  ++pool->slot_used;
  return ptr;
}

static void *cf_pool_realloc(void *ctx, void *ptr, cf_usize size)
{
  CF_UNUSED(ctx);
  CF_UNUSED(ptr);
  CF_UNUSED(size);
  return CF_NULL;
}

static void cf_pool_free(void *ctx, void *ptr)
{
  if(ctx == CF_NULL) return;
  cf_pool *pool = (cf_pool *) ctx;
  if(!cf_pool_is_valid(pool)) return;
  if(pool->data == CF_NULL) return;
  if(pool->slot_used == 0) return;
  cf_u8 *base = pool->data;
  cf_u8 *end  = pool->data + pool->slot_total * pool->slot_size;
  if((cf_u8 *)ptr < base || (cf_u8 *)ptr >= end) return;
  if(((cf_u8 *)ptr - base) % pool->slot_size != 0) return;
  *(void **) ptr = pool->list;
  pool->list = ptr;
  --pool->slot_used;
}

cf_pool cf_pool_create_empty()
{
  return (cf_pool) {CF_NULL, CF_NULL, 0, 0, 0, (cf_alloc) {CF_NULL, cf_pool_alloc, cf_pool_realloc, cf_pool_free}};
}

cf_bool cf_pool_is_valid(cf_pool *pool)
{
  if(pool == CF_NULL) return CF_FALSE;
  if(pool->data == CF_NULL)
  {
    if(pool->list != CF_NULL) return CF_FALSE;
    if(pool->slot_total != 0) return CF_FALSE;
    if(pool->slot_size != 0) return CF_FALSE;
    if(pool->slot_used != 0) return CF_FALSE;
  }
  else
  {
    if(pool->slot_size < sizeof(void *)) return CF_FALSE;
    if(pool->slot_total == 0 || pool->slot_size == 0) return CF_FALSE;
    if(pool->slot_total < pool->slot_used) return CF_FALSE;
  }
  if(!cf_alloc_is_valid(&pool->allocator)) return CF_FALSE;
  return CF_TRUE;
}

cf_status cf_pool_new(cf_pool *pool, cf_usize slot_total, cf_usize slot_size)
{
  if(pool == CF_NULL) return CF_ERR_NULL;
  *pool = cf_pool_create_empty();
  if(slot_total == 0 && slot_size == 0) return CF_OK;
  if(slot_total == 0 || slot_size < sizeof(void *)) return CF_ERR_INVALID;
  if(slot_size < sizeof (void *)) return CF_ERR_INVALID;
  pool->data = malloc(slot_total * slot_size);
  if(pool->data == CF_NULL) return CF_ERR_OOM;
  pool->allocator.ctx = pool;
  pool->slot_total = slot_total;
  pool->slot_size = slot_size;
  return cf_pool_reset(pool);
}

cf_status cf_pool_reset(cf_pool *pool)
{
  if(pool == CF_NULL) return CF_ERR_NULL;
  if(!cf_pool_is_valid(pool)) return CF_ERR_STATE;
  if(pool->data == CF_NULL) return CF_OK;
  pool->slot_used = 0;
  pool->list = pool->data;
  for (cf_usize i = 1; i < pool->slot_total; i++)
  {
    *(void **) pool->list = (void *) ((cf_u8 *) pool->list + pool->slot_size);
    pool->list = *(void **) pool->list;
  }
  *(void **) pool->list = CF_NULL;
  pool->list = pool->data;
  return CF_OK;
}

void cf_pool_destroy(cf_pool *pool)
{
  if(pool == CF_NULL) return;
  if(!cf_pool_is_valid(pool)) return;
  if(pool->data == CF_NULL) return;
  free(pool->data);
  *pool = cf_pool_create_empty();
}