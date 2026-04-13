#include "ALLOCATOR/cf_slab.h"
#include "ALLOCATOR/cf_alloc.h"
#include <stdlib.h>

static void *cf_slab_alloc(void *ctx, cf_usize size)
{
  if(ctx == CF_NULL) return CF_NULL;
  cf_slab *slab = (cf_slab *) ctx;
  if(!cf_slab_is_valid(slab)) return CF_NULL;
  if(slab->class_count == 0) return CF_NULL;
  cf_usize index;
  for (cf_usize i = 0; i < slab->class_count; i++)
  {
    if(size <= slab->class[i].class_slot_size)
    {
      index = i;
      for (cf_usize j = i + 1; j < slab->class_count; j++)
      {
        if(slab->class[j].class_slot_size >= size)
          if(
              slab->class[j].class_slot_size < slab->class[index].class_slot_size 
              || 
              slab->class[index].pool.slot_total == slab->class[index].pool.slot_used
            ) index = j;
      }
      return slab->class[index].pool.allocator.alloc(slab->class[index].pool.allocator.ctx, size);
    }
  }
  return CF_NULL;
}

static void *cf_slab_realloc(void *ctx, void *ptr, cf_usize size)
{
  CF_UNUSED(ctx);
  CF_UNUSED(ptr);
  CF_UNUSED(size);
  return CF_NULL;
}

static void cf_slab_free(void *ctx, void *ptr)
{
  if(ctx == CF_NULL || ptr == CF_NULL) return;
  cf_slab *slab = (cf_slab *) ctx;
  if(!cf_slab_is_valid(slab)) return;
  if(slab->class_count == 0) return;
  for (cf_usize i = 0; i < slab->class_count; i++)
  {
    cf_pool *pool = &slab->class[i].pool;
    cf_u8 *base = pool->data;
    cf_u8 *end  = pool->data + pool->slot_total * pool->slot_size;
    if((cf_u8 *)ptr >= base && (cf_u8 *)ptr < end)
    {
      pool->allocator.free(pool->allocator.ctx, ptr);
      return;
    }
  }
}

static cf_slab cf_slab_create_empty()
{
  cf_slab slab;
  slab.class_count = 0;
  for (cf_usize i = 0; i < CF_SLAB_MAX_CLASS; i++)
  {
    slab.class[i].pool = cf_pool_create_empty();
    slab.class[i].class_slot_size = 0;
  }
  slab.allocator = (cf_alloc) {CF_NULL, cf_slab_alloc, cf_slab_realloc, cf_slab_free};
  return slab;
}

cf_bool cf_slab_is_valid(cf_slab *slab)
{
  if(slab == CF_NULL) return CF_FALSE;
  if(slab->class_count > CF_SLAB_MAX_CLASS) return CF_FALSE;
  if(!cf_alloc_is_valid(&slab->allocator)) return CF_FALSE;
  for (cf_usize i = 0; i < slab->class_count; i++)
  {
    if(!cf_pool_is_valid(&slab->class[i].pool)) return CF_FALSE;
    if(slab->class[i].class_slot_size != slab->class[i].pool.slot_size) return CF_FALSE;
  }
  return CF_TRUE;
}

cf_status cf_slab_new(cf_slab *slab, cf_usize slots_total[], cf_usize slots_size[], cf_usize n)
{
  if(slab == CF_NULL) return CF_ERR_NULL;
  *slab = cf_slab_create_empty();
  if(n > CF_SLAB_MAX_CLASS) return CF_ERR_INVALID;
  for (cf_usize i = 0; i < n; i++)
  {
    cf_status state = CF_OK;
    cf_bool need_create = CF_TRUE;
    for (cf_usize j = 0; j < slab->class_count; j++)
    {
      if(slab->class[j].class_slot_size == slots_size[i])
      {
        need_create = CF_FALSE;
        if(slab->class[j].pool.slot_total < slots_total[i])
        {
          cf_pool_destroy(&slab->class[j].pool);
          state = cf_pool_new(&slab->class[j].pool, slots_total[i], slots_size[i]);
          if(state != CF_OK) {cf_slab_destroy(slab); return state;}
        }
        break;
      }
    }
    if(need_create)
    {
      state = cf_pool_new(&slab->class[slab->class_count].pool, slots_total[i], slots_size[i]);
      if(state != CF_OK) {cf_slab_destroy(slab); return state;}
      slab->class[slab->class_count].class_slot_size = slots_size[i];
      slab->class_count++;
    }   
  }
  slab->allocator.ctx = slab;
  return CF_OK;
}

cf_status cf_slab_reset(cf_slab *slab)
{
  if(slab == CF_NULL) return CF_ERR_NULL;
  if(!cf_slab_is_valid(slab)) return CF_ERR_STATE;
  for (cf_usize i = 0; i < slab->class_count; i++)
  {
    cf_status state;
    if((state = cf_pool_reset(&slab->class[i].pool)) != CF_OK)
      return state;
  }
  return CF_OK;
}

void cf_slab_destroy(cf_slab *slab)
{
  if(slab == CF_NULL) return;
  if(!cf_slab_is_valid(slab)) return;
  for (cf_usize i = 0; i < slab->class_count; i++)
    cf_pool_destroy(&slab->class[i].pool);
  *slab = cf_slab_create_empty();
}