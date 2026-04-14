#include "ALLOCATOR/cf_alloc_debug.h"

#include <stdio.h>
#include <stdlib.h>

static void *cf_alloc_debug_alloc(void *ctx, cf_usize size)
{
  if(ctx == CF_NULL) return CF_NULL;
  cf_alloc_debug *debug = (cf_alloc_debug *) ctx;
  cf_alloc_debug_node *new_node = malloc(sizeof(cf_alloc_debug_node));
  if(new_node == CF_NULL) return CF_NULL;
  void *ptr = debug->backing.alloc(debug->backing.ctx, size);
  if(ptr == CF_NULL)
  {
    free(new_node);
    debug->failed_alloc_count++;
    return CF_NULL;
  }
  new_node->ptr  = ptr;
  new_node->size = size;
  new_node->next = debug->head;
  debug->head = new_node;
  debug->alloc_count++;
  debug->live_count++;
  debug->bytes_live  += size;
  debug->bytes_total += size;
  if(debug->bytes_live > debug->bytes_peak)
    debug->bytes_peak = debug->bytes_live;
  return ptr;
}

static void *cf_alloc_debug_realloc(void *ctx, void *ptr, cf_usize size)
{
  if(ctx == CF_NULL || size == 0) return CF_NULL;
  cf_alloc_debug *debug = (cf_alloc_debug *) ctx;
  cf_alloc_debug_node *curr = debug->head;
  while(curr != CF_NULL)
  {
    if(curr->ptr == ptr)
    {
      void *new_ptr = debug->backing.realloc(debug->backing.ctx, ptr, size);
      if(new_ptr == CF_NULL)
      {
        debug->failed_realloc_count++;
        return CF_NULL;
      }
      debug->bytes_live  -= curr->size;
      debug->bytes_live  += size;
      debug->bytes_total += size;
      debug->realloc_count++;

      if(debug->bytes_live > debug->bytes_peak)
        debug->bytes_peak = debug->bytes_live;

      curr->ptr  = new_ptr;
      curr->size = size;
      return new_ptr;
    }
    curr = curr->next;
  }
  debug->invalid_free_count++;
  return CF_NULL;
}

static void cf_alloc_debug_free(void *ctx, void *ptr)
{
  if(ctx == CF_NULL || ptr == CF_NULL) return;
  cf_alloc_debug *debug = (cf_alloc_debug *) ctx;
  cf_alloc_debug_node *prev = CF_NULL;
  cf_alloc_debug_node *curr = debug->head;
  while(curr != CF_NULL)
  {
    if(curr->ptr == ptr)
    {
      if(prev == CF_NULL)
        debug->head = curr->next;
      else
        prev->next = curr->next;
      debug->bytes_live -= curr->size;
      debug->live_count--;
      debug->free_count++;
      free(curr);
      debug->backing.free(debug->backing.ctx, ptr);
      return;
    }
    prev = curr;
    curr = curr->next;
  }
  debug->invalid_free_count++;
}

/********************************************************************/
/* construction                                                     */
/********************************************************************/

cf_alloc_debug cf_alloc_debug_create_empty(void)
{
  cf_alloc_debug debug;
  debug.backing = cf_alloc_create_empty();
  debug.allocator = (cf_alloc) {
    CF_NULL,
    cf_alloc_debug_alloc,
    cf_alloc_debug_realloc,
    cf_alloc_debug_free
  };
  debug.head = CF_NULL;
  debug.alloc_count = 0;
  debug.free_count = 0;
  debug.realloc_count = 0;
  debug.live_count = 0;
  debug.bytes_live = 0;
  debug.bytes_peak = 0;
  debug.bytes_total = 0;
  debug.invalid_free_count = 0;
  debug.failed_alloc_count = 0;
  debug.failed_realloc_count = 0;
  return debug;
}

/********************************************************************/
/* validation                                                       */
/********************************************************************/

cf_bool cf_alloc_debug_is_valid(cf_alloc_debug *debug)
{
  if(debug == CF_NULL) return CF_FALSE;
  if(!cf_alloc_is_valid(&debug->backing)) return CF_FALSE;
  if(!cf_alloc_is_valid(&debug->allocator)) return CF_FALSE;
  if(debug->live_count > debug->alloc_count) return CF_FALSE;
  if(debug->free_count > debug->alloc_count + debug->realloc_count) return CF_FALSE;
  if(debug->bytes_live > debug->bytes_total) return CF_FALSE;
  if(debug->bytes_peak < debug->bytes_live) return CF_FALSE;

  return CF_TRUE;
}

/********************************************************************/
/* lifecycle                                                        */
/********************************************************************/

cf_status cf_alloc_debug_new(cf_alloc_debug *debug, cf_alloc *backing)
{
  if(debug == CF_NULL || backing == CF_NULL) return CF_ERR_NULL;
  if(!cf_alloc_is_valid(backing)) return CF_ERR_INVALID;
  *debug = cf_alloc_debug_create_empty();
  debug->backing = *backing;
  debug->allocator.ctx = debug;
  return CF_OK;
}

void cf_alloc_debug_destroy(cf_alloc_debug *debug)
{
  if(debug == CF_NULL) return;
  cf_alloc_debug_node *curr = debug->head;
  while(curr != CF_NULL)
  {
    cf_alloc_debug_node *next = curr->next;
    debug->backing.free(debug->backing.ctx, curr->ptr);
    free(curr);
    curr = next;
  }
  *debug = cf_alloc_debug_create_empty();
}

/********************************************************************/
/* operations                                                       */
/********************************************************************/

void cf_alloc_debug_report(cf_alloc_debug *debug)
{
  if(debug == CF_NULL) return;
  printf("=== cf_alloc_debug report ===\n");
  printf("  alloc_count          : %zu\n", debug->alloc_count);
  printf("  realloc_count        : %zu\n", debug->realloc_count);
  printf("  free_count           : %zu\n", debug->free_count);
  printf("  live_count           : %zu\n", debug->live_count);
  printf("  bytes_live           : %zu\n", debug->bytes_live);
  printf("  bytes_peak           : %zu\n", debug->bytes_peak);
  printf("  bytes_total          : %zu\n", debug->bytes_total);
  printf("  invalid_free_count   : %zu\n", debug->invalid_free_count);
  printf("  failed_alloc_count   : %zu\n", debug->failed_alloc_count);
  printf("  failed_realloc_count : %zu\n", debug->failed_realloc_count);
  if(debug->live_count == 0)
  {
    printf("  [OK] no leaks detected\n");
  }
  else
  {
    printf("  [LEAK] %zu allocation(s) still live:\n", debug->live_count);
    cf_alloc_debug_node *curr = debug->head;
    while(curr != CF_NULL)
    {
      printf("    ptr=%p  size=%zu\n", curr->ptr, curr->size);
      curr = curr->next;
    }
  }
  printf("=============================\n");
}