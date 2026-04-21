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

#include "ALLOCATOR/cf_alloc_debug.h"
#include "ALLOCATOR/cf_alloc.h"

#include <stdlib.h>
#include <stdio.h>

static void *cf_alloc_debug_alloc(void *ctx, cf_usize size)
{
  if(ctx == CF_NULL) return CF_NULL;
  cf_alloc_debug *alloc_debug = (cf_alloc_debug *) ctx;
  CF_ASSERT_TYPE_SIZE(*alloc_debug, cf_alloc_debug);

  void *ptr = alloc_debug->internal_allocator.alloc(alloc_debug->internal_allocator.ctx, size);
  if(ptr == CF_NULL) {alloc_debug->ptr_internal_invalid_alloc++; return CF_NULL;}

  if(alloc_debug->ptr_free > 0) 
    alloc_debug->ptr_free--;
  else 
    alloc_debug->ptr_live++;
  alloc_debug->ptr_all_live++;
  alloc_debug->latest_valid_ptr = ptr;
  if(alloc_debug->ptr_max_live < alloc_debug->ptr_live)
    alloc_debug->ptr_max_live = alloc_debug->ptr_live;

  cf_alloc_debug_node *new_node = malloc(sizeof (cf_alloc_debug_node));
  if(new_node == CF_NULL) {alloc_debug->ptr_invalid_alloc++; return ptr;}

  new_node->ptr = ptr;
  new_node->next = alloc_debug->head;
  alloc_debug->head = new_node;
  return ptr;
}

static void *cf_alloc_debug_realloc(void *ctx, void *ptr, cf_usize size)
{
  if(ctx == CF_NULL) return CF_NULL;
  cf_alloc_debug *alloc_debug = (cf_alloc_debug *) ctx;
  cf_alloc_debug_node *node = alloc_debug->head;

  while (node != CF_NULL)
  {
    if(node->ptr == ptr)
    {
      void *ptr_realloc = alloc_debug->internal_allocator.realloc(alloc_debug->internal_allocator.ctx, ptr, size);
      if(ptr_realloc == CF_NULL) {alloc_debug->ptr_internal_invalid_realloc++; return CF_NULL;}
      node->ptr = ptr_realloc;
      alloc_debug->latest_valid_ptr = ptr_realloc;
      return ptr_realloc;
    }
    node = node->next;
  }

  return CF_NULL;
}

static void cf_alloc_debug_free(void *ctx, void *ptr)
{
  if(ctx == CF_NULL) return;
  cf_alloc_debug *alloc_debug = (cf_alloc_debug *) ctx;
  cf_alloc_debug_node *node = alloc_debug->head;
  cf_alloc_debug_node *old_node = CF_NULL;
  while (node != CF_NULL)
  {
    if(node->ptr == ptr)
    {
      if(old_node == CF_NULL) 
        alloc_debug->head = node->next;
      else 
        old_node->next = node->next;
      
      if(alloc_debug->ptr_live > 0) 
        alloc_debug->ptr_live--;
      else
        alloc_debug->ptr_free++;
      alloc_debug->ptr_all_free++;
      alloc_debug->latest_valid_ptr = alloc_debug->head != CF_NULL ? alloc_debug->head->ptr : CF_NULL;

      if(alloc_debug->ptr_max_free < alloc_debug->ptr_free)
        alloc_debug->ptr_max_free = alloc_debug->ptr_free;

      alloc_debug->internal_allocator.free(alloc_debug->internal_allocator.ctx, ptr);
      free(node);
      return;
    }
    old_node = node;
    node = node->next;
  }
  alloc_debug->ptr_internal_invalid_free++;
}


static cf_alloc_debug cf_alloc_debug_create(void)
{
  cf_alloc_debug alloc_debug = {0};
  alloc_debug.allocator = (cf_alloc) 
  {
    .ctx = CF_NULL,
    .alloc = cf_alloc_debug_alloc,
    .realloc = cf_alloc_debug_realloc,
    .free = cf_alloc_debug_free,
  };
  return alloc_debug;
}

void cf_alloc_debug_new(cf_alloc_debug *alloc_debug, cf_alloc *alloc, char* statement)
{
  if (alloc_debug == CF_NULL || alloc == CF_NULL) return;
  CF_ASSERT_TYPE_SIZE(*alloc_debug, cf_alloc_debug);
  CF_ASSERT_TYPE_SIZE(*alloc, cf_alloc);
  *alloc_debug = cf_alloc_debug_create();
  alloc_debug->allocator.ctx = alloc_debug;
  alloc_debug->internal_allocator = *alloc;
  alloc_debug->statement = statement == CF_NULL ? "NO STATEMENT DECLARED" : statement;
}

void cf_alloc_debug_log(cf_alloc_debug *debug, int line)
{
  if(debug == CF_NULL) return;
  CF_ASSERT_TYPE_SIZE(*debug, cf_alloc_debug);
  printf
    (
"================== Debug log Line(%d) ==================\n\
      \n\
      statement (string): %s\n\
      \n\
        Tested allocator (pointer): %p\n\
        Debug allocator (pointer): %p\n\
      \n\
      live/free (diffr): (%zu/%zu)\n\
      Max lived (count): %zu\n\
      Max freed (count): %zu\n\
      All lived (count): %zu\n\
      All freed (count): %zu\n\
      \n\
      Invalid allocation by debuger   (count): %zu\n\
      Invalid allocation by Tested    (count): %zu\n\
      Invalid reallocation by Tested  (count): %zu\n\
      Invalid free by Tested          (count): %zu\n\
      \n\
      latest valid by Tested (pointer): %p\n\
========================================================\n"
    ,
    line,
    debug->statement,
    (void *)&debug->internal_allocator,
    (void *)&debug->allocator,
    debug->ptr_live,
    debug->ptr_free,
    debug->ptr_max_live,
    debug->ptr_max_free,
    debug->ptr_all_live,
    debug->ptr_all_free,
    debug->ptr_invalid_alloc,
    debug->ptr_internal_invalid_alloc,
    debug->ptr_internal_invalid_realloc,
    debug->ptr_internal_invalid_free,
    debug->latest_valid_ptr
  );
  
}