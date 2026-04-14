#include "ALLOCATOR/cf_alloc_debug.h"
#include "ALLOCATOR/cf_alloc.h"
#include "ALLOCATOR/cf_arena.h"
#include "ALLOCATOR/cf_pool.h"
#include "ALLOCATOR/cf_slab.h"
#include "RUNTIME/cf_status.h"

int main(void)
{
    cf_alloc_debug debug;
    cf_pool pool;
    cf_pool_new(&pool, 128, 128);
    cf_status_print(cf_alloc_debug_new(&debug, &pool.allocator));
    cf_u8 *data[7];
    for (cf_usize i = 0; i < 7; i++)
    {
        data[i] = debug.allocator.alloc(debug.allocator.ctx, pool.slot_size);
    }
    debug.allocator.free(debug.allocator.ctx, (void *)(0x4265f));
    debug.allocator.free(debug.allocator.ctx, data[2]);
    data[2] = debug.allocator.alloc(debug.allocator.ctx, pool.slot_size);
    CF_UNUSED(data);
    cf_alloc_debug_report(&debug);
    return 0;
}

