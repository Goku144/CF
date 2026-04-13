#if !defined(CF_SLAB_L)
#define CF_SLAB_L

#include "RUNTIME/cf_types.h"
#include "RUNTIME/cf_status.h"
#include "ALLOCATOR/cf_pool.h"

/** Maximum number of size classes a cf_slab may hold. */
#define CF_SLAB_MAX_CLASS 8

/**
 * cf_slab_class
 *
 * One size class inside a cf_slab.
 * Pairs a cf_pool (the actual free list) with the slot size it serves
 * so the slab can dispatch allocation requests to the best-fit class.
 *
 * @field pool            Pool that manages allocations for this class.
 * @field class_slot_size Slot size this class serves, in bytes.
 *                        Must equal pool.slot_size for the slab to be
 *                        considered valid.
 */
typedef struct cf_slab_class
{
  cf_pool  pool;
  cf_usize class_slot_size;
} cf_slab_class;

/**
 * cf_slab
 *
 * Multi-class slab allocator composed of up to CF_SLAB_MAX_CLASS
 * independent cf_pool instances, each handling a distinct object size.
 *
 * Allocation policy (best-fit with fallback):
 *   1. Find all classes whose slot size >= the requested size.
 *   2. Among those, prefer the class with the smallest slot size to
 *      minimise wasted space.
 *   3. If that class is full, fall back to the next-smallest class
 *      that still has free slots.
 *   4. Return CF_NULL if no suitable class has a free slot.
 *
 * Realloc is not supported and always returns CF_NULL.
 * Free scans classes to find the owning pool and delegates to it.
 *
 * @field class        Array of size-class descriptors.
 * @field class_count  Number of active size classes (<= CF_SLAB_MAX_CLASS).
 * @field allocator    cf_alloc interface for this slab; ctx points back
 *                     to the owning cf_slab instance.
 */
typedef struct cf_slab
{
  cf_slab_class class[CF_SLAB_MAX_CLASS];
  cf_usize      class_count;
  cf_alloc      allocator;
} cf_slab;


/********************************************************************/
/* validation                                                       */
/********************************************************************/

/**
 * cf_slab_is_valid
 *
 * Perform a structural sanity check on @p slab:
 *   - pointer must be non-null.
 *   - class_count must be <= CF_SLAB_MAX_CLASS.
 *   - the embedded allocator must pass cf_alloc_is_valid().
 *   - for every active class i: cf_pool_is_valid() must hold and
 *     class[i].class_slot_size must equal class[i].pool.slot_size.
 *
 * @param slab  Slab to inspect. May be CF_NULL.
 * @return  CF_TRUE  if the slab is internally consistent,
 *          CF_FALSE otherwise.
 */
cf_bool cf_slab_is_valid(cf_slab *slab);

/********************************************************************/
/* lifecycle                                                        */
/********************************************************************/

/**
 * cf_slab_new
 *
 * Initialise @p slab with up to @p n size classes described by the
 * parallel arrays @p slots_total and @p slots_size.
 *
 * Duplicate size values are merged: only one class is created per
 * unique slot size, and its pool capacity is set to the maximum
 * @p slots_total seen for that size.
 *
 * @param slab         Output parameter. Must not be CF_NULL.
 * @param slots_total  Array of @p n slot counts, one per class.
 * @param slots_size   Array of @p n slot sizes in bytes, one per class.
 * @param n            Number of entries in both arrays.
 *                     Must be <= CF_SLAB_MAX_CLASS.
 * @return  CF_OK          on success.
 *          CF_ERR_NULL    if @p slab is CF_NULL.
 *          CF_ERR_INVALID if @p n > CF_SLAB_MAX_CLASS.
 *          CF_ERR_OOM     if any underlying cf_pool_new() fails
 *                         (the slab is destroyed before returning).
 */
cf_status cf_slab_new(cf_slab *slab, cf_usize slots_total[], cf_usize slots_size[], cf_usize n);

/**
 * cf_slab_destroy
 *
 * Destroy every active size-class pool and reset the slab to its
 * zero-initialised state.
 * The slab may be re-initialised with cf_slab_new() afterwards.
 * Passing CF_NULL is safe and has no effect.
 * If cf_slab_is_valid() returns CF_FALSE the function returns
 * immediately without modifying memory.
 *
 * @param slab  Slab to destroy. May be CF_NULL.
 */
void cf_slab_destroy(cf_slab *slab);

/********************************************************************/
/* operations                                                       */
/********************************************************************/

/**
 * cf_slab_reset
 *
 * Reset every size-class pool so all slots become available again,
 * without freeing any backing memory.
 * All previously returned pointers become invalid after this call.
 *
 * @param slab  Slab to reset. Must not be CF_NULL.
 * @return  CF_OK        on success.
 *          CF_ERR_NULL  if @p slab is CF_NULL.
 *          CF_ERR_STATE if cf_slab_is_valid() returns CF_FALSE, or if
 *                       any internal cf_pool_reset() call fails.
 */
cf_status cf_slab_reset(cf_slab *slab);

#endif /* CF_SLAB_L */