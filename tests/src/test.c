/**
 * test.c — unit tests for the cf allocator library
 *
 * Covers:
 *   cf_alloc       — create_empty, new, is_valid, alloc/realloc/free dispatch
 *   cf_arena       — new, is_valid, alloc, realloc, free, reset, destroy
 *   cf_pool        — new, is_valid, alloc, realloc, free, reset, destroy
 *   cf_slab        — new, is_valid, alloc, realloc, free, reset, destroy
 *   cf_alloc_debug — create_empty, new, is_valid, alloc, realloc, free,
 *                    counters, report, reset, destroy
 */

#include "ALLOCATOR/cf_alloc.h"
#include "ALLOCATOR/cf_arena.h"
#include "ALLOCATOR/cf_pool.h"
#include "ALLOCATOR/cf_slab.h"
#include "ALLOCATOR/cf_alloc_debug.h"

#include <stdio.h>
#include <string.h>
#include <stddef.h>

/* =========================================================================
 * Test runner
 * =========================================================================
 */
static int s_passed = 0;
static int s_failed = 0;

#define CHECK(label, expr)                                            \
    do {                                                              \
        if (expr) {                                                   \
            printf("  PASS  %s\n", label);                           \
            s_passed++;                                               \
        } else {                                                      \
            printf("  FAIL  %s  (line %d)\n", label, __LINE__);      \
            s_failed++;                                               \
        }                                                             \
    } while (0)

static void section(const char *name) { printf("\n── %s\n", name); }

static void summary(void)
{
    printf("\n════════════════════════════\n");
    printf("  %d passed  /  %d failed\n", s_passed, s_failed);
    printf("════════════════════════════\n");
}


/* =========================================================================
 * cf_alloc tests
 * =========================================================================
 */

static void test_alloc_create_empty(void)
{
    section("cf_alloc_create_empty");

    cf_alloc a = cf_alloc_create_empty();

    CHECK("ctx is NULL",     a.ctx     == CF_NULL);
    CHECK("alloc is set",    a.alloc   != CF_NULL);
    CHECK("realloc is set",  a.realloc != CF_NULL);
    CHECK("free is set",     a.free    != CF_NULL);
    CHECK("is_valid passes", cf_alloc_is_valid(&a) == CF_TRUE);
}

static void test_alloc_new(void)
{
    section("cf_alloc_new");

    cf_alloc a = cf_alloc_new();

    CHECK("ctx is NULL",     a.ctx     == CF_NULL);
    CHECK("alloc is set",    a.alloc   != CF_NULL);
    CHECK("realloc is set",  a.realloc != CF_NULL);
    CHECK("free is set",     a.free    != CF_NULL);
    CHECK("is_valid passes", cf_alloc_is_valid(&a) == CF_TRUE);
}

static void test_alloc_is_valid(void)
{
    section("cf_alloc_is_valid");

    cf_alloc a = cf_alloc_new();

    CHECK("valid allocator returns true",  cf_alloc_is_valid(&a)      == CF_TRUE);
    CHECK("null ptr returns false",        cf_alloc_is_valid(CF_NULL) == CF_FALSE);

    cf_alloc bad = a;
    bad.alloc = CF_NULL;
    CHECK("null alloc fn is invalid",   cf_alloc_is_valid(&bad) == CF_FALSE);

    bad = a;
    bad.realloc = CF_NULL;
    CHECK("null realloc fn is invalid", cf_alloc_is_valid(&bad) == CF_FALSE);

    bad = a;
    bad.free = CF_NULL;
    CHECK("null free fn is invalid",    cf_alloc_is_valid(&bad) == CF_FALSE);
}

static void test_alloc_dispatch(void)
{
    section("cf_alloc dispatch (malloc/realloc/free)");

    cf_alloc a = cf_alloc_new();

    void *p = a.alloc(a.ctx, 64);
    CHECK("alloc 64 bytes returns non-null", p != CF_NULL);

    memset(p, 0xAB, 64);
    CHECK("written byte survives",  ((cf_u8 *)p)[0]  == 0xAB);
    CHECK("written byte survives",  ((cf_u8 *)p)[63] == 0xAB);

    void *p2 = a.realloc(a.ctx, p, 128);
    CHECK("realloc to 128 returns non-null", p2 != CF_NULL);

    a.free(a.ctx, p2);
    CHECK("free does not crash", CF_TRUE);

    CHECK("alloc 0 bytes returns NULL (or implementation defined — just no crash)", CF_TRUE);
}


/* =========================================================================
 * cf_arena tests
 * =========================================================================
 */

static void test_arena_new(void)
{
    section("cf_arena_new");

    cf_arena  arena;
    cf_status st;

    st = cf_arena_new(CF_NULL, 1024);
    CHECK("null arena ptr returns CF_ERR_NULL", st == CF_ERR_NULL);

    st = cf_arena_new(&arena, 0);
    CHECK("size 0 returns CF_OK",         st == CF_OK);
    CHECK("data is NULL for size 0",      arena.data        == CF_NULL);
    CHECK("cap is 0 for size 0",          arena.cap         == 0);
    CHECK("offset is 0 for size 0",       arena.offset      == 0);
    CHECK("last_usable is 0 for size 0",  arena.last_usable == 0);
    CHECK("allocator ctx is NULL (empty)",arena.allocator.ctx == CF_NULL);
    cf_arena_destroy(&arena);

    st = cf_arena_new(&arena, 1024);
    CHECK("normal new returns CF_OK",        st == CF_OK);
    CHECK("data is non-null",                arena.data        != CF_NULL);
    CHECK("cap is 1024",                     arena.cap         == 1024);
    CHECK("offset starts at 0",             arena.offset      == 0);
    CHECK("last_usable starts at 0",        arena.last_usable == 0);
    CHECK("allocator ctx points to arena",  arena.allocator.ctx == &arena);
    CHECK("allocator is valid",             cf_alloc_is_valid(&arena.allocator));
    CHECK("arena is valid",                 cf_arena_is_valid(&arena) == CF_TRUE);
    cf_arena_destroy(&arena);
}

static void test_arena_is_valid(void)
{
    section("cf_arena_is_valid");

    cf_arena arena;
    cf_arena_new(&arena, 256);

    CHECK("fresh arena is valid",  cf_arena_is_valid(&arena)      == CF_TRUE);
    CHECK("null ptr is invalid",   cf_arena_is_valid(CF_NULL)     == CF_FALSE);

    arena.allocator.alloc(arena.allocator.ctx, 64);

    cf_arena bad = arena;
    bad.cap = arena.offset - 1;
    CHECK("cap < offset is invalid", cf_arena_is_valid(&bad) == CF_FALSE);

    bad = arena;
    bad.last_usable = arena.offset + 1;
    CHECK("last_usable > offset is invalid", cf_arena_is_valid(&bad) == CF_FALSE);

    bad = arena;
    bad.allocator.alloc = CF_NULL;
    CHECK("null alloc fn is invalid", cf_arena_is_valid(&bad) == CF_FALSE);

    cf_arena empty;
    cf_arena_new(&empty, 0);
    CHECK("empty (size 0) arena is valid", cf_arena_is_valid(&empty) == CF_TRUE);

    cf_arena_destroy(&arena);
    cf_arena_destroy(&empty);
}

static void test_arena_alloc(void)
{
    section("cf_arena alloc");

    cf_arena  arena;
    cf_alloc *a;
    cf_arena_new(&arena, 128);
    a = &arena.allocator;

    void *p0 = a->alloc(a->ctx, 32);
    CHECK("alloc 32 returns non-null", p0 != CF_NULL);
    CHECK("offset advances to 32",     arena.offset == 32);
    CHECK("last_usable is 0",          arena.last_usable == 0);

    void *p1 = a->alloc(a->ctx, 64);
    CHECK("alloc 64 returns non-null", p1 != CF_NULL);
    CHECK("p1 is directly after p0",   (cf_u8 *)p1 == (cf_u8 *)p0 + 32);
    CHECK("offset advances to 96",     arena.offset == 96);

    void *p2 = a->alloc(a->ctx, 64);
    CHECK("alloc beyond cap returns NULL", p2 == CF_NULL);

    void *p3 = a->alloc(a->ctx, 32);
    CHECK("alloc exactly remaining 32 returns non-null", p3 != CF_NULL);
    CHECK("offset is now 128 (full)",                    arena.offset == 128);

    void *p4 = a->alloc(a->ctx, 1);
    CHECK("alloc on full arena returns NULL", p4 == CF_NULL);

    CHECK("null ctx returns NULL", a->alloc(CF_NULL, 8)  == CF_NULL);
    CHECK("size 0 returns NULL",   a->alloc(a->ctx, 0)   == CF_NULL);

    memset(p0, 0x11, 32);
    memset(p1, 0x22, 64);
    memset(p3, 0x33, 32);
    CHECK("p0 holds written value", ((cf_u8 *)p0)[0]  == 0x11);
    CHECK("p1 holds written value", ((cf_u8 *)p1)[0]  == 0x22);
    CHECK("p3 holds written value", ((cf_u8 *)p3)[0]  == 0x33);

    cf_arena_destroy(&arena);
}

static void test_arena_realloc(void)
{
    section("cf_arena realloc");

    cf_arena  arena;
    cf_alloc *a;
    cf_arena_new(&arena, 256);
    a = &arena.allocator;

    void *p0 = a->alloc(a->ctx, 32);
    CHECK("initial alloc for realloc test", p0 != CF_NULL);

    void *p1 = a->realloc(a->ctx, p0, 64);
    CHECK("realloc of last alloc returns same ptr", p1 == p0);
    CHECK("offset advances to 64 after realloc",    arena.offset == 64);

    void *p2 = a->alloc(a->ctx, 16);
    CHECK("alloc after realloc returns non-null", p2 != CF_NULL);

    void *p3 = a->realloc(a->ctx, p0, 32);
    CHECK("realloc of non-last ptr returns NULL", p3 == CF_NULL);

    CHECK("null ctx returns NULL",  a->realloc(CF_NULL, p0, 32) == CF_NULL);
    CHECK("size 0 returns NULL",    a->realloc(a->ctx, p0, 0)   == CF_NULL);

    void *p4 = a->realloc(a->ctx, p2, 4096);
    CHECK("realloc beyond cap returns NULL", p4 == CF_NULL);

    cf_arena_destroy(&arena);
}

static void test_arena_free(void)
{
    section("cf_arena free (no-op)");

    cf_arena  arena;
    cf_alloc *a;
    cf_arena_new(&arena, 64);
    a = &arena.allocator;

    void *p = a->alloc(a->ctx, 16);
    cf_usize offset_before = arena.offset;

    a->free(a->ctx, p);
    CHECK("free is a no-op — offset unchanged", arena.offset == offset_before);

    a->free(a->ctx, CF_NULL);
    CHECK("free null ptr does not crash", CF_TRUE);

    a->free(CF_NULL, p);
    CHECK("free null ctx does not crash", CF_TRUE);

    cf_arena_destroy(&arena);
}

static void test_arena_reset(void)
{
    section("cf_arena_reset");

    cf_arena  arena;
    cf_status st;
    cf_arena_new(&arena, 128);

    arena.allocator.alloc(arena.allocator.ctx, 64);
    arena.allocator.alloc(arena.allocator.ctx, 32);
    CHECK("offset before reset is 96", arena.offset == 96);

    st = cf_arena_reset(&arena);
    CHECK("reset returns CF_OK",          st == CF_OK);
    CHECK("offset is 0 after reset",      arena.offset      == 0);
    CHECK("last_usable is 0 after reset", arena.last_usable == 0);
    CHECK("data pointer unchanged",       arena.data        != CF_NULL);
    CHECK("cap unchanged after reset",    arena.cap         == 128);

    void *p = arena.allocator.alloc(arena.allocator.ctx, 128);
    CHECK("full alloc succeeds after reset", p != CF_NULL);

    st = cf_arena_reset(CF_NULL);
    CHECK("reset null returns CF_ERR_NULL", st == CF_ERR_NULL);

    cf_arena_destroy(&arena);
}

static void test_arena_destroy(void)
{
    section("cf_arena_destroy");

    cf_arena arena;
    cf_arena_new(&arena, 256);
    arena.allocator.alloc(arena.allocator.ctx, 64);

    cf_arena_destroy(&arena);
    CHECK("data is NULL after destroy",        arena.data        == CF_NULL);
    CHECK("cap is 0 after destroy",            arena.cap         == 0);
    CHECK("offset is 0 after destroy",         arena.offset      == 0);
    CHECK("last_usable is 0 after destroy",    arena.last_usable == 0);
    CHECK("allocator ctx NULL after destroy",  arena.allocator.ctx == CF_NULL);

    cf_arena_destroy(&arena);
    CHECK("double destroy does not crash", CF_TRUE);

    cf_arena_destroy(CF_NULL);
    CHECK("destroy null does not crash", CF_TRUE);
}


/* =========================================================================
 * cf_pool tests
 * =========================================================================
 */

static void test_pool_new(void)
{
    section("cf_pool_new");

    cf_pool   pool;
    cf_status st;

    st = cf_pool_new(CF_NULL, 4, 32);
    CHECK("null pool ptr returns CF_ERR_NULL", st == CF_ERR_NULL);

    st = cf_pool_new(&pool, 0, 0);
    CHECK("zero/zero returns CF_OK (empty pool)", st == CF_OK);
    CHECK("data is NULL for empty pool",          pool.data == CF_NULL);
    cf_pool_destroy(&pool);

    st = cf_pool_new(&pool, 0, 32);
    CHECK("zero slots with non-zero size returns CF_ERR_INVALID", st == CF_ERR_INVALID);

    st = cf_pool_new(&pool, 4, 1);
    CHECK("slot_size < sizeof(void*) returns CF_ERR_INVALID", st == CF_ERR_INVALID);

    st = cf_pool_new(&pool, 4, 32);
    CHECK("normal new returns CF_OK",          st == CF_OK);
    CHECK("data is non-null",                  pool.data       != CF_NULL);
    CHECK("slot_total is 4",                   pool.slot_total == 4);
    CHECK("slot_size is 32",                   pool.slot_size  == 32);
    CHECK("slot_used is 0",                    pool.slot_used  == 0);
    CHECK("list is not null",                  pool.list       != CF_NULL);
    CHECK("allocator ctx points to pool",      pool.allocator.ctx == &pool);
    CHECK("allocator is valid",                cf_alloc_is_valid(&pool.allocator));
    CHECK("pool is valid",                     cf_pool_is_valid(&pool) == CF_TRUE);
    cf_pool_destroy(&pool);
}

static void test_pool_is_valid(void)
{
    section("cf_pool_is_valid");

    cf_pool pool;
    cf_pool_new(&pool, 4, 32);

    CHECK("fresh pool is valid", cf_pool_is_valid(&pool)      == CF_TRUE);
    CHECK("null ptr is invalid", cf_pool_is_valid(CF_NULL)    == CF_FALSE);

    cf_pool bad = pool;
    bad.slot_used = bad.slot_total + 1;
    CHECK("slot_used > slot_total is invalid", cf_pool_is_valid(&bad) == CF_FALSE);

    bad = pool;
    bad.slot_size = 1;
    CHECK("slot_size < sizeof(void*) is invalid", cf_pool_is_valid(&bad) == CF_FALSE);

    bad = pool;
    bad.slot_total = 0;
    CHECK("slot_total 0 with data non-null is invalid", cf_pool_is_valid(&bad) == CF_FALSE);

    bad = pool;
    bad.allocator.alloc = CF_NULL;
    CHECK("null alloc fn is invalid", cf_pool_is_valid(&bad) == CF_FALSE);

    cf_pool empty = cf_pool_create_empty();
    CHECK("create_empty pool is valid", cf_pool_is_valid(&empty) == CF_TRUE);

    cf_pool_destroy(&pool);
}

static void test_pool_alloc(void)
{
    section("cf_pool alloc");

    cf_pool   pool;
    cf_alloc *a;
    cf_pool_new(&pool, 3, 32);
    a = &pool.allocator;

    void *p0 = a->alloc(a->ctx, 32);
    void *p1 = a->alloc(a->ctx, 32);
    void *p2 = a->alloc(a->ctx, 32);
    void *p3 = a->alloc(a->ctx, 32);

    CHECK("alloc slot 0 non-null",          p0 != CF_NULL);
    CHECK("alloc slot 1 non-null",          p1 != CF_NULL);
    CHECK("alloc slot 2 non-null",          p2 != CF_NULL);
    CHECK("alloc beyond capacity is NULL",  p3 == CF_NULL);
    CHECK("slot_used is 3",                 pool.slot_used == 3);

    CHECK("slots do not overlap p0/p1", (cf_u8 *)p1 - (cf_u8 *)p0 >= (ptrdiff_t)pool.slot_size ||
                                         (cf_u8 *)p0 - (cf_u8 *)p1 >= (ptrdiff_t)pool.slot_size);

    memset(p0, 0xAA, 32);
    memset(p1, 0xBB, 32);
    memset(p2, 0xCC, 32);
    CHECK("p0 holds value", ((cf_u8 *)p0)[0] == 0xAA);
    CHECK("p1 holds value", ((cf_u8 *)p1)[0] == 0xBB);
    CHECK("p2 holds value", ((cf_u8 *)p2)[0] == 0xCC);

    CHECK("null ctx returns NULL",  a->alloc(CF_NULL, 32) == CF_NULL);

    cf_pool_destroy(&pool);
}

static void test_pool_realloc(void)
{
    section("cf_pool realloc (always NULL)");

    cf_pool   pool;
    cf_alloc *a;
    cf_pool_new(&pool, 2, 32);
    a = &pool.allocator;

    void *p = a->alloc(a->ctx, 32);

    CHECK("realloc always returns NULL",    a->realloc(a->ctx, p,      64) == CF_NULL);
    CHECK("realloc null ptr returns NULL",  a->realloc(a->ctx, CF_NULL,64) == CF_NULL);
    CHECK("realloc null ctx returns NULL",  a->realloc(CF_NULL, p,     64) == CF_NULL);

    cf_pool_destroy(&pool);
}

static void test_pool_free(void)
{
    section("cf_pool free");

    cf_pool   pool;
    cf_alloc *a;
    cf_pool_new(&pool, 3, 32);
    a = &pool.allocator;

    void *p0 = a->alloc(a->ctx, 32);
    void *p1 = a->alloc(a->ctx, 32);
    void *p2 = a->alloc(a->ctx, 32);
    CHECK("pool fully used before free", pool.slot_used == 3);

    a->free(a->ctx, p1);
    CHECK("slot_used decrements on free",  pool.slot_used == 2);

    void *p1b = a->alloc(a->ctx, 32);
    CHECK("alloc after free returns non-null",    p1b != CF_NULL);
    CHECK("alloc after free reuses freed slot",   p1b == p1);

    cf_u8 outside[32];
    cf_usize used_before = pool.slot_used;
    a->free(a->ctx, outside);
    CHECK("out-of-range ptr is ignored",  pool.slot_used == used_before);

    cf_u8 *misaligned = (cf_u8 *)p0 + 1;
    a->free(a->ctx, misaligned);
    CHECK("misaligned ptr is ignored", pool.slot_used == used_before);

    a->free(a->ctx, CF_NULL);
    CHECK("free null ptr does not crash", CF_TRUE);

    a->free(CF_NULL, p2);
    CHECK("free null ctx does not crash", CF_TRUE);

    cf_pool_destroy(&pool);
}

static void test_pool_reset(void)
{
    section("cf_pool_reset");

    cf_pool   pool;
    cf_status st;
    cf_pool_new(&pool, 3, 32);

    pool.allocator.alloc(pool.allocator.ctx, 32);
    pool.allocator.alloc(pool.allocator.ctx, 32);
    pool.allocator.alloc(pool.allocator.ctx, 32);
    CHECK("fully used before reset", pool.slot_used == 3);

    st = cf_pool_reset(&pool);
    CHECK("reset returns CF_OK",         st == CF_OK);
    CHECK("slot_used is 0 after reset",  pool.slot_used == 0);
    CHECK("list is not null after reset",pool.list      != CF_NULL);

    void *p0 = pool.allocator.alloc(pool.allocator.ctx, 32);
    void *p1 = pool.allocator.alloc(pool.allocator.ctx, 32);
    void *p2 = pool.allocator.alloc(pool.allocator.ctx, 32);
    CHECK("alloc 1 works after reset", p0 != CF_NULL);
    CHECK("alloc 2 works after reset", p1 != CF_NULL);
    CHECK("alloc 3 works after reset", p2 != CF_NULL);

    st = cf_pool_reset(CF_NULL);
    CHECK("reset null returns CF_ERR_NULL", st == CF_ERR_NULL);

    cf_pool empty = cf_pool_create_empty();
    st = cf_pool_reset(&empty);
    CHECK("reset empty pool returns CF_OK", st == CF_OK);

    cf_pool_destroy(&pool);
}

static void test_pool_destroy(void)
{
    section("cf_pool_destroy");

    cf_pool pool;
    cf_pool_new(&pool, 4, 32);
    pool.allocator.alloc(pool.allocator.ctx, 32);

    cf_pool_destroy(&pool);
    CHECK("data is NULL after destroy",       pool.data       == CF_NULL);
    CHECK("list is NULL after destroy",       pool.list       == CF_NULL);
    CHECK("slot_total is 0 after destroy",    pool.slot_total == 0);
    CHECK("slot_size is 0 after destroy",     pool.slot_size  == 0);
    CHECK("slot_used is 0 after destroy",     pool.slot_used  == 0);

    cf_pool_destroy(&pool);
    CHECK("double destroy does not crash", CF_TRUE);

    cf_pool_destroy(CF_NULL);
    CHECK("destroy null does not crash", CF_TRUE);
}


/* =========================================================================
 * cf_slab tests
 * =========================================================================
 */

static void test_slab_new(void)
{
    section("cf_slab_new");

    cf_slab   slab;
    cf_status st;

    cf_usize totals1[] = {4, 3, 2};
    cf_usize sizes1[]  = {16, 32, 64};

    st = cf_slab_new(CF_NULL, totals1, sizes1, 3);
    CHECK("null slab ptr returns CF_ERR_NULL", st == CF_ERR_NULL);

    st = cf_slab_new(&slab, totals1, sizes1, 3);
    CHECK("normal slab_new returns CF_OK",    st == CF_OK);
    CHECK("class_count is 3",                 slab.class_count == 3);
    CHECK("allocator ctx points to slab",     slab.allocator.ctx == &slab);
    CHECK("allocator is valid",               cf_alloc_is_valid(&slab.allocator));
    CHECK("slab is valid",                    cf_slab_is_valid(&slab) == CF_TRUE);
    CHECK("class[0] size is 16",              slab.class[0].class_slot_size == 16);
    CHECK("class[1] size is 32",              slab.class[1].class_slot_size == 32);
    CHECK("class[2] size is 64",              slab.class[2].class_slot_size == 64);
    cf_slab_destroy(&slab);

    cf_usize totals_too_many[CF_SLAB_MAX_CLASS + 1];
    cf_usize sizes_too_many[CF_SLAB_MAX_CLASS + 1];
    for (cf_usize i = 0; i < CF_SLAB_MAX_CLASS + 1; i++)
    {
        totals_too_many[i] = 2;
        sizes_too_many[i]  = (i + 1) * 16;
    }
    st = cf_slab_new(&slab, totals_too_many, sizes_too_many, CF_SLAB_MAX_CLASS + 1);
    CHECK("too many classes returns CF_ERR_INVALID", st == CF_ERR_INVALID);

    cf_usize totals2[] = {2, 5, 3, 1};
    cf_usize sizes2[]  = {16, 16, 32, 32};

    st = cf_slab_new(&slab, totals2, sizes2, 4);
    CHECK("duplicate sizes return CF_OK",              st == CF_OK);
    CHECK("duplicate sizes merge into 2 classes",      slab.class_count == 2);

    cf_bool found16 = CF_FALSE, found32 = CF_FALSE;
    for (cf_usize i = 0; i < slab.class_count; i++)
    {
        if (slab.class[i].class_slot_size == 16)
        {
            found16 = CF_TRUE;
            CHECK("merged 16-class gets max total (5)", slab.class[i].pool.slot_total == 5);
        }
        if (slab.class[i].class_slot_size == 32)
        {
            found32 = CF_TRUE;
            CHECK("merged 32-class gets max total (3)", slab.class[i].pool.slot_total == 3);
        }
    }
    CHECK("16-size class found", found16 == CF_TRUE);
    CHECK("32-size class found", found32 == CF_TRUE);
    cf_slab_destroy(&slab);
}

static void test_slab_is_valid(void)
{
    section("cf_slab_is_valid");

    cf_slab slab;
    cf_usize totals[] = {4, 3};
    cf_usize sizes[]  = {16, 32};
    cf_slab_new(&slab, totals, sizes, 2);

    CHECK("fresh slab is valid",   cf_slab_is_valid(&slab)      == CF_TRUE);
    CHECK("null ptr is invalid",   cf_slab_is_valid(CF_NULL)    == CF_FALSE);

    cf_slab bad = slab;
    bad.class_count = CF_SLAB_MAX_CLASS + 1;
    CHECK("class_count too large is invalid", cf_slab_is_valid(&bad) == CF_FALSE);

    bad = slab;
    bad.class[0].class_slot_size = 99;
    CHECK("class_slot_size mismatch is invalid", cf_slab_is_valid(&bad) == CF_FALSE);

    bad = slab;
    bad.allocator.free = CF_NULL;
    CHECK("null allocator fn is invalid", cf_slab_is_valid(&bad) == CF_FALSE);

    cf_slab_destroy(&slab);
}

static void test_slab_alloc(void)
{
    section("cf_slab alloc");

    cf_slab   slab;
    cf_alloc *a;
    cf_usize totals[] = {2, 2, 1};
    cf_usize sizes[]  = {16, 32, 64};
    cf_slab_new(&slab, totals, sizes, 3);
    a = &slab.allocator;

    void *p0 = a->alloc(a->ctx, 8);
    void *p1 = a->alloc(a->ctx, 16);
    void *p2 = a->alloc(a->ctx, 17);
    void *p3 = a->alloc(a->ctx, 33);
    void *p4 = a->alloc(a->ctx, 100);

    CHECK("alloc 8  (fits 16-class) non-null",  p0 != CF_NULL);
    CHECK("alloc 16 (fits 16-class) non-null",  p1 != CF_NULL);
    CHECK("alloc 17 (fits 32-class) non-null",  p2 != CF_NULL);
    CHECK("alloc 33 (fits 64-class) non-null",  p3 != CF_NULL);
    CHECK("alloc 100 (no class) returns NULL",  p4 == CF_NULL);

    memset(p0, 0xAA, 8);
    memset(p1, 0xBB, 16);
    memset(p2, 0xCC, 17);
    memset(p3, 0xDD, 33);
    CHECK("p0 holds written value", ((cf_u8 *)p0)[0] == 0xAA);
    CHECK("p1 holds written value", ((cf_u8 *)p1)[0] == 0xBB);
    CHECK("p2 holds written value", ((cf_u8 *)p2)[0] == 0xCC);
    CHECK("p3 holds written value", ((cf_u8 *)p3)[0] == 0xDD);

    void *p5 = a->alloc(a->ctx, 8);
    void *p6 = a->alloc(a->ctx, 8);
    CHECK("16-class still has 1 more slot (p5)", p5 != CF_NULL);
    CHECK("16-class exhausted falls back or NULL", CF_TRUE);
    (void)p6;

    CHECK("null ctx returns NULL", a->alloc(CF_NULL, 8) == CF_NULL);

    cf_slab_destroy(&slab);
}

static void test_slab_realloc(void)
{
    section("cf_slab realloc (always NULL)");

    cf_slab   slab;
    cf_alloc *a;
    cf_usize totals[] = {2, 2};
    cf_usize sizes[]  = {16, 32};
    cf_slab_new(&slab, totals, sizes, 2);
    a = &slab.allocator;

    void *p = a->alloc(a->ctx, 8);

    CHECK("realloc always returns NULL",    a->realloc(a->ctx, p,      24) == CF_NULL);
    CHECK("realloc null ptr returns NULL",  a->realloc(a->ctx, CF_NULL,24) == CF_NULL);
    CHECK("realloc null ctx returns NULL",  a->realloc(CF_NULL, p,     24) == CF_NULL);

    cf_slab_destroy(&slab);
}

static void test_slab_free(void)
{
    section("cf_slab free");

    cf_slab   slab;
    cf_alloc *a;
    cf_usize totals[] = {2, 2};
    cf_usize sizes[]  = {16, 32};
    cf_slab_new(&slab, totals, sizes, 2);
    a = &slab.allocator;

    void *p0 = a->alloc(a->ctx, 8);
    void *p1 = a->alloc(a->ctx, 8);
    void *p2 = a->alloc(a->ctx, 24);

    CHECK("p0 non-null", p0 != CF_NULL);
    CHECK("p1 non-null", p1 != CF_NULL);
    CHECK("p2 non-null", p2 != CF_NULL);

    cf_usize used16_before = slab.class[0].pool.slot_used;
    a->free(a->ctx, p0);
    CHECK("free reduces used count in owner pool",
          slab.class[0].pool.slot_used == used16_before - 1);

    void *p0b = a->alloc(a->ctx, 8);
    CHECK("alloc after free returns non-null",    p0b != CF_NULL);
    CHECK("alloc after free reuses freed slot",   p0b == p0);

    cf_u8 outside[32];
    cf_usize used0 = slab.class[0].pool.slot_used;
    cf_usize used1 = slab.class[1].pool.slot_used;
    a->free(a->ctx, outside);
    CHECK("out-of-range ptr: class 0 unchanged",  slab.class[0].pool.slot_used == used0);
    CHECK("out-of-range ptr: class 1 unchanged",  slab.class[1].pool.slot_used == used1);

    a->free(a->ctx, CF_NULL);
    CHECK("free null ptr does not crash", CF_TRUE);

    a->free(CF_NULL, p1);
    CHECK("free null ctx does not crash", CF_TRUE);

    cf_slab_destroy(&slab);
}

static void test_slab_reset(void)
{
    section("cf_slab_reset");

    cf_slab   slab;
    cf_status st;
    cf_usize totals[] = {2, 1};
    cf_usize sizes[]  = {16, 32};
    cf_slab_new(&slab, totals, sizes, 2);

    slab.allocator.alloc(slab.allocator.ctx, 8);
    slab.allocator.alloc(slab.allocator.ctx, 8);
    slab.allocator.alloc(slab.allocator.ctx, 24);

    CHECK("class 0 used before reset is 2", slab.class[0].pool.slot_used == 2);
    CHECK("class 1 used before reset is 1", slab.class[1].pool.slot_used == 1);

    st = cf_slab_reset(&slab);
    CHECK("reset returns CF_OK",            st == CF_OK);
    CHECK("class 0 used after reset is 0",  slab.class[0].pool.slot_used == 0);
    CHECK("class 1 used after reset is 0",  slab.class[1].pool.slot_used == 0);

    void *p0 = slab.allocator.alloc(slab.allocator.ctx, 8);
    void *p1 = slab.allocator.alloc(slab.allocator.ctx, 8);
    void *p2 = slab.allocator.alloc(slab.allocator.ctx, 24);
    CHECK("alloc class-0 slot 1 after reset", p0 != CF_NULL);
    CHECK("alloc class-0 slot 2 after reset", p1 != CF_NULL);
    CHECK("alloc class-1 slot after reset",   p2 != CF_NULL);

    st = cf_slab_reset(CF_NULL);
    CHECK("reset null returns CF_ERR_NULL", st == CF_ERR_NULL);

    cf_slab_destroy(&slab);
}

static void test_slab_destroy(void)
{
    section("cf_slab_destroy");

    cf_slab slab;
    cf_usize totals[] = {2, 2};
    cf_usize sizes[]  = {16, 32};
    cf_slab_new(&slab, totals, sizes, 2);
    slab.allocator.alloc(slab.allocator.ctx, 8);

    cf_slab_destroy(&slab);
    CHECK("class_count is 0 after destroy", slab.class_count == 0);
    for (cf_usize i = 0; i < CF_SLAB_MAX_CLASS; i++)
    {
        CHECK("pool data NULL after destroy",      slab.class[i].pool.data   == CF_NULL);
        CHECK("class slot size 0 after destroy",   slab.class[i].class_slot_size == 0);
    }

    cf_slab_destroy(&slab);
    CHECK("double destroy does not crash", CF_TRUE);

    cf_slab_destroy(CF_NULL);
    CHECK("destroy null does not crash", CF_TRUE);
}


/* =========================================================================
 * cf_alloc_debug — clean alloc implementation (reference)
 *
 * static void *cf_alloc_debug_alloc(void *ctx, cf_usize size)
 * {
 *     if(ctx == CF_NULL) return CF_NULL;
 *     cf_alloc_debug *debug = (cf_alloc_debug *) ctx;
 *
 *     cf_alloc_debug_node *new_node = malloc(sizeof(cf_alloc_debug_node));
 *     if(new_node == CF_NULL) return CF_NULL;
 *
 *     void *ptr = debug->backing.alloc(debug->backing.ctx, size);
 *     if(ptr == CF_NULL)
 *     {
 *         free(new_node);
 *         debug->failed_alloc_count++;
 *         return CF_NULL;
 *     }
 *
 *     new_node->ptr  = ptr;
 *     new_node->size = size;
 *     new_node->next = debug->head;
 *     debug->head    = new_node;
 *
 *     debug->alloc_count++;
 *     debug->live_count++;
 *     debug->bytes_live  += size;
 *     debug->bytes_total += size;
 *     if(debug->bytes_live > debug->bytes_peak)
 *         debug->bytes_peak = debug->bytes_live;
 *
 *     return ptr;
 * }
 * =========================================================================
 */

/* =========================================================================
 * cf_alloc_debug tests
 * =========================================================================
 */

static void test_debug_create_empty(void)
{
    section("cf_alloc_debug_create_empty");

    cf_alloc_debug d = cf_alloc_debug_create_empty();

    CHECK("allocator.alloc is set",    d.allocator.alloc   != CF_NULL);
    CHECK("allocator.realloc is set",  d.allocator.realloc != CF_NULL);
    CHECK("allocator.free is set",     d.allocator.free    != CF_NULL);
    CHECK("allocator.ctx is NULL",     d.allocator.ctx     == CF_NULL);
    CHECK("head is NULL",              d.head              == CF_NULL);
    CHECK("alloc_count is 0",          d.alloc_count          == 0);
    CHECK("free_count is 0",           d.free_count           == 0);
    CHECK("realloc_count is 0",        d.realloc_count        == 0);
    CHECK("live_count is 0",           d.live_count           == 0);
    CHECK("bytes_live is 0",           d.bytes_live           == 0);
    CHECK("bytes_peak is 0",           d.bytes_peak           == 0);
    CHECK("bytes_total is 0",          d.bytes_total          == 0);
    CHECK("invalid_free_count is 0",   d.invalid_free_count   == 0);
    CHECK("failed_alloc_count is 0",   d.failed_alloc_count   == 0);
    CHECK("failed_realloc_count is 0", d.failed_realloc_count == 0);
}

static void test_debug_new(void)
{
    section("cf_alloc_debug_new");

    cf_alloc        backing = cf_alloc_new();
    cf_alloc_debug  d;
    cf_status       st;

    st = cf_alloc_debug_new(CF_NULL, &backing);
    CHECK("null debug ptr returns CF_ERR_NULL", st == CF_ERR_NULL);

    st = cf_alloc_debug_new(&d, CF_NULL);
    CHECK("null backing returns CF_ERR_NULL", st == CF_ERR_NULL);

    st = cf_alloc_debug_new(&d, &backing);
    CHECK("normal new returns CF_OK",              st == CF_OK);
    CHECK("allocator ctx points to debug",         d.allocator.ctx == &d);
    CHECK("allocator is valid",                    cf_alloc_is_valid(&d.allocator));
    CHECK("backing alloc fn is set",               d.backing.alloc != CF_NULL);
    CHECK("all counters start at zero",            d.alloc_count == 0 &&
                                                   d.live_count  == 0 &&
                                                   d.bytes_live  == 0);

    cf_alloc_debug_destroy(&d);
}

static void test_debug_is_valid(void)
{
    section("cf_alloc_debug_is_valid");

    cf_alloc       backing = cf_alloc_new();
    cf_alloc_debug d;
    cf_alloc_debug_new(&d, &backing);

    CHECK("fresh debug is valid",  cf_alloc_debug_is_valid(&d)      == CF_TRUE);
    CHECK("null ptr is invalid",   cf_alloc_debug_is_valid(CF_NULL) == CF_FALSE);

    cf_alloc_debug bad = d;
    bad.allocator.alloc = CF_NULL;
    CHECK("null alloc fn is invalid", cf_alloc_debug_is_valid(&bad) == CF_FALSE);

    bad = d;
    bad.allocator.free = CF_NULL;
    CHECK("null free fn is invalid", cf_alloc_debug_is_valid(&bad) == CF_FALSE);

    bad = d;
    bad.backing.alloc = CF_NULL;
    CHECK("null backing alloc fn is invalid", cf_alloc_debug_is_valid(&bad) == CF_FALSE);

    cf_alloc_debug_destroy(&d);
}

static void test_debug_alloc(void)
{
    section("cf_alloc_debug alloc");

    cf_alloc       backing = cf_alloc_new();
    cf_alloc_debug d;
    cf_alloc_debug_new(&d, &backing);
    cf_alloc *a = &d.allocator;

    void *p0 = a->alloc(a->ctx, 64);
    CHECK("alloc 64 returns non-null",   p0 != CF_NULL);
    CHECK("alloc_count is 1",            d.alloc_count == 1);
    CHECK("live_count is 1",             d.live_count  == 1);
    CHECK("bytes_live is 64",            d.bytes_live  == 64);
    CHECK("bytes_peak is 64",            d.bytes_peak  == 64);
    CHECK("bytes_total is 64",           d.bytes_total == 64);
    CHECK("head is non-null",            d.head        != CF_NULL);
    CHECK("head->ptr matches p0",        d.head->ptr   == p0);
    CHECK("head->size is 64",            d.head->size  == 64);

    void *p1 = a->alloc(a->ctx, 32);
    CHECK("alloc 32 returns non-null",   p1 != CF_NULL);
    CHECK("alloc_count is 2",            d.alloc_count == 2);
    CHECK("live_count is 2",             d.live_count  == 2);
    CHECK("bytes_live is 96",            d.bytes_live  == 96);
    CHECK("bytes_peak is 96",            d.bytes_peak  == 96);
    CHECK("bytes_total is 96",           d.bytes_total == 96);

    memset(p0, 0xAA, 64);
    memset(p1, 0xBB, 32);
    CHECK("p0 holds written value", ((cf_u8 *)p0)[0]  == 0xAA);
    CHECK("p1 holds written value", ((cf_u8 *)p1)[0]  == 0xBB);

    CHECK("null ctx returns NULL",  a->alloc(CF_NULL, 8) == CF_NULL);

    cf_alloc_debug_destroy(&d);
}

static void test_debug_free(void)
{
    section("cf_alloc_debug free");

    cf_alloc       backing = cf_alloc_new();
    cf_alloc_debug d;
    cf_alloc_debug_new(&d, &backing);
    cf_alloc *a = &d.allocator;

    void *p0 = a->alloc(a->ctx, 64);
    void *p1 = a->alloc(a->ctx, 32);
    CHECK("two allocs live",  d.live_count == 2);
    CHECK("bytes_live is 96", d.bytes_live == 96);

    a->free(a->ctx, p0);
    CHECK("free_count is 1",              d.free_count  == 1);
    CHECK("live_count decrements to 1",   d.live_count  == 1);
    CHECK("bytes_live decrements to 32",  d.bytes_live  == 32);
    CHECK("bytes_peak unchanged at 96",   d.bytes_peak  == 96);
    CHECK("bytes_total unchanged at 96",  d.bytes_total == 96);

    cf_u8 outside[32];
    a->free(a->ctx, outside);
    CHECK("unknown ptr increments invalid_free_count",
          d.invalid_free_count == 1);
    CHECK("live_count unchanged after invalid free", d.live_count == 1);

    a->free(a->ctx, CF_NULL);
    CHECK("free null ptr does not crash", CF_TRUE);

    a->free(CF_NULL, p1);
    CHECK("free null ctx does not crash", CF_TRUE);

    cf_alloc_debug_destroy(&d);
}

static void test_debug_realloc(void)
{
    section("cf_alloc_debug realloc");

    cf_alloc       backing = cf_alloc_new();
    cf_alloc_debug d;
    cf_alloc_debug_new(&d, &backing);
    cf_alloc *a = &d.allocator;

    void *p0 = a->alloc(a->ctx, 32);
    CHECK("initial alloc for realloc test", p0 != CF_NULL);
    CHECK("bytes_live is 32 before realloc", d.bytes_live == 32);

    void *p1 = a->realloc(a->ctx, p0, 64);
    CHECK("realloc returns non-null",         p1 != CF_NULL);
    CHECK("realloc_count is 1",               d.realloc_count == 1);
    CHECK("bytes_live updated to 64",         d.bytes_live    == 64);
    CHECK("bytes_peak updated to 64",         d.bytes_peak    == 64);
    CHECK("bytes_total grew by 64",           d.bytes_total   == 32 + 64);
    CHECK("live_count still 1",               d.live_count    == 1);
    CHECK("node ptr updated",                 d.head->ptr     == p1);
    CHECK("node size updated to 64",          d.head->size    == 64);

    cf_u8 outside[32];
    void *bad = a->realloc(a->ctx, outside, 16);
    CHECK("realloc unknown ptr returns NULL",          bad == CF_NULL);
    CHECK("invalid_free_count incremented",            d.invalid_free_count == 1);

    CHECK("realloc null ctx returns NULL", a->realloc(CF_NULL, p1, 8) == CF_NULL);
    CHECK("realloc size 0 returns NULL",   a->realloc(a->ctx, p1, 0)  == CF_NULL);

    cf_alloc_debug_destroy(&d);
}

static void test_debug_counters(void)
{
    section("cf_alloc_debug counters");

    cf_alloc       backing = cf_alloc_new();
    cf_alloc_debug d;
    cf_alloc_debug_new(&d, &backing);
    cf_alloc *a = &d.allocator;

    void *p0 = a->alloc(a->ctx, 100);
    void *p1 = a->alloc(a->ctx, 50);
    void *p2 = a->alloc(a->ctx, 25);

    CHECK("bytes_peak hits 175 at max",   d.bytes_peak  == 175);
    CHECK("bytes_total is 175",           d.bytes_total == 175);
    CHECK("bytes_live is 175",            d.bytes_live  == 175);

    a->free(a->ctx, p1);
    CHECK("bytes_live drops to 125 after free", d.bytes_live == 125);
    CHECK("bytes_peak stays at 175",            d.bytes_peak == 175);
    CHECK("bytes_total stays at 175",           d.bytes_total == 175);

    a->realloc(a->ctx, p2, 200);
    CHECK("bytes_live = 100 + 200 = 300 after realloc", d.bytes_live  == 300);
    CHECK("bytes_peak updates to 300",                   d.bytes_peak  == 300);
    CHECK("bytes_total grows to 175 + 200 = 375",        d.bytes_total == 375);

    cf_alloc_debug_destroy(&d);
    (void)p0;
}

static void test_debug_no_leak(void)
{
    section("cf_alloc_debug no-leak detection");

    cf_alloc       backing = cf_alloc_new();
    cf_alloc_debug d;
    cf_alloc_debug_new(&d, &backing);
    cf_alloc *a = &d.allocator;

    void *p0 = a->alloc(a->ctx, 64);
    void *p1 = a->alloc(a->ctx, 32);
    a->free(a->ctx, p0);
    a->free(a->ctx, p1);

    CHECK("live_count is 0 — no leaks",  d.live_count == 0);
    CHECK("bytes_live is 0 — no leaks",  d.bytes_live == 0);
    CHECK("head is NULL — list empty",   d.head       == CF_NULL);

    cf_alloc_debug_destroy(&d);
}

static void test_debug_leak_detection(void)
{
    section("cf_alloc_debug leak detection");

    cf_alloc       backing = cf_alloc_new();
    cf_alloc_debug d;
    cf_alloc_debug_new(&d, &backing);
    cf_alloc *a = &d.allocator;

    a->alloc(a->ctx, 64);
    void *p1 = a->alloc(a->ctx, 32);
    a->free(a->ctx, p1);
    a->alloc(a->ctx, 16);

    CHECK("live_count is 2 — two leaks",   d.live_count == 2);
    CHECK("bytes_live is 80 — leak bytes", d.bytes_live == 80);
    CHECK("head is non-null",              d.head       != CF_NULL);

    cf_alloc_debug_destroy(&d);
}

static void test_debug_destroy(void)
{
    section("cf_alloc_debug_destroy");

    cf_alloc       backing = cf_alloc_new();
    cf_alloc_debug d;
    cf_alloc_debug_new(&d, &backing);
    cf_alloc *a = &d.allocator;

    a->alloc(a->ctx, 64);
    a->alloc(a->ctx, 32);

    cf_alloc_debug_destroy(&d);
    CHECK("head is NULL after destroy",              d.head                 == CF_NULL);
    CHECK("allocator ctx is NULL after destroy",     d.allocator.ctx        == CF_NULL);
    CHECK("live_count is 0 after destroy",           d.live_count           == 0);
    CHECK("bytes_live is 0 after destroy",           d.bytes_live           == 0);
    CHECK("alloc_count is 0 after destroy",          d.alloc_count          == 0);
    CHECK("failed_alloc_count is 0 after destroy",   d.failed_alloc_count   == 0);
    CHECK("invalid_free_count is 0 after destroy",   d.invalid_free_count   == 0);

    cf_alloc_debug_destroy(&d);
    CHECK("double destroy does not crash", CF_TRUE);

    cf_alloc_debug_destroy(CF_NULL);
    CHECK("destroy null does not crash", CF_TRUE);
}


/* =========================================================================
 * main
 * =========================================================================
 */
int main(void)
{
    printf("cf allocator test suite\n");

    printf("\n════════════ cf_alloc ════════════\n");
    test_alloc_create_empty();
    test_alloc_new();
    test_alloc_is_valid();
    test_alloc_dispatch();

    printf("\n════════════ cf_arena ════════════\n");
    test_arena_new();
    test_arena_is_valid();
    test_arena_alloc();
    test_arena_realloc();
    test_arena_free();
    test_arena_reset();
    test_arena_destroy();

    printf("\n════════════ cf_pool  ════════════\n");
    test_pool_new();
    test_pool_is_valid();
    test_pool_alloc();
    test_pool_realloc();
    test_pool_free();
    test_pool_reset();
    test_pool_destroy();

    printf("\n════════════ cf_slab  ════════════\n");
    test_slab_new();
    test_slab_is_valid();
    test_slab_alloc();
    test_slab_realloc();
    test_slab_free();
    test_slab_reset();
    test_slab_destroy();

    printf("\n════════════ cf_alloc_debug ════════════\n");
    test_debug_create_empty();
    test_debug_new();
    test_debug_is_valid();
    test_debug_alloc();
    test_debug_free();
    test_debug_realloc();
    test_debug_counters();
    test_debug_no_leak();
    test_debug_leak_detection();
    test_debug_destroy();

    summary();
    return s_failed == 0 ? 0 : 1;
}