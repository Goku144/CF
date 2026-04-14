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

/*
 * test_alloc_new_model.c — unit tests for allocator validation by capability
 *
 * Covers:
 *   cf_alloc       — create_empty, new, is_valid, alloc/realloc/free dispatch
 *   cf_arena       — alloc + realloc required
 *   cf_pool        — alloc + free required
 *   cf_slab        — alloc + free required
 *   cf_alloc_debug — backing allocator valid if alloc exists
 */

#include "ALLOCATOR/cf_alloc.h"
#include "ALLOCATOR/cf_arena.h"
#include "ALLOCATOR/cf_pool.h"
#include "ALLOCATOR/cf_slab.h"
#include "ALLOCATOR/cf_alloc_debug.h"

#include "MEMORY/cf_memory.h"

#include <stdio.h>
#include <string.h>

/* =========================================================================
 * Test runner
 * =========================================================================
 */
static int s_passed = 0;
static int s_failed = 0;

#define CHECK(label, expr)                                            \
    do {                                                              \
        if (expr) {                                                   \
            printf("  PASS  %s\n", label);                            \
            s_passed++;                                               \
        } else {                                                      \
            printf("  FAIL  %s  (line %d)\n", label, __LINE__);       \
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

    CHECK("ctx is NULL",              a.ctx == CF_NULL);
    CHECK("alloc is set",             a.alloc != CF_NULL);
    CHECK("realloc is NULL",          a.realloc == CF_NULL);
    CHECK("free is NULL",             a.free == CF_NULL);
    CHECK("is_valid passes (alloc only)", cf_alloc_is_valid(&a) == CF_TRUE);
}

static void test_alloc_new(void)
{
    section("cf_alloc_new");

    cf_alloc a = cf_alloc_new();

    CHECK("ctx is NULL",         a.ctx == CF_NULL);
    CHECK("alloc is set",        a.alloc != CF_NULL);
    CHECK("realloc is set",      a.realloc != CF_NULL);
    CHECK("free is set",         a.free != CF_NULL);
    CHECK("is_valid passes",     cf_alloc_is_valid(&a) == CF_TRUE);
}

static void test_alloc_is_valid(void)
{
    section("cf_alloc_is_valid");

    cf_alloc a = cf_alloc_new();

    CHECK("valid allocator returns true",  cf_alloc_is_valid(&a) == CF_TRUE);
    CHECK("null ptr returns false",        cf_alloc_is_valid(CF_NULL) == CF_FALSE);

    cf_alloc bad = a;
    bad.alloc = CF_NULL;
    CHECK("null alloc fn is invalid", cf_alloc_is_valid(&bad) == CF_FALSE);

    bad = a;
    bad.realloc = CF_NULL;
    CHECK("null realloc fn is still valid", cf_alloc_is_valid(&bad) == CF_TRUE);

    bad = a;
    bad.free = CF_NULL;
    CHECK("null free fn is still valid", cf_alloc_is_valid(&bad) == CF_TRUE);

    bad = cf_alloc_create_empty();
    CHECK("create_empty allocator is valid", cf_alloc_is_valid(&bad) == CF_TRUE);
}

static void test_alloc_dispatch(void)
{
    section("cf_alloc dispatch");

    cf_alloc a = cf_alloc_new();

    void *p = a.alloc(a.ctx, 64);
    CHECK("alloc 64 bytes returns non-null", p != CF_NULL);

    memset(p, 0xAB, 64);
    CHECK("written byte survives", ((cf_u8 *)p)[0] == 0xAB);
    CHECK("written byte survives", ((cf_u8 *)p)[63] == 0xAB);

    void *p2 = a.realloc(a.ctx, p, 128);
    CHECK("realloc to 128 returns non-null", p2 != CF_NULL);

    a.free(a.ctx, p2);
    CHECK("free does not crash", CF_TRUE);
}

/* =========================================================================
 * cf_arena tests
 * =========================================================================
 */

static void test_arena_validation_model(void)
{
    section("cf_arena validation model");

    cf_arena arena;
    cf_arena_new(&arena, 128);

    CHECK("arena is valid", cf_arena_is_valid(&arena) == CF_TRUE);
    CHECK("arena alloc exists", arena.allocator.alloc != CF_NULL);
    CHECK("arena realloc exists", arena.allocator.realloc != CF_NULL);
    CHECK("arena free may be NULL", arena.allocator.free == CF_NULL);

    cf_arena bad = arena;
    bad.allocator.alloc = CF_NULL;
    CHECK("arena invalid if alloc is NULL", cf_arena_is_valid(&bad) == CF_FALSE);

    bad = arena;
    bad.allocator.realloc = CF_NULL;
    CHECK("arena invalid if realloc is NULL", cf_arena_is_valid(&bad) == CF_FALSE);

    cf_arena_destroy(&arena);
}

/* =========================================================================
 * cf_pool tests
 * =========================================================================
 */

static void test_pool_validation_model(void)
{
    section("cf_pool validation model");

    cf_pool pool;
    cf_pool_new(&pool, 4, 32);

    CHECK("pool is valid", cf_pool_is_valid(&pool) == CF_TRUE);
    CHECK("pool alloc exists", pool.allocator.alloc != CF_NULL);
    CHECK("pool realloc is NULL", pool.allocator.realloc == CF_NULL);
    CHECK("pool free exists", pool.allocator.free != CF_NULL);

    cf_pool bad = pool;
    bad.allocator.alloc = CF_NULL;
    CHECK("pool invalid if alloc is NULL", cf_pool_is_valid(&bad) == CF_FALSE);

    bad = pool;
    bad.allocator.free = CF_NULL;
    CHECK("pool invalid if free is NULL", cf_pool_is_valid(&bad) == CF_FALSE);

    bad = pool;
    bad.allocator.realloc = CF_NULL;
    CHECK("pool still valid if realloc is NULL", cf_pool_is_valid(&bad) == CF_TRUE);

    cf_pool_destroy(&pool);
}

/* =========================================================================
 * cf_slab tests
 * =========================================================================
 */

static void test_slab_validation_model(void)
{
    section("cf_slab validation model");

    cf_slab slab;
    cf_usize totals[] = {2, 2};
    cf_usize sizes[]  = {16, 32};
    cf_slab_new(&slab, totals, sizes, 2);

    CHECK("slab alloc exists", slab.allocator.alloc != CF_NULL);
    CHECK("slab realloc is NULL", slab.allocator.realloc == CF_NULL);
    CHECK("slab free exists", slab.allocator.free != CF_NULL);

    /*
     * If this fails, your current cf_slab_is_valid still wrongly requires
     * realloc != NULL and should be updated to match the new model.
     */
    CHECK("slab is valid under alloc/free model", cf_slab_is_valid(&slab) == CF_TRUE);

    cf_slab bad = slab;
    bad.allocator.alloc = CF_NULL;
    CHECK("slab invalid if alloc is NULL", cf_slab_is_valid(&bad) == CF_FALSE);

    bad = slab;
    bad.allocator.free = CF_NULL;
    CHECK("slab invalid if free is NULL", cf_slab_is_valid(&bad) == CF_FALSE);

    bad = slab;
    bad.allocator.realloc = CF_NULL;
    CHECK("slab still valid if realloc is NULL", cf_slab_is_valid(&bad) == CF_TRUE);

    cf_slab_destroy(&slab);
}

/* =========================================================================
 * cf_alloc_debug tests
 * =========================================================================
 */

static void test_debug_new_model(void)
{
    section("cf_alloc_debug new model");

    cf_alloc backing = cf_alloc_create_empty();
    cf_alloc_debug d;
    cf_status st = cf_alloc_debug_new(&d, &backing);

    CHECK("debug accepts alloc-only backing", st == CF_OK);
    CHECK("allocator ctx points to debug", d.allocator.ctx == &d);
    CHECK("debug object is valid", cf_alloc_debug_is_valid(&d) == CF_TRUE);

    cf_alloc_debug_destroy(&d);
}

static void test_debug_alloc_free_full_backing(void)
{
    section("cf_alloc_debug alloc/free with full backing");

    cf_alloc backing = cf_alloc_new();
    cf_alloc_debug d;
    cf_alloc_debug_new(&d, &backing);

    cf_alloc *a = &d.allocator;

    void *p0 = a->alloc(a->ctx, 64);
    CHECK("alloc returns non-null", p0 != CF_NULL);
    CHECK("alloc_count is 1", d.alloc_count == 1);
    CHECK("live_count is 1", d.live_count == 1);
    CHECK("bytes_live is 64", d.bytes_live == 64);

    a->free(a->ctx, p0);
    CHECK("free_count is 1", d.free_count == 1);
    CHECK("live_count back to 0", d.live_count == 0);
    CHECK("bytes_live back to 0", d.bytes_live == 0);

    cf_alloc_debug_destroy(&d);
}

/* =========================================================================
 * cf_memory tests
 * =========================================================================
 */

static void test_bytes_create_empty(void)
{
    section("cf_bytes_create_empty");

    cf_bytes bytes = cf_bytes_create_empty();

    CHECK("data is NULL", bytes.data == CF_NULL);
    CHECK("len is 0",     bytes.len  == 0);
}

static void test_buffer_create_empty(void)
{
    section("cf_buffer_create_empty");

    cf_buffer buffer = cf_buffer_create_empty();

    CHECK("data is NULL",          buffer.data == CF_NULL);
    CHECK("len is 0",              buffer.len  == 0);
    CHECK("cap is 0",              buffer.cap  == 0);
    CHECK("allocator is valid",    cf_alloc_is_valid(&buffer.allocator) == CF_TRUE);
}

static void test_bytes_is_valid(void)
{
    section("cf_bytes_is_valid");

    cf_bytes bytes = cf_bytes_create_empty();
    CHECK("empty bytes is valid",          cf_bytes_is_valid(&bytes) == CF_TRUE);
    CHECK("null ptr is invalid",           cf_bytes_is_valid(CF_NULL) == CF_FALSE);

    cf_bytes bad;
    bad.data = CF_NULL;
    bad.len  = 4;
    CHECK("NULL data with non-zero len invalid", cf_bytes_is_valid(&bad) == CF_FALSE);

    cf_u8 data[4] = {0};
    bad.data = data;
    bad.len  = 4;
    CHECK("normal bytes is valid",         cf_bytes_is_valid(&bad) == CF_TRUE);
}

static void test_buffer_is_valid(void)
{
    section("cf_buffer_is_valid");

    cf_buffer buffer = cf_buffer_create_empty();
    CHECK("empty buffer is valid",         cf_buffer_is_valid(&buffer) == CF_TRUE);
    CHECK("null ptr is invalid",           cf_buffer_is_valid(CF_NULL) == CF_FALSE);

    cf_buffer bad = buffer;
    bad.data = (cf_u8 *)1;
    CHECK("data non-null with cap 0 invalid", cf_buffer_is_valid(&bad) == CF_FALSE);

    bad = buffer;
    bad.data = (cf_u8 *)1;
    bad.cap  = 4;
    bad.len  = 8;
    CHECK("len > cap invalid",             cf_buffer_is_valid(&bad) == CF_FALSE);

    cf_alloc alloc = cf_alloc_new();
    cf_u8 data[8] = {0};
    bad.data = data;
    bad.len  = 4;
    bad.cap  = 8;
    bad.allocator = alloc;
    CHECK("normal buffer is valid",        cf_buffer_is_valid(&bad) == CF_TRUE);
}

static void test_bytes_is_empty(void)
{
    section("cf_bytes_is_empty");

    cf_bytes bytes = cf_bytes_create_empty();
    CHECK("empty bytes is empty",          cf_bytes_is_empty(&bytes) == CF_TRUE);

    cf_u8 data[1] = {0};
    bytes.data = data;
    bytes.len  = 1;
    CHECK("non-empty bytes is not empty",  cf_bytes_is_empty(&bytes) == CF_FALSE);

    CHECK("null ptr returns false",        cf_bytes_is_empty(CF_NULL) == CF_FALSE);
}

static void test_buffer_is_empty(void)
{
    section("cf_buffer_is_empty");

    cf_buffer buffer = cf_buffer_create_empty();
    CHECK("empty buffer is empty",         cf_buffer_is_empty(&buffer) == CF_TRUE);

    cf_alloc alloc = cf_alloc_new();
    buffer.data = (cf_u8 *)1;
    buffer.len  = 1;
    buffer.cap  = 1;
    buffer.allocator = alloc;
    CHECK("non-empty buffer is not empty", cf_buffer_is_empty(&buffer) == CF_FALSE);

    CHECK("null ptr returns false",        cf_buffer_is_empty(CF_NULL) == CF_FALSE);
}

static void test_bytes_is_eq(void)
{
    section("cf_bytes_is_eq");

    cf_u8 d1[4] = {1,2,3,4};
    cf_u8 d2[4] = {1,2,3,4};
    cf_u8 d3[4] = {1,2,3,9};

    cf_bytes b1 = {d1, 4};
    cf_bytes b2 = {d2, 4};
    cf_bytes b3 = {d3, 4};
    cf_bytes e  = cf_bytes_create_empty();

    CHECK("equal bytes returns true",      cf_bytes_is_eq(&b1, &b2) == CF_TRUE);
    CHECK("different bytes returns false", cf_bytes_is_eq(&b1, &b3) == CF_FALSE);
    CHECK("empty equals empty",            cf_bytes_is_eq(&e, &e)   == CF_TRUE);
    CHECK("null arg returns false",        cf_bytes_is_eq(&b1, CF_NULL) == CF_FALSE);
}

static void test_buffer_is_eq(void)
{
    section("cf_buffer_is_eq");

    cf_alloc alloc = cf_alloc_new();

    cf_buffer b1 = {0};
    cf_buffer b2 = {0};
    cf_buffer b3 = {0};

    cf_u8 d1[4] = {1,2,3,4};
    cf_u8 d2[4] = {1,2,3,4};
    cf_u8 d3[4] = {1,2,3,8};

    b1.data = d1; b1.len = 4; b1.cap = 4; b1.allocator = alloc;
    b2.data = d2; b2.len = 4; b2.cap = 4; b2.allocator = alloc;
    b3.data = d3; b3.len = 4; b3.cap = 4; b3.allocator = alloc;

    CHECK("equal buffers returns true",      cf_buffer_is_eq(&b1, &b2) == CF_TRUE);
    CHECK("different buffers returns false", cf_buffer_is_eq(&b1, &b3) == CF_FALSE);
    CHECK("null arg returns false",          cf_buffer_is_eq(&b1, CF_NULL) == CF_FALSE);
}

static void test_bytes_slice(void)
{
    section("cf_bytes_slice");

    cf_u8 data[6] = {10,20,30,40,50,60};
    cf_bytes src = {data, 6};
    cf_bytes dst;
    cf_status st;

    st = cf_bytes_slice(&dst, &src, 2, 3);
    CHECK("slice returns CF_OK",            st == CF_OK);
    CHECK("slice points to correct offset", dst.data == data + 2);
    CHECK("slice len is 3",                 dst.len == 3);
    CHECK("slice byte[0] correct",          dst.data[0] == 30);
    CHECK("slice byte[2] correct",          dst.data[2] == 50);

    st = cf_bytes_slice(&dst, &src, 6, 0);
    CHECK("empty slice at end returns CF_OK", st == CF_OK);
    CHECK("empty slice at end len is 0",      dst.len == 0);

    st = cf_bytes_slice(&dst, &src, 7, 0);
    CHECK("index > len returns CF_ERR_BOUNDS", st == CF_ERR_BOUNDS);

    st = cf_bytes_slice(&dst, &src, 4, 3);
    CHECK("size past end returns CF_ERR_BOUNDS", st == CF_ERR_BOUNDS);

    st = cf_bytes_slice(CF_NULL, &src, 0, 1);
    CHECK("null dst returns CF_ERR_NULL", st == CF_ERR_NULL);
}

static void test_buffer_slice(void)
{
    section("cf_buffer_slice");

    cf_alloc alloc = cf_alloc_new();
    cf_u8 data[6] = {10,20,30,40,50,60};
    cf_buffer src = {data, 6, 6, alloc};
    cf_bytes dst;
    cf_status st;

    st = cf_buffer_slice(&dst, &src, 1, 4);
    CHECK("slice returns CF_OK",            st == CF_OK);
    CHECK("slice points to correct offset", dst.data == data + 1);
    CHECK("slice len is 4",                 dst.len == 4);
    CHECK("slice byte[0] correct",          dst.data[0] == 20);
    CHECK("slice byte[3] correct",          dst.data[3] == 50);

    st = cf_buffer_slice(&dst, &src, 6, 0);
    CHECK("empty slice at end returns CF_OK", st == CF_OK);
    CHECK("empty slice at end len is 0",      dst.len == 0);

    st = cf_buffer_slice(&dst, &src, 7, 0);
    CHECK("index > len returns CF_ERR_BOUNDS", st == CF_ERR_BOUNDS);

    st = cf_buffer_slice(&dst, &src, 5, 2);
    CHECK("size past end returns CF_ERR_BOUNDS", st == CF_ERR_BOUNDS);
}

static void test_bytes_fill(void)
{
    section("cf_bytes_fill");

    cf_u8 data[6] = {1,2,3,4,5,6};
    cf_bytes bytes = {data, 6};
    cf_status st;

    st = cf_bytes_fill(&bytes, 0xAA, 4);
    CHECK("fill returns CF_OK",            st == CF_OK);
    CHECK("byte 0 changed",                data[0] == 0xAA);
    CHECK("byte 3 changed",                data[3] == 0xAA);
    CHECK("byte 4 unchanged",              data[4] == 5);

    st = cf_bytes_fill(&bytes, 0xBB, 7);
    CHECK("fill past len returns CF_ERR_BOUNDS", st == CF_ERR_BOUNDS);

    st = cf_bytes_fill(CF_NULL, 0xCC, 1);
    CHECK("null bytes returns CF_ERR_NULL", st == CF_ERR_NULL);
}

static void test_buffer_fill(void)
{
    section("cf_buffer_fill");

    cf_alloc alloc = cf_alloc_new();
    cf_u8 data[6] = {1,2,3,4,5,6};
    cf_buffer buffer = {data, 6, 6, alloc};
    cf_status st;

    st = cf_buffer_fill(&buffer, 0x11, 3);
    CHECK("fill returns CF_OK",            st == CF_OK);
    CHECK("byte 0 changed",                data[0] == 0x11);
    CHECK("byte 2 changed",                data[2] == 0x11);
    CHECK("byte 3 unchanged",              data[3] == 4);

    st = cf_buffer_fill(&buffer, 0x22, 9);
    CHECK("fill past len returns CF_ERR_BOUNDS", st == CF_ERR_BOUNDS);
}

static void test_buffer_init(void)
{
    section("cf_buffer_init");

    cf_buffer buffer;
    cf_status st;
    cf_alloc alloc = cf_alloc_new();

    st = cf_buffer_init(CF_NULL, &alloc, 16);
    CHECK("null buffer returns CF_ERR_NULL", st == CF_ERR_NULL);

    st = cf_buffer_init(&buffer, &alloc, 0);
    CHECK("init with zero capacity returns CF_OK", st == CF_OK);
    CHECK("zero-cap data is NULL",                 buffer.data == CF_NULL);
    CHECK("zero-cap len is 0",                     buffer.len == 0);
    CHECK("zero-cap cap is 0",                     buffer.cap == 0);

    st = cf_buffer_init(&buffer, &alloc, 16);
    CHECK("init with capacity returns CF_OK",      st == CF_OK);
    CHECK("data is non-null",                      buffer.data != CF_NULL);
    CHECK("len starts at 0",                       buffer.len == 0);
    CHECK("cap is 16",                             buffer.cap == 16);

    cf_buffer_destroy(&buffer);
}

static void test_buffer_reserve(void)
{
    section("cf_buffer_reserve");

    cf_buffer buffer;
    cf_status st;
    cf_alloc alloc = cf_alloc_new();

    st = cf_buffer_init(&buffer, &alloc, 8);
    CHECK("init returns CF_OK", st == CF_OK);

    st = cf_buffer_reserve(&buffer, 4);
    CHECK("reserve smaller cap returns CF_OK", st == CF_OK);
    CHECK("cap unchanged when already enough", buffer.cap == 8);

    st = cf_buffer_reserve(&buffer, 32);
    CHECK("reserve bigger cap returns CF_OK", st == CF_OK);
    CHECK("cap grows to 32",                  buffer.cap == 32);
    CHECK("len unchanged after reserve",      buffer.len == 0);
    CHECK("data still non-null",              buffer.data != CF_NULL);

    st = cf_buffer_reserve(CF_NULL, 16);
    CHECK("null buffer returns CF_ERR_NULL", st == CF_ERR_NULL);

    cf_buffer_destroy(&buffer);
}

static void test_buffer_clear(void)
{
    section("cf_buffer_clear");

    cf_alloc alloc = cf_alloc_new();
    cf_u8 data[8] = {1,2,3,4,5,6,7,8};
    cf_buffer buffer = {data, 5, 8, alloc};

    cf_buffer_clear(&buffer);
    CHECK("clear sets len to 0", buffer.len == 0);
    CHECK("cap unchanged",       buffer.cap == 8);
    CHECK("data unchanged",      buffer.data == data);

    cf_buffer_clear(CF_NULL);
    CHECK("clear null does not crash", CF_TRUE);
}

static void test_buffer_destroy(void)
{
    section("cf_buffer_destroy");

    cf_buffer buffer;
    cf_alloc alloc = cf_alloc_new();
    cf_status st = cf_buffer_init(&buffer, &alloc, 16);
    CHECK("init returns CF_OK", st == CF_OK);

    cf_buffer_destroy(&buffer);
    CHECK("data NULL after destroy",         buffer.data == CF_NULL);
    CHECK("len 0 after destroy",             buffer.len == 0);
    CHECK("cap 0 after destroy",             buffer.cap == 0);

    cf_buffer_destroy(&buffer);
    CHECK("double destroy does not crash",   CF_TRUE);

    cf_buffer_destroy(CF_NULL);
    CHECK("destroy null does not crash",     CF_TRUE);
}

static void test_buffer_append_bytes(void)
{
    section("cf_buffer_append_bytes");

    cf_alloc alloc = cf_alloc_new();
    cf_buffer buffer = cf_buffer_create_empty();
    cf_status st = cf_buffer_init(&buffer, &alloc, 8);

    CHECK("init returns CF_OK", st == CF_OK);

    cf_u8 src_data[3] = {10, 20, 30};
    cf_bytes src = {src_data, 3};

    st = cf_buffer_append_bytes(&buffer, &src);
    CHECK("append returns CF_OK", st == CF_OK);
    CHECK("len becomes 3",        buffer.len == 3);
    CHECK("byte 0 correct",       buffer.data[0] == 10);
    CHECK("byte 1 correct",       buffer.data[1] == 20);
    CHECK("byte 2 correct",       buffer.data[2] == 30);

    cf_buffer_destroy(&buffer);
}

static void test_buffer_append_byte(void)
{
    section("cf_buffer_append_byte");

    cf_alloc alloc = cf_alloc_new();
    cf_buffer buffer = cf_buffer_create_empty();
    cf_status st = cf_buffer_init(&buffer, &alloc, 4);

    CHECK("init returns CF_OK", st == CF_OK);

    st = cf_buffer_append_byte(&buffer, 0xAB);
    CHECK("append byte returns CF_OK", st == CF_OK);
    CHECK("len becomes 1",             buffer.len == 1);
    CHECK("stored byte correct",       buffer.data[0] == 0xAB);

    st = cf_buffer_append_byte(&buffer, 0xCD);
    CHECK("second append byte returns CF_OK", st == CF_OK);
    CHECK("len becomes 2",                    buffer.len == 2);
    CHECK("stored second byte correct",       buffer.data[1] == 0xCD);

    cf_buffer_destroy(&buffer);
}

static void test_buffer_append_bytes_null(void)
{
    section("cf_buffer_append_bytes_null");

    cf_alloc alloc = cf_alloc_new();
    cf_buffer buffer = cf_buffer_create_empty();
    cf_status st = cf_buffer_init(&buffer, &alloc, 8);

    CHECK("init returns CF_OK", st == CF_OK);

    cf_u8 src_data[2] = {1, 2};
    cf_bytes src = {src_data, 2};

    st = cf_buffer_append_bytes(CF_NULL, &src);
    CHECK("null dst returns CF_ERR_NULL", st == CF_ERR_NULL);

    st = cf_buffer_append_bytes(&buffer, CF_NULL);
    CHECK("null src returns CF_ERR_NULL", st == CF_ERR_NULL);

    cf_buffer_destroy(&buffer);
}

static void test_buffer_append_bytes_invalid_src(void)
{
    section("cf_buffer_append_bytes_invalid_src");

    cf_alloc alloc = cf_alloc_new();
    cf_buffer buffer = cf_buffer_create_empty();
    cf_status st = cf_buffer_init(&buffer, &alloc, 8);

    CHECK("init returns CF_OK", st == CF_OK);

    cf_bytes bad_src = {CF_NULL, 2};

    st = cf_buffer_append_bytes(&buffer, &bad_src);
    CHECK("invalid src returns CF_ERR_STATE", st == CF_ERR_STATE);

    cf_buffer_destroy(&buffer);
}

static void test_buffer_append_bytes_multiple(void)
{
    section("cf_buffer_append_bytes_multiple");

    cf_alloc alloc = cf_alloc_new();
    cf_buffer buffer = cf_buffer_create_empty();
    cf_status st = cf_buffer_init(&buffer, &alloc, 8);

    CHECK("init returns CF_OK", st == CF_OK);

    cf_u8 a_data[2] = {1, 2};
    cf_u8 b_data[3] = {3, 4, 5};

    cf_bytes a = {a_data, 2};
    cf_bytes b = {b_data, 3};

    st = cf_buffer_append_bytes(&buffer, &a);
    CHECK("first append returns CF_OK", st == CF_OK);

    st = cf_buffer_append_bytes(&buffer, &b);
    CHECK("second append returns CF_OK", st == CF_OK);

    CHECK("len becomes 5", buffer.len == 5);
    CHECK("byte 0 correct", buffer.data[0] == 1);
    CHECK("byte 1 correct", buffer.data[1] == 2);
    CHECK("byte 2 correct", buffer.data[2] == 3);
    CHECK("byte 3 correct", buffer.data[3] == 4);
    CHECK("byte 4 correct", buffer.data[4] == 5);

    cf_buffer_destroy(&buffer);
}

static void test_buffer_append_bytes_grow(void)
{
    section("cf_buffer_append_bytes_grow");

    cf_alloc alloc = cf_alloc_new();
    cf_buffer buffer = cf_buffer_create_empty();
    cf_status st = cf_buffer_init(&buffer, &alloc, 4);

    CHECK("init returns CF_OK", st == CF_OK);

    cf_u8 a_data[3] = {1, 2, 3};
    cf_u8 b_data[3] = {4, 5, 6};

    cf_bytes a = {a_data, 3};
    cf_bytes b = {b_data, 3};

    st = cf_buffer_append_bytes(&buffer, &a);
    CHECK("first append returns CF_OK", st == CF_OK);

    st = cf_buffer_append_bytes(&buffer, &b);
    CHECK("second append requiring grow returns CF_OK", st == CF_OK);

    CHECK("len becomes 6", buffer.len == 6);
    CHECK("byte 0 correct", buffer.data[0] == 1);
    CHECK("byte 1 correct", buffer.data[1] == 2);
    CHECK("byte 2 correct", buffer.data[2] == 3);
    CHECK("byte 3 correct", buffer.data[3] == 4);
    CHECK("byte 4 correct", buffer.data[4] == 5);
    CHECK("byte 5 correct", buffer.data[5] == 6);

    cf_buffer_destroy(&buffer);
}

static void test_bytes_copy_as_buffer(void)
{
    section("cf_bytes_copy_as_buffer");

    cf_alloc alloc = cf_alloc_new();
    cf_buffer buffer = cf_buffer_create_empty();
    cf_status st = cf_buffer_init(&buffer, &alloc, 2);

    CHECK("init returns CF_OK", st == CF_OK);

    cf_u8 src_data[4] = {10,20,30,40};
    cf_bytes src = {src_data, 4};

    st = cf_bytes_copy_as_buffer(&buffer, &src);
    CHECK("copy returns CF_OK",     st == CF_OK);
    CHECK("buffer len becomes 4",   buffer.len == 4);
    CHECK("buffer cap >= 4",        buffer.cap >= 4);
    CHECK("byte 0 correct",         buffer.data[0] == 10);
    CHECK("byte 1 correct",         buffer.data[1] == 20);
    CHECK("byte 2 correct",         buffer.data[2] == 30);
    CHECK("byte 3 correct",         buffer.data[3] == 40);

    src_data[0] = 99;
    CHECK("copy is independent from source", buffer.data[0] == 10);

    cf_buffer_destroy(&buffer);
}

static void test_buffer_view_as_bytes(void)
{
    section("cf_buffer_view_as_bytes");

    cf_alloc alloc = cf_alloc_new();
    cf_u8 data[6] = {1,2,3,4,5,6};
    cf_buffer buffer = {data, 4, 6, alloc};
    cf_bytes bytes;
    cf_status st;

    st = cf_buffer_view_as_bytes(&bytes, &buffer);
    CHECK("view returns CF_OK",         st == CF_OK);
    CHECK("view points to same data",   bytes.data == buffer.data);
    CHECK("view len equals buffer len", bytes.len == buffer.len);
    CHECK("view byte 0 correct",        bytes.data[0] == 1);
    CHECK("view byte 3 correct",        bytes.data[3] == 4);

    buffer.data[1] = 99;
    CHECK("view reflects buffer changes", bytes.data[1] == 99);
}

static void test_buffer_truncate(void)
{
    section("cf_buffer_truncate");

    cf_alloc alloc = cf_alloc_new();
    cf_u8 data[8] = {10,20,30,40,50,60,70,80};
    cf_buffer buffer = {data, 5, 8, alloc};
    cf_status st;

    st = cf_buffer_truncate(&buffer, 3);
    CHECK("truncate returns CF_OK",  st == CF_OK);
    CHECK("len becomes 3",           buffer.len == 3);
    CHECK("cap unchanged",           buffer.cap == 8);
    CHECK("data unchanged",          buffer.data == data);
    CHECK("byte 0 unchanged",        buffer.data[0] == 10);
    CHECK("byte 2 unchanged",        buffer.data[2] == 30);

    st = cf_buffer_truncate(&buffer, 3);
    CHECK("truncate same len CF_OK", st == CF_OK);
    CHECK("len still 3",             buffer.len == 3);

    st = cf_buffer_truncate(&buffer, 0);
    CHECK("truncate to zero CF_OK",  st == CF_OK);
    CHECK("len becomes 0",           buffer.len == 0);

    st = cf_buffer_truncate(&buffer, 1);
    CHECK("truncate past len returns CF_ERR_BOUNDS", st == CF_ERR_BOUNDS);

    st = cf_buffer_truncate(CF_NULL, 0);
    CHECK("null buffer returns CF_ERR_NULL", st == CF_ERR_NULL);
}

/* =========================================================================
 * main
 * =========================================================================
 */

int main(void)
{
    printf("cf allocator capability-model test suite\n");

    test_alloc_create_empty();
    test_alloc_new();
    test_alloc_is_valid();
    test_alloc_dispatch();
    test_arena_validation_model();
    test_pool_validation_model();
    test_slab_validation_model();
    test_debug_new_model();
    test_debug_alloc_free_full_backing();

    printf("\n════════════ cf_memory ════════════\n");
    test_bytes_create_empty();
    test_buffer_create_empty();
    test_bytes_is_valid();
    test_buffer_is_valid();
    test_bytes_is_empty();
    test_buffer_is_empty();
    test_bytes_is_eq();
    test_buffer_is_eq();
    test_bytes_slice();
    test_buffer_slice();
    test_bytes_fill();
    test_buffer_fill();
    test_buffer_init();
    test_buffer_reserve();
    test_buffer_clear();
    test_buffer_destroy();
    test_buffer_append_bytes();
    test_buffer_append_byte();
    test_buffer_append_bytes_null();
    test_buffer_append_bytes_invalid_src();
    test_buffer_append_bytes_multiple();
    test_buffer_append_bytes_grow();
    test_bytes_copy_as_buffer();
    test_buffer_view_as_bytes();
    test_buffer_truncate();
    summary();
    return s_failed == 0 ? 0 : 1;
}