#include "ALLOCATOR/cf_alloc.h"
#include "ALLOCATOR/cf_alloc_debug.h"
#include "MEMORY/cf_array.h"
#include "RUNTIME/cf_types.h"
#include "SECURITY/cf_aes.h"
#include "SECURITY/cf_base64.h"
#include "SECURITY/cf_hex.h"
#include "TEXT/cf_ascii.h"
#include "TEXT/cf_string.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct test_alloc_state
{
  cf_usize alloc_calls;
  cf_usize realloc_calls;
  cf_usize free_calls;
  cf_usize fail_alloc_count;
  cf_usize fail_realloc_count;
  cf_usize bad_ctx_alloc_calls;
  cf_usize bad_ctx_realloc_calls;
  cf_usize bad_ctx_free_calls;
  void *expected_ctx;
} test_alloc_state;

static test_alloc_state *g_test_alloc_state = CF_NULL;
static int g_test_passed = 0;
static int g_test_failed = 0;

static void test_section(const char *name)
{
  printf("\n── %s\n", name);
}

static void test_pass(const char *message)
{
  g_test_passed++;
  printf("  PASS  %s\n", message);
}

static void test_check(_Bool condition, const char *message)
{
  if(condition) test_pass(message);
  else
  {
    g_test_failed++;
    printf("  FAIL  %s\n", message);
  }
}

static void test_check_string_value(cf_string *str, const char *expected, const char *message)
{
  test_check(
    str != CF_NULL &&
    expected != CF_NULL &&
    cf_string_is_valid(str) == CF_TRUE &&
    str->data != CF_NULL &&
    strcmp((char *)str->data, expected) == 0 &&
    str->len == strlen(expected),
    message
  );
}

static void *test_alloc(void *ctx, cf_usize size)
{
  if(g_test_alloc_state == CF_NULL) return CF_NULL;

  g_test_alloc_state->alloc_calls++;
  if(ctx != g_test_alloc_state->expected_ctx)
    g_test_alloc_state->bad_ctx_alloc_calls++;

  if(g_test_alloc_state->fail_alloc_count > 0)
  {
    g_test_alloc_state->fail_alloc_count--;
    return CF_NULL;
  }

  return malloc(size);
}

static void *test_realloc(void *ctx, void *ptr, cf_usize size)
{
  if(g_test_alloc_state == CF_NULL) return CF_NULL;

  g_test_alloc_state->realloc_calls++;
  if(ctx != g_test_alloc_state->expected_ctx)
    g_test_alloc_state->bad_ctx_realloc_calls++;

  if(g_test_alloc_state->fail_realloc_count > 0)
  {
    g_test_alloc_state->fail_realloc_count--;
    return CF_NULL;
  }

  return realloc(ptr, size);
}

static void test_free(void *ctx, void *ptr)
{
  if(g_test_alloc_state == CF_NULL) return;

  g_test_alloc_state->free_calls++;
  if(ctx != g_test_alloc_state->expected_ctx)
    g_test_alloc_state->bad_ctx_free_calls++;

  free(ptr);
}

static void test_cf_alloc_debug_new_model(void)
{
  test_alloc_state state = {0};
  cf_alloc backing =
  {
    .ctx = &state,
    .alloc = test_alloc,
    .realloc = test_realloc,
    .free = test_free,
  };
  cf_alloc_debug debug = {0};
  cf_alloc_debug debug_default = {0};
  cf_alloc_debug untouched =
  {
    .ptr_live = 77,
    .ptr_all_live = 88,
    .statement = "unchanged",
  };
  char statement[] = "tests/src/test.c cf_alloc_debug test";

  g_test_alloc_state = &state;
  state.expected_ctx = &state;

  test_section("cf_alloc_debug_new");

  cf_alloc_debug_new(&debug, &backing, statement);
  cf_alloc_debug_new(&debug_default, &backing, CF_NULL);

  test_check(debug.internal_allocator.ctx == &state, "internal allocator stores backing ctx");
  test_check(debug.internal_allocator.alloc == backing.alloc, "internal allocator stores alloc fn");
  test_check(debug.internal_allocator.realloc == backing.realloc, "internal allocator stores realloc fn");
  test_check(debug.internal_allocator.free == backing.free, "internal allocator stores free fn");
  test_check(debug.allocator.ctx == &debug, "public allocator ctx points to debug object");
  test_check(debug.allocator.alloc != CF_NULL, "public alloc callback is set");
  test_check(debug.allocator.realloc != CF_NULL, "public realloc callback is set");
  test_check(debug.allocator.free != CF_NULL, "public free callback is set");
  test_check(debug.statement == statement, "custom statement is stored");
  test_check(strcmp(debug_default.statement, "NO STATEMENT DECLARED") == 0, "default statement fallback is stored");
  test_check(debug.ptr_live == 0, "live count starts at zero");
  test_check(debug.ptr_free == 0, "free count starts at zero");
  test_check(debug.ptr_max_live == 0, "max live count starts at zero");
  test_check(debug.ptr_max_free == 0, "max free count starts at zero");
  test_check(debug.ptr_all_live == 0, "all live count starts at zero");
  test_check(debug.ptr_all_free == 0, "all free count starts at zero");
  test_check(debug.ptr_invalid_alloc == 0, "invalid tracking alloc count starts at zero");
  test_check(debug.ptr_internal_invalid_alloc == 0, "internal alloc failure count starts at zero");
  test_check(debug.ptr_internal_invalid_realloc == 0, "internal realloc failure count starts at zero");
  test_check(debug.ptr_internal_invalid_free == 0, "internal free failure count starts at zero");
  test_check(debug.latest_valid_ptr == CF_NULL, "latest valid pointer starts null");
  test_check(debug.head == CF_NULL, "head starts null");

  cf_alloc_debug_log(&debug, __LINE__);

  cf_alloc_debug_new(CF_NULL, &backing, statement);
  cf_alloc_debug_new(&untouched, CF_NULL, statement);
  test_check(untouched.ptr_live == 77, "null backing argument leaves target unchanged");
  test_check(untouched.ptr_all_live == 88, "null backing preserves other fields");
  test_check(strcmp(untouched.statement, "unchanged") == 0, "null backing preserves statement");
}

static void test_cf_alloc_debug_callbacks(void)
{
  test_alloc_state state = {0};
  cf_alloc backing =
  {
    .ctx = &state,
    .alloc = test_alloc,
    .realloc = test_realloc,
    .free = test_free,
  };
  cf_alloc_debug debug = {0};
  cf_u8 external_byte = 0;
  cf_u8 *ptr_a = CF_NULL;
  cf_u8 *ptr_b = CF_NULL;
  cf_u8 *ptr_c = CF_NULL;
  cf_u8 *ptr_c_realloc = CF_NULL;

  g_test_alloc_state = &state;
  state.expected_ctx = &state;

  test_section("cf_alloc_debug callbacks");

  cf_alloc_debug_new(&debug, &backing, "cf_alloc_debug callback section");

  test_check(debug.allocator.alloc(CF_NULL, 16) == CF_NULL, "alloc with null ctx returns null");
  test_check(debug.allocator.realloc(CF_NULL, &external_byte, 16) == CF_NULL, "realloc with null ctx returns null");
  debug.allocator.free(CF_NULL, &external_byte);
  test_check(debug.ptr_live == 0, "null ctx free does not change live count");
  test_check(debug.ptr_all_live == 0, "null ctx alloc path does not change all_live");
  test_check(debug.ptr_all_free == 0, "null ctx free path does not change all_free");
  test_check(debug.ptr_internal_invalid_alloc == 0, "null ctx alloc does not count as internal alloc failure");
  test_check(debug.ptr_internal_invalid_realloc == 0, "null ctx realloc does not count as internal realloc failure");
  test_check(debug.ptr_internal_invalid_free == 0, "null ctx free does not count as internal free failure");

  state.fail_alloc_count = 1;
  test_check(debug.allocator.alloc(debug.allocator.ctx, 32) == CF_NULL, "failed backing alloc returns null");
  test_check(state.alloc_calls == 1, "backing alloc called once for failed alloc");
  test_check(debug.ptr_internal_invalid_alloc == 1, "failed backing alloc increments invalid alloc count");
  test_check(debug.ptr_live == 0, "failed backing alloc does not change live count");
  test_check(debug.ptr_all_live == 0, "failed backing alloc does not change total live count");
  test_check(debug.head == CF_NULL, "failed backing alloc does not create list node");
  test_check(debug.latest_valid_ptr == CF_NULL, "failed backing alloc leaves latest pointer null");

  cf_alloc_debug_log(&debug, __LINE__);

  ptr_a = debug.allocator.alloc(debug.allocator.ctx, 32);
  ptr_b = debug.allocator.alloc(debug.allocator.ctx, 24);
  ptr_c = debug.allocator.alloc(debug.allocator.ctx, 8);
  test_check(ptr_a != CF_NULL, "first tracked alloc succeeds");
  test_check(ptr_b != CF_NULL, "second tracked alloc succeeds");
  test_check(ptr_c != CF_NULL, "third tracked alloc succeeds");
  test_check(state.alloc_calls == 4, "three successful allocs plus one failed alloc hit backing allocator");

  ptr_a[0] = 0xAA;
  ptr_a[31] = 0x55;
  ptr_b[0] = 0xB1;
  ptr_b[23] = 0xB2;
  ptr_c[0] = 0xC3;

  test_check(debug.ptr_live == 3, "three successful allocs set live count to three");
  test_check(debug.ptr_free == 0, "allocs do not advance free count");
  test_check(debug.ptr_max_live == 3, "max live count tracks peak allocations");
  test_check(debug.ptr_max_free == 0, "max free count stays zero before frees");
  test_check(debug.ptr_all_live == 3, "all_live counts successful allocs");
  test_check(debug.ptr_all_free == 0, "all_free stays zero before frees");
  test_check(debug.latest_valid_ptr == ptr_c, "latest valid pointer tracks newest alloc");
  test_check(debug.head != CF_NULL, "head exists after successful allocs");
  test_check(debug.head != CF_NULL && debug.head->ptr == ptr_c, "head points to newest pointer");
  test_check(debug.head != CF_NULL && debug.head->next != CF_NULL, "second node exists");
  test_check(debug.head != CF_NULL && debug.head->next != CF_NULL && debug.head->next->ptr == ptr_b, "second node tracks middle pointer");
  test_check(
    debug.head != CF_NULL &&
    debug.head->next != CF_NULL &&
    debug.head->next->next != CF_NULL &&
    debug.head->next->next->ptr == ptr_a,
    "third node tracks oldest pointer"
  );

  cf_alloc_debug_log(&debug, __LINE__);

  test_check(debug.allocator.realloc(debug.allocator.ctx, &external_byte, 16) == CF_NULL, "realloc of untracked pointer returns null");
  test_check(state.realloc_calls == 0, "realloc of untracked pointer does not call backing realloc");
  test_check(debug.ptr_internal_invalid_realloc == 0, "realloc of untracked pointer does not count as internal realloc failure");

  state.fail_realloc_count = 1;
  test_check(debug.allocator.realloc(debug.allocator.ctx, ptr_c, 64) == CF_NULL, "failed backing realloc returns null");
  test_check(state.realloc_calls == 1, "tracked realloc calls backing realloc");
  test_check(debug.ptr_internal_invalid_realloc == 1, "failed backing realloc increments invalid realloc count");
  test_check(debug.latest_valid_ptr == ptr_c, "failed realloc keeps latest valid pointer unchanged");
  test_check(debug.head != CF_NULL && debug.head->ptr == ptr_c, "failed realloc keeps tracked pointer unchanged");

  cf_alloc_debug_log(&debug, __LINE__);

  ptr_c_realloc = debug.allocator.realloc(debug.allocator.ctx, ptr_c, 64);
  test_check(ptr_c_realloc != CF_NULL, "successful realloc returns non-null");
  test_check(state.realloc_calls == 2, "second tracked realloc hits backing realloc");
  test_check(ptr_c_realloc[0] == 0xC3, "realloc preserves prior contents");
  test_check(debug.ptr_live == 3, "realloc does not change live count");
  test_check(debug.ptr_all_live == 3, "realloc does not change all_live count");
  test_check(debug.ptr_all_free == 0, "realloc does not change all_free count");
  test_check(debug.latest_valid_ptr == ptr_c_realloc, "successful realloc updates latest valid pointer");
  test_check(debug.head != CF_NULL && debug.head->ptr == ptr_c_realloc, "successful realloc updates tracked head pointer");

  cf_alloc_debug_log(&debug, __LINE__);

  debug.allocator.free(debug.allocator.ctx, &external_byte);
  test_check(debug.ptr_internal_invalid_free == 1, "free of untracked pointer increments invalid free count");
  test_check(state.free_calls == 0, "free of untracked pointer does not call backing free");

  debug.allocator.free(debug.allocator.ctx, ptr_b);
  test_check(state.free_calls == 1, "free of middle node hits backing free");
  test_check(debug.ptr_live == 2, "free decrements live count");
  test_check(debug.ptr_free == 0, "first successful free keeps free ratio at zero while live side drains");
  test_check(debug.ptr_max_free == 0, "max free count stays zero while frees are absorbed by live ratio");
  test_check(debug.ptr_all_free == 1, "all_free increments after first free");
  test_check(debug.latest_valid_ptr == ptr_c_realloc, "free of middle node keeps latest pointer at head");
  test_check(debug.head != CF_NULL && debug.head->ptr == ptr_c_realloc, "head stays at most recent allocation after middle free");
  test_check(debug.head != CF_NULL && debug.head->next != CF_NULL && debug.head->next->ptr == ptr_a, "middle free relinks list to oldest node");

  cf_alloc_debug_log(&debug, __LINE__);

  debug.allocator.free(debug.allocator.ctx, ptr_c_realloc);
  test_check(state.free_calls == 2, "free of head node hits backing free");
  test_check(debug.ptr_live == 1, "second free decrements live count");
  test_check(debug.ptr_free == 0, "second free still keeps free ratio at zero");
  test_check(debug.ptr_max_free == 0, "max free count remains zero after second free");
  test_check(debug.ptr_all_free == 2, "all_free increments after second free");
  test_check(debug.latest_valid_ptr == ptr_a, "free of head node moves latest pointer to new head");
  test_check(debug.head != CF_NULL && debug.head->ptr == ptr_a, "free of head node promotes next node to head");
  test_check(debug.head != CF_NULL && debug.head->next == CF_NULL, "one node remains after freeing head");

  cf_alloc_debug_log(&debug, __LINE__);

  debug.allocator.free(debug.allocator.ctx, ptr_a);
  test_check(state.free_calls == 3, "free of final node hits backing free");
  test_check(debug.ptr_live == 0, "final free drains live ratio back to zero");
  test_check(debug.ptr_free == 0, "free ratio stays zero after balanced tracked frees");
  test_check(debug.ptr_max_free == 0, "max free count stays zero when no free-side surplus appears");
  test_check(debug.ptr_all_free == 3, "all_free equals number of successful frees");
  test_check(debug.latest_valid_ptr == CF_NULL, "latest valid pointer becomes null when list is empty");
  test_check(debug.head == CF_NULL, "head becomes null after final free");

  cf_alloc_debug_log(&debug, __LINE__);

  test_check(state.bad_ctx_alloc_calls == 0, "backing alloc receives backing ctx");
  test_check(state.bad_ctx_realloc_calls == 0, "backing realloc receives backing ctx");
  test_check(state.bad_ctx_free_calls == 0, "backing free receives backing ctx");
}

static void test_cf_alloc_debug_log_fn(void)
{
  test_alloc_state state = {0};
  cf_alloc backing =
  {
    .ctx = &state,
    .alloc = test_alloc,
    .realloc = test_realloc,
    .free = test_free,
  };
  cf_alloc_debug debug = {0};

  g_test_alloc_state = &state;
  state.expected_ctx = &state;

  test_section("cf_alloc_debug_log");

  cf_alloc_debug_new(&debug, &backing, CF_NULL);
  cf_alloc_debug_log(CF_NULL, __LINE__);
  cf_alloc_debug_log(&debug, __LINE__);
  test_pass("debug log handles null and valid debug objects");
}

static void test_cf_alloc_debug_ratio_switch(void)
{
  test_alloc_state state = {0};
  cf_alloc backing =
  {
    .ctx = &state,
    .alloc = test_alloc,
    .realloc = test_realloc,
    .free = test_free,
  };
  cf_alloc_debug debug = {0};
  cf_u8 *ptr_a = CF_NULL;
  cf_u8 *ptr_b = CF_NULL;

  g_test_alloc_state = &state;
  state.expected_ctx = &state;

  test_section("cf_alloc_debug ratio switch");

  cf_alloc_debug_new(&debug, &backing, "cf_alloc_debug ratio switch");

  /* Prime the debug counters to simulate free-side surplus so alloc can consume it. */
  debug.ptr_free = 2;
  debug.ptr_max_free = 2;
  cf_alloc_debug_log(&debug, __LINE__);

  ptr_a = debug.allocator.alloc(debug.allocator.ctx, 16);
  test_check(ptr_a != CF_NULL, "alloc succeeds while free ratio has surplus");
  test_check(debug.ptr_live == 0, "alloc consumes free surplus before growing live count");
  test_check(debug.ptr_free == 1, "alloc consumes one step from free-side surplus");
  test_check(debug.ptr_all_live == 1, "all_live still counts the allocation event");
  test_check(debug.ptr_max_live == 0, "max live stays unchanged while alloc only consumes free surplus");
  cf_alloc_debug_log(&debug, __LINE__);

  ptr_b = debug.allocator.alloc(debug.allocator.ctx, 16);
  test_check(ptr_b != CF_NULL, "second alloc succeeds while free surplus remains");
  test_check(debug.ptr_live == 0, "second alloc also consumes free surplus before live grows");
  test_check(debug.ptr_free == 0, "second alloc clears the free-side surplus");
  test_check(debug.ptr_all_live == 2, "all_live counts both allocations that consumed surplus");
  cf_alloc_debug_log(&debug, __LINE__);

  debug.allocator.free(debug.allocator.ctx, ptr_b);
  test_check(debug.ptr_live == 0, "free sees no live-side surplus to drain");
  test_check(debug.ptr_free == 1, "free grows the free side once live side is zero");
  test_check(debug.ptr_all_free == 1, "all_free still counts the free operation");
  test_check(debug.ptr_max_free == 2, "max free keeps the earlier primed surplus peak");
  cf_alloc_debug_log(&debug, __LINE__);

  debug.allocator.free(debug.allocator.ctx, ptr_a);
  test_check(debug.ptr_live == 0, "final free leaves live side at zero");
  test_check(debug.ptr_free == 2, "final free restores the free-side surplus");
  test_check(debug.ptr_all_free == 2, "all_free counts both cleanup frees");
  test_check(debug.ptr_max_free == 2, "max free remains at the highest free-side peak");
  test_check(debug.head == CF_NULL, "cleanup empties the tracked list");
  cf_alloc_debug_log(&debug, __LINE__);
}

static void test_cf_array_basic_flow(void)
{
  cf_array array = {0};
  cf_array_element first = {.data = (cf_u8 *)"A", .elem_size = sizeof(cf_u8), .len = 1};
  cf_array_element second = {.data = (cf_u8 *)"BC", .elem_size = sizeof(cf_u8), .len = 2};
  cf_array_element third = {.data = (cf_u8 *)"DEF", .elem_size = sizeof(cf_u8), .len = 3};
  cf_array_element popped = {0};
  cf_status status = CF_OK;

  test_section("cf_array basic flow");

  status = cf_array_init(&array, 1);
  test_check(status == CF_OK, "array init succeeds");
  test_check(array.len == 0, "array starts empty");
  test_check(array.cap == 1, "array starts with requested capacity");

  status = cf_array_push(&array, &first, &second, &third, CF_NULL);
  test_check(status == CF_OK, "array push accepts multiple elements");
  test_check(array.len == 3, "array length reflects pushed elements");
  test_check(array.cap >= 3, "array grows to fit additional elements");
  test_check(array.data[0].data == first.data, "first pushed element is stored");
  test_check(array.data[1].data == second.data, "second pushed element is stored");
  test_check(array.data[2].data == third.data, "third pushed element is stored");
  test_check(array.data[0].elem_size == sizeof(cf_u8), "stored element type width is preserved");
  test_check(array.data[2].len == third.len, "stored metadata is preserved");

  status = cf_array_pop(&array, &popped);
  test_check(status == CF_OK, "array pop succeeds on non-empty array");
  test_check(array.len == 2, "array pop decrements length");
  test_check(popped.data == third.data, "array pop returns the last element");
  test_check(popped.len == third.len, "array pop preserves popped length");

  status = cf_array_reset(&array);
  test_check(status == CF_OK, "array reset succeeds");
  test_check(array.len == 0, "array reset clears logical length");
  test_check(array.cap >= 3, "array reset preserves capacity");

  popped = (cf_array_element){.data = (cf_u8 *)"x", .elem_size = sizeof(cf_u16), .len = 9};
  status = cf_array_pop(&array, &popped);
  test_check(status == CF_OK, "array pop on empty array succeeds");
  test_check(popped.data == CF_NULL, "array pop on empty array clears output data");
  test_check(popped.elem_size == 0, "array pop on empty array clears output type width");
  test_check(popped.len == 0, "array pop on empty array clears output length");

  cf_array_destroy(&array);
}

static void test_cf_array_accessors(void)
{
  cf_array array = {0};
  cf_array_element first = {.data = (cf_u8 *)"AB", .elem_size = sizeof(cf_u8), .len = 2};
  cf_array_element second = {.data = (cf_u16[]){10, 20}, .elem_size = sizeof(cf_u16), .len = 2};
  cf_array_element third = {.data = (cf_u32[]){30}, .elem_size = sizeof(cf_u32), .len = 1};
  cf_array_element read_back = {0};
  cf_array_element replacement = {.data = (cf_u64[]){77}, .elem_size = sizeof(cf_u64), .len = 1};
  cf_status status = CF_OK;

  test_section("cf_array accessors");

  test_check(cf_array_is_valid(&array) == CF_TRUE, "zero-initialized array is structurally valid");
  test_check(cf_array_is_empty(&array) == CF_TRUE, "zero-initialized array reports empty");

  status = cf_array_init(&array, 0);
  test_check(status == CF_OK, "array init with zero capacity succeeds");
  test_check(cf_array_is_valid(&array) == CF_TRUE, "initialized array stays structurally valid");
  test_check(cf_array_is_empty(&array) == CF_TRUE, "initialized empty array reports empty");

  status = cf_array_push(&array, &first, &second, CF_NULL);
  test_check(status == CF_OK, "push prepares array for accessor tests");
  test_check(cf_array_is_empty(&array) == CF_FALSE, "non-empty array reports not empty");

  status = cf_array_peek(&array, &read_back);
  test_check(status == CF_OK, "peek succeeds on non-empty array");
  test_check(read_back.data == second.data, "peek reads the last element");
  test_check(read_back.len == second.len, "peek preserves element length");
  test_check(array.len == 2, "peek does not change array length");

  status = cf_array_get(&array, 0, &read_back);
  test_check(status == CF_OK, "get succeeds for a valid index");
  test_check(read_back.data == first.data, "get reads the requested element");
  test_check(read_back.elem_size == first.elem_size, "get preserves requested element size");
  test_check(read_back.len == first.len, "get preserves requested element length");

  status = cf_array_set(&array, 1, &replacement);
  test_check(status == CF_OK, "set succeeds for a valid index");
  test_check(array.len == 2, "set does not change array length");

  status = cf_array_get(&array, 1, &read_back);
  test_check(status == CF_OK, "get succeeds after set");
  test_check(read_back.data == replacement.data, "set replaces the stored data pointer");
  test_check(read_back.elem_size == replacement.elem_size, "set replaces the stored element size");
  test_check(read_back.len == replacement.len, "set replaces the stored element length");

  status = cf_array_get(&array, 2, &read_back);
  test_check(status == CF_ERR_BOUNDS, "get rejects an out-of-bounds index");
  status = cf_array_set(&array, 2, &third);
  test_check(status == CF_ERR_BOUNDS, "set rejects an out-of-bounds index");
  status = cf_array_get(&array, 0, CF_NULL);
  test_check(status == CF_ERR_NULL, "get rejects a null output pointer");
  status = cf_array_set(&array, 0, CF_NULL);
  test_check(status == CF_ERR_NULL, "set rejects a null input pointer");
  status = cf_array_peek(&array, CF_NULL);
  test_check(status == CF_ERR_NULL, "peek rejects a null output pointer");

  cf_array_destroy(&array);
  test_check(cf_array_is_valid(&array) == CF_TRUE, "destroy resets array to a valid empty state");
  test_check(cf_array_is_empty(&array) == CF_TRUE, "destroyed array reports empty");
}

static void test_cf_array_state_and_diagnostics(void)
{
  cf_array array = {0};
  cf_array invalid = {.data = CF_NULL, .len = 1, .cap = 0};
  cf_array_element element = {.data = (cf_u8 *)"Z", .elem_size = sizeof(cf_u8), .len = 1};
  cf_array_element read_back = {0};
  cf_status status = CF_OK;

  test_section("cf_array state and diagnostics");

  test_check(cf_array_init(CF_NULL, 0) == CF_ERR_NULL, "array init rejects null array");
  test_check(cf_array_reserve(CF_NULL, 1) == CF_ERR_NULL, "array reserve rejects null array");
  test_check(cf_array_reset(CF_NULL) == CF_ERR_NULL, "array reset rejects null array");
  test_check(cf_array_peek(CF_NULL, &read_back) == CF_ERR_NULL, "array peek rejects null array");
  test_check(cf_array_pop(CF_NULL, &read_back) == CF_ERR_NULL, "array pop rejects null array");
  test_check(cf_array_get(CF_NULL, 0, &read_back) == CF_ERR_NULL, "array get rejects null array");
  test_check(cf_array_set(CF_NULL, 0, &element) == CF_ERR_NULL, "array set rejects null array");
  test_check(cf_array_is_valid(CF_NULL) == CF_FALSE, "array validity rejects null array");
  test_check(cf_array_is_empty(CF_NULL) == CF_FALSE, "null array does not report empty");

  status = cf_array_init(&array, 0);
  test_check(status == CF_OK, "array init prepares reserve tests");
  status = cf_array_reserve(&array, 4);
  test_check(status == CF_OK, "array reserve allocates backing storage");
  test_check(array.cap >= 4, "array reserve stores requested capacity");
  test_check(array.len == 0, "array reserve keeps length unchanged");

  status = cf_array_push(&array, CF_NULL);
  test_check(status == CF_OK, "array push with null first element is a no-op");
  test_check(array.len == 0, "null push keeps array length unchanged");

  status = cf_array_push(&array, &element, CF_NULL);
  test_check(status == CF_OK, "array push prepares diagnostics");
  cf_array_info(&array);
  cf_array_info(CF_NULL);
  test_pass("array info handles valid and null arrays");

  test_check(cf_array_is_valid(&invalid) == CF_FALSE, "array validity detects impossible len/cap state");
  test_check(cf_array_is_empty(&invalid) == CF_FALSE, "invalid array does not report empty");
  test_check(cf_array_reserve(&invalid, 2) == CF_ERR_STATE, "array reserve rejects invalid state");

  cf_array_destroy(&array);
}

static void test_cf_ascii_helpers(void)
{
  test_section("cf_ascii helpers");

  test_check(cf_ascii_is_alpha('A') == CF_TRUE, "ascii alpha accepts uppercase lower bound");
  test_check(cf_ascii_is_alpha('Z') == CF_TRUE, "ascii alpha accepts uppercase upper bound");
  test_check(cf_ascii_is_alpha('a') == CF_TRUE, "ascii alpha accepts lowercase lower bound");
  test_check(cf_ascii_is_alpha('z') == CF_TRUE, "ascii alpha accepts lowercase upper bound");
  test_check(cf_ascii_is_alpha('@') == CF_FALSE, "ascii alpha rejects char before uppercase letters");
  test_check(cf_ascii_is_alpha('[') == CF_FALSE, "ascii alpha rejects char after uppercase letters");
  test_check(cf_ascii_is_alpha('`') == CF_FALSE, "ascii alpha rejects char before lowercase letters");
  test_check(cf_ascii_is_alpha('{') == CF_FALSE, "ascii alpha rejects char after lowercase letters");

  test_check(cf_ascii_is_digit('0') == CF_TRUE, "ascii digit accepts lower bound");
  test_check(cf_ascii_is_digit('9') == CF_TRUE, "ascii digit accepts upper bound");
  test_check(cf_ascii_is_digit('/') == CF_FALSE, "ascii digit rejects char before digits");
  test_check(cf_ascii_is_digit(':') == CF_FALSE, "ascii digit rejects char after digits");

  test_check(cf_ascii_is_alnum('A') == CF_TRUE, "ascii alnum accepts letters");
  test_check(cf_ascii_is_alnum('7') == CF_TRUE, "ascii alnum accepts digits");
  test_check(cf_ascii_is_alnum('_') == CF_FALSE, "ascii alnum rejects punctuation");

  test_check(cf_ascii_is_space(' ') == CF_TRUE, "ascii space accepts space");
  test_check(cf_ascii_is_space('\t') == CF_TRUE, "ascii space accepts horizontal tab");
  test_check(cf_ascii_is_space('\n') == CF_TRUE, "ascii space accepts newline");
  test_check(cf_ascii_is_space('\v') == CF_TRUE, "ascii space accepts vertical tab");
  test_check(cf_ascii_is_space('\f') == CF_TRUE, "ascii space accepts form feed");
  test_check(cf_ascii_is_space('\r') == CF_TRUE, "ascii space accepts carriage return");
  test_check(cf_ascii_is_space('\0') == CF_FALSE, "ascii space rejects nul");
  test_check(cf_ascii_is_space('A') == CF_FALSE, "ascii space rejects non-whitespace");

  test_check(cf_ascii_is_upper('A') == CF_TRUE, "ascii upper accepts lower bound");
  test_check(cf_ascii_is_upper('Z') == CF_TRUE, "ascii upper accepts upper bound");
  test_check(cf_ascii_is_upper('a') == CF_FALSE, "ascii upper rejects lowercase");
  test_check(cf_ascii_is_lower('a') == CF_TRUE, "ascii lower accepts lower bound");
  test_check(cf_ascii_is_lower('z') == CF_TRUE, "ascii lower accepts upper bound");
  test_check(cf_ascii_is_lower('A') == CF_FALSE, "ascii lower rejects uppercase");

  test_check(cf_ascii_to_upper('a') == 'A', "ascii to upper converts lowercase");
  test_check(cf_ascii_to_upper('Z') == 'Z', "ascii to upper keeps uppercase");
  test_check(cf_ascii_to_upper('7') == '7', "ascii to upper keeps non-letters");
  test_check(cf_ascii_to_lower('A') == 'a', "ascii to lower converts uppercase");
  test_check(cf_ascii_to_lower('z') == 'z', "ascii to lower keeps lowercase");
  test_check(cf_ascii_to_lower('7') == '7', "ascii to lower keeps non-letters");

  test_check(cf_ascii_hex_value('0') == 0, "ascii hex converts zero");
  test_check(cf_ascii_hex_value('9') == 9, "ascii hex converts nine");
  test_check(cf_ascii_hex_value('a') == 10, "ascii hex converts lowercase a");
  test_check(cf_ascii_hex_value('f') == 15, "ascii hex converts lowercase f");
  test_check(cf_ascii_hex_value('A') == 10, "ascii hex converts uppercase A");
  test_check(cf_ascii_hex_value('F') == 15, "ascii hex converts uppercase F");
  test_check(cf_ascii_hex_value('g') == -1, "ascii hex rejects lowercase past f");
  test_check(cf_ascii_hex_value('G') == -1, "ascii hex rejects uppercase past F");
  test_check(cf_ascii_hex_value('/') == -1, "ascii hex rejects char before digits");
  test_check(cf_ascii_hex_value(':') == -1, "ascii hex rejects char after digits");
}

static void test_cf_string_build_and_lifecycle(void)
{
  cf_string str = {0};
  cf_string suffix = {0};
  char *owned = CF_NULL;
  cf_status status = CF_OK;

  test_section("cf_string build and lifecycle");

  test_check(cf_string_init(CF_NULL, 0) == CF_ERR_NULL, "string init rejects null string");
  status = cf_string_init(&str, 0);
  test_check(status == CF_OK, "string init with zero capacity succeeds");
  test_check(cf_string_is_valid(&str) == CF_TRUE, "initialized string is valid");
  test_check(cf_string_is_empty(&str) == CF_TRUE, "initialized string reports empty");

  status = cf_string_reserve(&str, 8);
  test_check(status == CF_OK, "string reserve succeeds");
  test_check(str.cap >= 9, "string reserve keeps room for terminator");
  test_check(str.data != CF_NULL && str.data[str.len] == '\0', "string reserve writes terminator");

  status = cf_string_append_char(&str, 'c');
  test_check(status == CF_OK, "string append char succeeds");
  status = cf_string_append_cstr(&str, "ypher");
  test_check(status == CF_OK, "string append cstr succeeds");
  test_check_string_value(&str, "cypher", "append char and cstr build expected text");

  status = cf_string_init(&suffix, 0);
  test_check(status == CF_OK, "source string init succeeds");
  status = cf_string_from_cstr(&suffix, "Framework");
  test_check(status == CF_OK, "string from cstr writes source text");
  status = cf_string_append_str(&str, &suffix);
  test_check(status == CF_OK, "string append str succeeds");
  test_check_string_value(&str, "cypherFramework", "append str copies source contents");

  status = cf_string_as_cstr(&owned, &str);
  test_check(status == CF_OK, "string as cstr succeeds");
  test_check(owned != CF_NULL && strcmp(owned, "cypherFramework") == 0, "as cstr returns an owned null-terminated copy");
  str.allocator.free(str.allocator.ctx, owned);
  owned = CF_NULL;

  status = cf_string_trunc(&str, 6);
  test_check(status == CF_OK, "string trunc succeeds");
  test_check_string_value(&str, "cypher", "string trunc keeps requested prefix");
  test_check(cf_string_trunc(&str, 99) == CF_ERR_BOUNDS, "string trunc rejects too-large length");

  status = cf_string_reset(&str);
  test_check(status == CF_OK, "string reset succeeds");
  test_check(cf_string_is_empty(&str) == CF_TRUE, "string reset reports empty");
  test_check(str.data != CF_NULL && str.data[0] == '\0', "string reset keeps terminator");

  cf_string_info(&str);
  cf_string_info(CF_NULL);
  test_pass("string info handles valid and null strings");

  cf_string_destroy(&suffix);
  cf_string_destroy(&str);
  test_check(cf_string_is_valid(&str) == CF_TRUE, "destroy resets string to valid empty state");
  test_check(cf_string_is_empty(&str) == CF_TRUE, "destroyed string reports empty");
}

static void test_cf_string_queries_and_slices(void)
{
  cf_string haystack = {0};
  cf_string same = {0};
  cf_string needle = {0};
  char c = '\0';
  char *tail = CF_NULL;
  char *slice = CF_NULL;
  cf_status status = CF_OK;

  test_section("cf_string queries and slices");

  cf_string_init(&haystack, 0);
  cf_string_init(&same, 0);
  cf_string_init(&needle, 0);
  cf_string_from_cstr(&haystack, "alpha beta gamma");
  cf_string_from_cstr(&same, "alpha beta gamma");
  cf_string_from_cstr(&needle, "beta");

  test_check(cf_string_eq(&haystack, &same) == CF_TRUE, "string eq accepts identical strings");
  test_check(cf_string_eq(&haystack, &needle) == CF_FALSE, "string eq rejects different strings");
  test_check(cf_string_eq(&haystack, CF_NULL) == CF_FALSE, "string eq rejects null argument");
  test_check(cf_string_contains_char(&haystack, 'g') == CF_TRUE, "contains char finds present character");
  test_check(cf_string_contains_char(&haystack, 'z') == CF_FALSE, "contains char rejects missing character");
  test_check(cf_string_contains_cstr(&haystack, "beta") == CF_TRUE, "contains cstr finds present substring");
  test_check(cf_string_contains_cstr(&haystack, "delta") == CF_FALSE, "contains cstr rejects missing substring");
  test_check(cf_string_contains_str(&haystack, &needle) == CF_TRUE, "contains str finds present string");
  test_check(cf_string_contains_str(&needle, &haystack) == CF_FALSE, "contains str rejects longer missing string");

  status = cf_string_char_at(&haystack, 6, &c);
  test_check(status == CF_OK && c == 'b', "char at returns requested character");
  test_check(cf_string_char_at(&haystack, haystack.len, &c) == CF_ERR_BOUNDS, "char at rejects end index");
  test_check(cf_string_char_at(&haystack, 0, CF_NULL) == CF_ERR_NULL, "char at rejects null output");

  status = cf_string_str_at(&haystack, 6, &tail);
  test_check(status == CF_OK, "str at copies suffix");
  test_check(tail != CF_NULL && strcmp(tail, "beta gamma") == 0, "str at suffix content is correct");
  haystack.allocator.free(haystack.allocator.ctx, tail);
  tail = CF_NULL;
  test_check(cf_string_str_at(&haystack, haystack.len, &tail) == CF_ERR_BOUNDS, "str at rejects end index");
  test_check(cf_string_str_at(&haystack, 0, CF_NULL) == CF_ERR_NULL, "str at rejects null output");

  status = cf_string_slice(&slice, &haystack, 6, 9);
  test_check(status == CF_OK, "string slice copies inclusive range");
  test_check(slice != CF_NULL && strcmp(slice, "beta") == 0, "string slice content is correct");
  haystack.allocator.free(haystack.allocator.ctx, slice);
  slice = CF_NULL;
  test_check(cf_string_slice(&slice, &haystack, 4, 3) == CF_ERR_INVALID, "string slice rejects reversed range");
  test_check(cf_string_slice(&slice, &haystack, 0, haystack.len) == CF_ERR_BOUNDS, "string slice rejects end past content");
  test_check(cf_string_slice(CF_NULL, &haystack, 0, 1) == CF_ERR_NULL, "string slice rejects null output");

  cf_string_destroy(&needle);
  cf_string_destroy(&same);
  cf_string_destroy(&haystack);
}

static void test_cf_string_mutation_helpers(void)
{
  cf_string str = {0};
  cf_string csv = {0};
  cf_array parts = {0};
  cf_array_element part = {0};
  cf_status status = CF_OK;

  test_section("cf_string mutation helpers");

  cf_string_init(&str, 0);
  cf_string_from_cstr(&str, " \talpha beta\n");

  status = cf_string_trim_left(&str);
  test_check(status == CF_OK, "string trim left succeeds");
  test_check_string_value(&str, "alpha beta\n", "trim left removes leading whitespace and updates length");

  status = cf_string_trim_right(&str);
  test_check(status == CF_OK, "string trim right succeeds");
  test_check_string_value(&str, "alpha beta", "trim right removes trailing whitespace");

  cf_string_from_cstr(&str, "\n alpha beta \t");
  status = cf_string_trim(&str);
  test_check(status == CF_OK, "string trim succeeds");
  test_check_string_value(&str, "alpha beta", "trim removes both leading and trailing whitespace");

  status = cf_string_strip(&str);
  test_check(status == CF_OK, "string strip succeeds");
  test_check_string_value(&str, "alphabeta", "strip removes all framework whitespace");

  status = cf_string_replace(&str, 'a', 'A');
  test_check(status == CF_OK, "string replace succeeds");
  test_check_string_value(&str, "AlphAbetA", "replace swaps every target character");

  cf_string_init(&csv, 0);
  cf_array_init(&parts, 0);
  cf_string_from_cstr(&csv, "red,,green,blue,");
  status = cf_string_split(&parts, &csv, ',');
  test_check(status == CF_OK, "string split succeeds");
  test_check(parts.len == 3, "string split skips empty fields");

  status = cf_array_get(&parts, 0, &part);
  test_check(status == CF_OK && part.data != CF_NULL && strcmp((char *)part.data, "red") == 0, "split part 0 is correct");
  csv.allocator.free(csv.allocator.ctx, part.data);
  status = cf_array_get(&parts, 1, &part);
  test_check(status == CF_OK && part.data != CF_NULL && strcmp((char *)part.data, "green") == 0, "split part 1 is correct");
  csv.allocator.free(csv.allocator.ctx, part.data);
  status = cf_array_get(&parts, 2, &part);
  test_check(status == CF_OK && part.data != CF_NULL && strcmp((char *)part.data, "blue") == 0, "split part 2 is correct");
  csv.allocator.free(csv.allocator.ctx, part.data);

  test_check(cf_string_split(CF_NULL, &csv, ',') == CF_ERR_NULL, "string split rejects null destination");

  cf_array_destroy(&parts);
  cf_string_destroy(&csv);
  cf_string_destroy(&str);
}

static void test_cf_string_invalid_state(void)
{
  cf_string invalid = {.data = CF_NULL, .len = 1, .cap = 0};
  cf_string str = {0};

  test_section("cf_string invalid state");

  test_check(cf_string_is_valid(CF_NULL) == CF_FALSE, "string validity rejects null string");
  test_check(cf_string_is_empty(CF_NULL) == CF_FALSE, "null string does not report empty");
  test_check(cf_string_is_valid(&invalid) == CF_FALSE, "string validity detects impossible len/cap state");
  test_check(cf_string_is_empty(&invalid) == CF_FALSE, "invalid string does not report empty");
  test_check(cf_string_reserve(&invalid, 2) == CF_ERR_STATE, "string reserve rejects invalid state");
  test_check(cf_string_reset(&invalid) == CF_ERR_STATE, "string reset rejects invalid state");
  test_check(cf_string_trunc(&invalid, 0) == CF_ERR_STATE, "string trunc rejects invalid state");

  cf_string_init(&str, 0);
  test_check(cf_string_from_cstr(CF_NULL, "x") == CF_ERR_NULL, "string from cstr rejects null destination");
  test_check(cf_string_from_cstr(&str, CF_NULL) == CF_ERR_NULL, "string from cstr rejects null source");
  test_check(cf_string_as_cstr(CF_NULL, &str) == CF_ERR_NULL, "string as cstr rejects null output");
  cf_string_destroy(&str);
}

static void test_cf_hex_encode_decode(void)
{
  cf_u8 raw[] = {0x00, 0x2A, 0xFF, 0x10};
  cf_string encoded = {0};
  cf_string hex = {0};
  cf_buffer decoded = {0};
  cf_bytes src = {.data = raw, .elem_size = sizeof(cf_u8), .len = sizeof(raw)};
  cf_status status = CF_OK;

  test_section("cf_hex encode/decode");

  status = cf_string_init(&encoded, 0);
  test_check(status == CF_OK, "hex encode destination init succeeds");
  status = cf_hex_encode(&encoded, src);
  test_check(status == CF_OK, "hex encode succeeds");
  test_check_string_value(&encoded, "002AFF10", "hex encode writes contiguous uppercase text");

  status = cf_string_from_cstr(&encoded, "prefix:");
  test_check(status == CF_OK, "hex encode append target reset succeeds");
  status = cf_hex_encode(&encoded, (cf_bytes){.data = raw + 1, .elem_size = 1, .len = 2});
  test_check(status == CF_OK, "hex encode appends to existing string");
  test_check_string_value(&encoded, "prefix:2AFF", "hex encode preserves existing destination content");

  test_check(cf_hex_encode(CF_NULL, src) == CF_ERR_NULL, "hex encode rejects null destination");
  test_check(cf_hex_encode(&encoded, (cf_bytes){.data = CF_NULL, .elem_size = 1, .len = 1}) == CF_ERR_NULL, "hex encode rejects non-empty null source data");
  test_check(cf_hex_encode(&encoded, (cf_bytes){.data = raw, .elem_size = 2, .len = 1}) == CF_ERR_INVALID, "hex encode rejects non-byte source view");
  test_check(cf_hex_encode(&encoded, (cf_bytes){.data = CF_NULL, .elem_size = 1, .len = 0}) == CF_OK, "hex encode accepts empty null source view");

  status = cf_string_init(&hex, 0);
  test_check(status == CF_OK, "hex decode source init succeeds");
  status = cf_buffer_init(&decoded, 0);
  test_check(status == CF_OK, "hex decode destination init succeeds");
  status = cf_string_from_cstr(&hex, "002aff10");
  test_check(status == CF_OK, "hex decode source text write succeeds");
  status = cf_hex_decode(&decoded, &hex);
  test_check(status == CF_OK, "hex decode accepts lowercase input");
  test_check(decoded.len == 4, "hex decode writes expected byte count");
  test_check(decoded.data[0] == 0x00, "hex decode byte 0 is correct");
  test_check(decoded.data[1] == 0x2A, "hex decode byte 1 is correct");
  test_check(decoded.data[2] == 0xFF, "hex decode byte 2 is correct");
  test_check(decoded.data[3] == 0x10, "hex decode byte 3 is correct");

  status = cf_string_from_cstr(&hex, "Aa");
  test_check(status == CF_OK, "hex decode mixed-case source write succeeds");
  status = cf_hex_decode(&decoded, &hex);
  test_check(status == CF_OK, "hex decode appends mixed-case input");
  test_check(decoded.len == 5, "hex decode appends to existing buffer");
  test_check(decoded.data[4] == 0xAA, "hex decode mixed-case byte is correct");

  test_check(cf_hex_decode(CF_NULL, &hex) == CF_ERR_NULL, "hex decode rejects null destination");
  status = cf_string_from_cstr(&hex, "ABC");
  test_check(status == CF_OK, "hex decode odd-length source write succeeds");
  test_check(cf_hex_decode(&decoded, &hex) == CF_ERR_INVALID, "hex decode rejects odd-length text");
  status = cf_string_from_cstr(&hex, "0G");
  test_check(status == CF_OK, "hex decode invalid source write succeeds");
  test_check(cf_hex_decode(&decoded, &hex) == CF_ERR_INVALID, "hex decode rejects invalid hex character");
  test_check(cf_hex_decode(&decoded, CF_NULL) == CF_ERR_STATE, "hex decode rejects null source string state");

  cf_buffer_destroy(&decoded);
  cf_string_destroy(&hex);
  cf_string_destroy(&encoded);
}

static void test_cf_base64_encode_decode(void)
{
  cf_u8 raw[] = {'M', 'a', 'n'};
  cf_string encoded = {0};
  cf_string base64 = {0};
  cf_buffer decoded = {0};
  cf_bytes src = {.data = raw, .elem_size = sizeof(cf_u8), .len = sizeof(raw)};
  cf_status status = CF_OK;

  test_section("cf_base64 encode/decode");

  status = cf_string_init(&encoded, 0);
  test_check(status == CF_OK, "base64 encode destination init succeeds");
  status = cf_base64_encode(&encoded, src);
  test_check(status == CF_OK, "base64 encode succeeds for three-byte input");
  test_check_string_value(&encoded, "TWFu", "base64 encode writes unpadded text");

  status = cf_string_from_cstr(&encoded, "");
  test_check(status == CF_OK, "base64 encode destination reset succeeds");
  status = cf_base64_encode(&encoded, (cf_bytes){.data = (cf_u8 *)"Ma", .elem_size = 1, .len = 2});
  test_check(status == CF_OK, "base64 encode succeeds for two-byte input");
  test_check_string_value(&encoded, "TWE=", "base64 encode writes one padding character");

  status = cf_string_from_cstr(&encoded, "prefix:");
  test_check(status == CF_OK, "base64 encode append target reset succeeds");
  status = cf_base64_encode(&encoded, (cf_bytes){.data = (cf_u8 *)"M", .elem_size = 1, .len = 1});
  test_check(status == CF_OK, "base64 encode succeeds for one-byte input");
  test_check_string_value(&encoded, "prefix:TQ==", "base64 encode appends padded text");

  test_check(cf_base64_encode(CF_NULL, src) == CF_ERR_NULL, "base64 encode rejects null destination");
  test_check(cf_base64_encode(&encoded, (cf_bytes){.data = CF_NULL, .elem_size = 1, .len = 1}) == CF_ERR_NULL, "base64 encode rejects non-empty null source data");
  test_check(cf_base64_encode(&encoded, (cf_bytes){.data = raw, .elem_size = 2, .len = 1}) == CF_ERR_INVALID, "base64 encode rejects non-byte source view");
  test_check(cf_base64_encode(&encoded, (cf_bytes){.data = CF_NULL, .elem_size = 1, .len = 0}) == CF_OK, "base64 encode accepts empty null source view");

  status = cf_string_init(&base64, 0);
  test_check(status == CF_OK, "base64 decode source init succeeds");
  status = cf_buffer_init(&decoded, 0);
  test_check(status == CF_OK, "base64 decode destination init succeeds");
  status = cf_string_from_cstr(&base64, "TWFu");
  test_check(status == CF_OK, "base64 decode source text write succeeds");
  status = cf_base64_decode(&decoded, &base64);
  test_check(status == CF_OK, "base64 decode succeeds for unpadded input");
  test_check(decoded.len == 3, "base64 decode writes expected byte count");
  test_check(decoded.data[0] == 'M', "base64 decode byte 0 is correct");
  test_check(decoded.data[1] == 'a', "base64 decode byte 1 is correct");
  test_check(decoded.data[2] == 'n', "base64 decode byte 2 is correct");

  status = cf_string_from_cstr(&base64, "TWE=");
  test_check(status == CF_OK, "base64 decode padded source write succeeds");
  status = cf_base64_decode(&decoded, &base64);
  test_check(status == CF_OK, "base64 decode appends padded input");
  test_check(decoded.len == 5, "base64 decode appends decoded bytes");
  test_check(decoded.data[3] == 'M', "base64 decode padded byte 0 is correct");
  test_check(decoded.data[4] == 'a', "base64 decode padded byte 1 is correct");

  test_check(cf_base64_decode(CF_NULL, &base64) == CF_ERR_NULL, "base64 decode rejects null destination");
  status = cf_string_from_cstr(&base64, "ABC");
  test_check(status == CF_OK, "base64 decode invalid-length source write succeeds");
  test_check(cf_base64_decode(&decoded, &base64) == CF_ERR_INVALID, "base64 decode rejects non-multiple-of-four text");
  test_check(cf_base64_decode(&decoded, CF_NULL) == CF_ERR_STATE, "base64 decode rejects null source string state");

  cf_buffer_destroy(&decoded);
  cf_string_destroy(&base64);
  cf_string_destroy(&encoded);
}

static void test_cf_types_native_groups(void)
{
  test_section("cf_types native groups");

  test_check(cf_types_type_size(sizeof(cf_u8)) == CF_NATIVE_1_BYTE, "1-byte values map to CF_NATIVE_1_BYTE");
  test_check(cf_types_type_size(sizeof(cf_u16)) == CF_NATIVE_2_BYTE, "2-byte values map to CF_NATIVE_2_BYTE");
  test_check(cf_types_type_size(sizeof(cf_u32)) == CF_NATIVE_4_BYTE, "4-byte values map to CF_NATIVE_4_BYTE");
  test_check(cf_types_type_size(sizeof(cf_u64)) == CF_NATIVE_8_BYTE, "8-byte values map to CF_NATIVE_8_BYTE");
  test_check(cf_types_type_size(sizeof(cf_u128)) == CF_NATIVE_16_BYTE, "16-byte values map to CF_NATIVE_16_BYTE");
  test_check(cf_types_type_size(3) == CF_NATIVE_UNKNOWN, "non-native size maps to unknown");
  test_check(strcmp(cf_types_as_char(sizeof(cf_u8)), "1 byte: char/bool-sized primitive value or struct with 1-byte total size.") == 0, "1-byte description is stable");
  test_check(strcmp(cf_types_as_char(sizeof(cf_u64)), "8 bytes: long/pointer/double-sized primitive value or struct with 8-byte total size.") == 0, "8-byte description is stable");
  test_check(strcmp(cf_types_as_char(24), "Not a primitive type size or a struct with a native primitive-sized total width.") == 0, "non-native description is stable");
}

static void test_cf_aes_known_vector(
  const char *label,
  const cf_u8 *key,
  cf_aes_key_size key_size,
  const cf_u8 plaintext[CF_AES_BLOCK_SIZE],
  const cf_u8 expected_ciphertext[CF_AES_BLOCK_SIZE])
{
  cf_aes aes = {0};
  cf_u8 ciphertext[CF_AES_BLOCK_SIZE] = {0};
  cf_u8 decrypted[CF_AES_BLOCK_SIZE] = {0};
  cf_status status = cf_aes_init(&aes, key, key_size);
  char message[128];

  snprintf(message, sizeof(message), "%s init succeeds", label);
  test_check(status == CF_OK, message);
  if(status != CF_OK) return;

  cf_aes_encrypt_block(&aes, ciphertext, plaintext);
  snprintf(message, sizeof(message), "%s encrypt matches known ciphertext", label);
  test_check(memcmp(ciphertext, expected_ciphertext, CF_AES_BLOCK_SIZE) == 0, message);

  cf_aes_decrypt_block(&aes, decrypted, expected_ciphertext);
  snprintf(message, sizeof(message), "%s decrypt restores plaintext", label);
  test_check(memcmp(decrypted, plaintext, CF_AES_BLOCK_SIZE) == 0, message);
}

static void test_cf_aes_block_vectors(void)
{
  static const cf_u8 plaintext[CF_AES_BLOCK_SIZE] =
  {
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF
  };
  static const cf_u8 key_128[CF_AES_MAX_ROUND_KEYS * 4] =
  {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
  };
  static const cf_u8 key_192[CF_AES_MAX_ROUND_KEYS * 4] =
  {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17
  };
  static const cf_u8 key_256[CF_AES_MAX_ROUND_KEYS * 4] =
  {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F
  };
  static const cf_u8 ciphertext_128[CF_AES_BLOCK_SIZE] =
  {
    0x69, 0xC4, 0xE0, 0xD8, 0x6A, 0x7B, 0x04, 0x30,
    0xD8, 0xCD, 0xB7, 0x80, 0x70, 0xB4, 0xC5, 0x5A
  };
  static const cf_u8 ciphertext_192[CF_AES_BLOCK_SIZE] =
  {
    0xDD, 0xA9, 0x7C, 0xA4, 0x86, 0x4C, 0xDF, 0xE0,
    0x6E, 0xAF, 0x70, 0xA0, 0xEC, 0x0D, 0x71, 0x91
  };
  static const cf_u8 ciphertext_256[CF_AES_BLOCK_SIZE] =
  {
    0x8E, 0xA2, 0xB7, 0xCA, 0x51, 0x67, 0x45, 0xBF,
    0xEA, 0xFC, 0x49, 0x90, 0x4B, 0x49, 0x60, 0x89
  };

  test_section("cf_aes block vectors");

  test_cf_aes_known_vector("AES-128", key_128, CF_AES_KEY_128, plaintext, ciphertext_128);
  test_cf_aes_known_vector("AES-192", key_192, CF_AES_KEY_192, plaintext, ciphertext_192);
  test_cf_aes_known_vector("AES-256", key_256, CF_AES_KEY_256, plaintext, ciphertext_256);
}

int main(void)
{
  printf("cf allocator capability-model test suite\n");
  test_cf_alloc_debug_new_model();
  test_cf_alloc_debug_callbacks();
  test_cf_alloc_debug_log_fn();
  test_cf_alloc_debug_ratio_switch();
  test_cf_array_basic_flow();
  test_cf_array_accessors();
  test_cf_array_state_and_diagnostics();
  test_cf_ascii_helpers();
  test_cf_string_build_and_lifecycle();
  test_cf_string_queries_and_slices();
  test_cf_string_mutation_helpers();
  test_cf_string_invalid_state();
  test_cf_hex_encode_decode();
  test_cf_base64_encode_decode();
  test_cf_aes_block_vectors();
  test_cf_types_native_groups();
  printf
  (
    "\n================ Test Summary ================\n"
    "  Passed : %-6d    Failed : %-6d\n"
    "==============================================\n",
    g_test_passed,
    g_test_failed
  );

  return g_test_failed == 0 ? 0 : 1;
}
