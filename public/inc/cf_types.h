#if !defined(CF_TYPES_H)
#define CF_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/******************************************
 *
 * Base types and helper macros
 * for CypherFramework
 *
 ******************************************/

/* Basic constants and helper macros */
#define CF_TRUE true
#define CF_FALSE false
#define CF_NULL NULL
//if parametre isn't used
#define CF_UNUSED(x) ((void)(x))
// works only on array defined a[] no pointer defined *a
#define CF_ARRAY_COUNT(x) (sizeof(x) / sizeof((x)[0]))
// 
#define CF_STATIC_ASSERT(expr, msg) _Static_assert((expr), msg)

// typedef the unsigned types
typedef uint8_t cf_u8;
typedef uint16_t cf_u16;
typedef uint32_t cf_u32;
typedef uint64_t cf_u64;
typedef uintptr_t cf_uptr;
typedef size_t cf_usize;

// typedef the signed types
typedef int8_t cf_i8;
typedef int16_t cf_i16;
typedef int32_t cf_i32;
typedef int64_t cf_i64;
typedef intptr_t cf_iptr;
typedef ptrdiff_t cf_isize;

//typedef logic types
typedef bool cf_bool;

#endif // CF_TYPES_H