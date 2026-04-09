#include "../../public/inc/cf_types.h"

CF_STATIC_ASSERT(sizeof(cf_u8) == 1, "cf_u8 must be 1 byte");

CF_STATIC_ASSERT(sizeof(cf_u8) == 2, "cf_u8 test failed");
/* Uncomment to force failure */
/* CF_STATIC_ASSERT(sizeof(cf_u8) == 2, "cf_u8 test failed"); */

int main(void)
{
    return 0;
}