#include "MEMORY/cf_memory.h"
#include "MEMORY/cf_array.h"

#include "RUNTIME/cf_status.h"

#include "SECURITY/cf_base64.h"
#include "SECURITY/cf_hex.h"

#include "MATH/cf_math.h"

#include "TEXT/cf_string.h"

#include <time.h>
#include <stdio.h>
#include <string.h>

static cf_u8 test_g8_mul_mod_reference(cf_u8 p, cf_u8 q)
{
  cf_u16 product = 0;

  for (cf_u16 shift = 0; shift < 8; shift++)
  {
    if (((cf_u16)q >> shift) & 0x01)
      product ^= (cf_u16)p << shift;
  }

  for (int bit = 15; bit >= 8; bit--)
  {
    if (product & (cf_u16)(1U << bit))
      product ^= (cf_u16)(0x11BU << (bit - 8));
  }

  return (cf_u8)product;
}

int main(void)
{
  for (cf_u16 p = 0; p <= 0xFF; p++)
  {
    for (cf_u16 q = 0; q <= 0xFF; q++)
    {
      cf_u8 g8_mul_result = cf_math_g8_mul_mod((cf_u8)p, (cf_u8)q);
      cf_u8 g8_mul_expected = test_g8_mul_mod_reference((cf_u8)p, (cf_u8)q);

      if(g8_mul_result != g8_mul_expected)
      {
        printf(
          "cf_math_g8_mul_mod(0x%02X, 0x%02X) failed: expected 0x%02X, got 0x%02X\n",
          p,
          q,
          g8_mul_expected,
          g8_mul_result);

        return 1;
      }
    }
  }

  printf("cf_math_g8_mul_mod passed all 65536 input pairs\n");

  cf_string str;
  cf_string_init(&str, 1);
  cf_string_from_cstr(&str, "Hello World!\n");
  // cf_string_strip(&str);
  cf_bytes byte;
  cf_buffer_as_bytes(&str, &byte, 0, str.len - 1);
  cf_string buff;
  cf_string_init(&buff, 1);
  cf_base64_encode(&buff, byte);
  cf_buffer d;
  cf_buffer_init(&d, 1);
  // cf_hex_decode(&d, &buff);
  printf("%s", ((char *)buff.data));
  printf("\n");
  cf_base64_decode(&d, &buff);
  printf("%s", ((char *)d.data));
  printf("%zu\n", d.len);
  // printf("%s", ((char *)d.data));


  // cf_array arr;
  // cf_string_replace(&str,';', '-');
  // cf_string_split(&arr, &str, ';');
  // for (size_t i = 0; i < arr.len; i++)
  //   printf("%s\n", (char *)arr.data[i].data);
  
  return 0;
}
