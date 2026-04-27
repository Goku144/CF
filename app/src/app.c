#include "MEMORY/cf_memory.h"
#include "MEMORY/cf_array.h"

#include "RUNTIME/cf_io.h"
#include "RUNTIME/cf_log.h"
#include "RUNTIME/cf_status.h"
#include "RUNTIME/cf_time.h"

#include "SECURITY/cf_aes.h"
#include "SECURITY/cf_base64.h"
#include "SECURITY/cf_hex.h"

#include "MATH/cf_math.h"
#include "MATH/cf_tensor.h"

#include "TEXT/cf_ascii.h"
#include "TEXT/cf_string.h"

#include <stdio.h>

int main(void)
{
  cf_tensor a, b, out;

  cf_tensor_init(&a, (cf_usize[]){3, 0, 0, 0, 0, 0, 0, 0}, 1, CF_TENSOR_DOUBLE);
  cf_tensor_init(&b, (cf_usize[]){3, 0, 0, 0, 0, 0, 0, 0}, 1, CF_TENSOR_DOUBLE);
  cf_tensor_init(&out, (cf_usize[]){1, 1, 0, 0, 0, 0, 0, 0}, 2, CF_TENSOR_DOUBLE);

  ((double *)a.data)[0] = 1.0;


  // ((double *)a.data)[3] = 4.0;
  // ((double *)a.data)[4] = 5.0;
  // ((double *)a.data)[5] = 6.0;

  ((double *)b.data)[0] = 7.0;


  // ((double *)b.data)[3] = 10.0;
  // ((double *)b.data)[4] = 11.0;
  // ((double *)b.data)[5] = 12.0;

  CF_LOG_INFO(cf_status_as_char(cf_tensor_matrice_mul(&a, &b, &out)));
  cf_tensor_print(&a);
  cf_tensor_print(&b);
  cf_tensor_print(&out);

  cf_tensor_destroy(&a);
  cf_tensor_destroy(&b);
  cf_tensor_destroy(&out);
  return 0;
}
