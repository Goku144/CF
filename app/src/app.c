#include "RUNTIME/cf_time.h"
#include "RUNTIME/cf_log.h"
#include "MATH/cf_math.h"

int main()
{
  cf_math a = {0}, b = {0}, c = {0};
  CF_LOG_ERROR(cf_status_as_char(cf_math_init_eye(&a, CF_NULL)));
  CF_LOG_ERROR(cf_status_as_char(cf_math_init_eye(&b, CF_NULL)));
  CF_LOG_ERROR(cf_status_as_char(cf_math_init_eye(&c, CF_NULL)));
  
  
  
  return 0;
}