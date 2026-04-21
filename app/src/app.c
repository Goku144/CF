#include "MEMORY/cf_memory.h"
#include "MEMORY/cf_array.h"

#include "RUNTIME/cf_status.h"

#include "TEXT/cf_string.h"

#include <time.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
  cf_string str;
  cf_string_init(&str, 1);
  cf_string_from_cstr(&str, " printf(\"hmida\\n\"); exit(0);  ");
  cf_string_strip(&str);

  cf_array arr;
  cf_string_replace(&str,';', '-');
  cf_string_split(&arr, &str, ';');
  for (size_t i = 0; i < arr.len; i++)
    printf("%s\n", (char *)arr.data[i].data);
  
  return 0;
}