/*
 * CF Framework
 * Copyright (C) 2026 Orion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "MATH/cf_math.h"

#include <stdio.h>

int main(void)
{
  printf("cf_math lifecycle tests moved to app/src/app.c for the CUDA handler redesign.\n");
  printf("basic helper: rotl8(0x12, 4) = 0x%02x\n", cf_math_rotl8(0x12U, 4U));
  return 0;
}
