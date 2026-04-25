##
# CF Framework
# Copyright (C) 2026 Orion
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
##

##############
# PUBLIC VARS
##############

INC := public/inc

############
# C OPTIONS
############

CC ?= $(shell command -v nvcc >/dev/null 2>&1)
CC_AVAILABLE := $(shell command -v $(CC) >/dev/null 2>&1 && echo 1 || echo 0)
FLAG_C := -Wall -Wextra -Wpedantic -Werror -O3
SRCS_C :=
OBJS_C :=

##############
# ASM OPTIONS
##############

ASM ?= $(shell command -v nvcc >/dev/null 2>&1)
ASM_AVAILABLE := $(shell command -v $(ASM) >/dev/null 2>&1 && echo 1 || echo 0)
FLAG_ASM := -f elf64 -w+all
SRCS_ASM :=
OBJS_ASM :=

###############
# NVCC OPTIONS 
###############

NVCC ?= $(shell command -v nvcc >/dev/null 2>&1)
CUDA_AVAILABLE := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1 || echo 0)
FLAG_CUDA := -O3
SRCS_CUDA :=
OBJS_CUDA :=

#############
# CONDITIONS
#############

ifeq ($(CC_AVAILABLE),1)
SRCS_C := $(shell find lib/src -name '*.c')
OBJS_C := $(patsubst lib/src/%.c, lib/bin/%.o, $(SRCS_C))
else
$(info try to install: sudo apt install gcc)
$(error because GCC is not available no compiling!)
endif

ifeq ($(ASM_AVAILABLE),1)
SRCS_ASM := $(shell find lib/src -name '*.asm')
OBJS_ASM := $(patsubst lib/src/%.asm, lib/bin/%.o, $(SRCS_ASM))
else
$(info try to install: sudo apt install nasm)
$(warning because NASM is not available no optimized functions!)
endif

ifeq ($(CUDA_AVAILABLE),1)
FLAG_C += -DCF_CUDA_AVAILABLE=1
SRCS_CUDA := $(shell find lib/src -name '*.cu')
OBJS_CUDA := $(patsubst lib/src/%.cu, lib/bin/%.o, $(SRCS_CUDA))
LINK := $(NVCC)
else
$(info visit the website to intsall: https://developer.nvidia.com/cuda/toolkit)
$(warning because CUDA is not available running on cpu only!)
LINK := $(CC)
endif

############
# Build App
############

app: runApp

runApp: app/build/app
	@./$<

app/build/app: app/bin/app.o $(OBJS_C) $(OBJS_ASM) $(OBJS_CUDA)
	@mkdir -p $(dir $@)
	@$(LINK) $^ -o $@

app/bin/app.o: app/src/app.c
	@mkdir -p $(dir $@)
	@$(CC) $(FLAG_C) -I$(INC) -c $< -o $@

############
# Build Lib
############

lib: $(OBJS_C) $(OBJS_ASM) $(OBJS_CUDA)

lib/bin/%.o: lib/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(FLAG_C) -I$(INC) -c $< -o $@

lib/bin/%.o: lib/src/%.asm
	@mkdir -p $(dir $@)
	$(ASM) $(FLAG_ASM) $< -o $@

lib/bin/%.o: lib/src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAG_CUDA) -I$(INC) -c $< -o $@

############
# Run Tests
############

test: runTests

runTests: tests/build/test
	@./$< > public/doc/test.result.txt 2>&1

tests/build/test: tests/bin/test.o $(OBJS_C) $(OBJS_ASM) $(OBJS_CUDA)
	@mkdir -p $(dir $@)
	@$(LINK) $^ -o $@

tests/bin/test.o: tests/src/test.c
	@mkdir -p $(dir $@)
	@$(CC) $(FLAG_C) -I$(INC) -c $< -o $@

############
# Utility
############

clean:
	rm -rf lib/bin app/bin app/build tests/bin tests/build

.PHONY: app runApp lib test runTests clean 
