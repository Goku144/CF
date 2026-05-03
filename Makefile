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

INC := -Ipublic/inc -I/usr/local/cuda/include

############
# C OPTIONS
############

CC ?= $(if $(shell command -v gcc 2>&1))
CC_AVAILABLE := $(if $(CC),1,0)
FLAG_C := -Wall -Wextra -Wpedantic -Werror -O3
LIBS_C := -ldnnl -lmimalloc
SRCS_C := $(shell find lib/src -name '*.c')
OBJS_C := $(patsubst lib/src/%.c, lib/bin/%.o, $(SRCS_C))

##############
# ASM OPTIONS
##############

ASM ?= $(shell command -v nasm 2>&1)
ASM_AVAILABLE := $(if $(ASM),1,0)
FLAG_ASM := -f elf64 -w+all
LIBS_ASM :=
SRCS_ASM := $(shell find lib/src -name '*.asm')
OBJS_ASM := $(patsubst lib/src/%.asm, lib/bin/%.o, $(SRCS_ASM))

###############
# NVCC OPTIONS 
###############

NVCC ?= $(shell command -v nvcc 2>&1)
CUDA_AVAILABLE := $(if $(NVCC),1,0)
FLAG_CUDA := -O3 -Wno-deprecated-gpu-targets
LIBS_CUDA := -lcudnn -lcusparse -lcusolver -lcurand -lcublasLt -lcublas -lcudart
SRCS_CUDA := $(shell find lib/src -name '*.cu')
OBJS_CUDA := $(patsubst lib/src/%.cu, lib/bin/%.o, $(SRCS_CUDA))

#############
# CONDITIONS
#############

ifeq ($(CC_AVAILABLE),0)
$(info try to install: sudo apt install gcc)
$(error because GCC is not available no compiling!)
endif

ifeq ($(ASM_AVAILABLE),0)
$(info try to install: sudo apt install nasm)
$(error because NASM is not available no optimized functions!)
endif

ifeq ($(CUDA_AVAILABLE),0)
$(info visit the website to install: https://developer.nvidia.com/cuda/toolkit)
$(error because CUDA is not available compiling .cu sources)
endif

############
# Build App
############

app: runApp

runApp: app/build/app
	@./$<

app/build/app: app/bin/app.o $(OBJS_C) $(OBJS_ASM) $(OBJS_CUDA)
	@mkdir -p $(dir $@)
	@$(NVCC) $^ $(LIBS_CUDA) $(LIBS_C) -o $@

app/bin/app.o: app/src/app.cu
	@mkdir -p $(dir $@)
	@$(NVCC) $(INC) -c $< -o $@

############
# Build Lib
############

lib: $(OBJS_C) $(OBJS_ASM) $(OBJS_CUDA)

lib/bin/%.o: lib/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(FLAG_C) $(INC) -c $< -o $@

lib/bin/%.o: lib/src/%.asm
	@mkdir -p $(dir $@)
	$(ASM) $(FLAG_ASM) $< -o $@

lib/bin/%.o: lib/src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAG_CUDA) $(INC) -c $< -o $@

############
# Run Tests
############

test: runTests

runTests: tests/build/test
	@mkdir -p public/doc/test
	@./$< > public/doc/test/test.result.txt 2>&1

tests/build/test: tests/bin/test.o $(OBJS_C) $(OBJS_ASM) $(OBJS_CUDA)
	@mkdir -p $(dir $@)
	@$(NVCC) $^ $(LIBS_CUDA) $(LIBS_C) -o $@

tests/bin/test.o: tests/src/test.c
	@mkdir -p $(dir $@)
	@$(CC) $(FLAG_C) $(INC) -c $< -o $@

############
# Utility
############

clean:
	rm -rf lib/bin app/bin app/build tests/bin tests/build

.PHONY: app runApp lib test runTests clean 