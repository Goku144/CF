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

CC  := gcc
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
ASM := nasm
INC := public/inc
FLAG := -Wall -Wextra -Wpedantic -Werror -O3
FLAG_CUDA := -O3
FLAG_ASM := -f elf64 -w+all
SRCS := $(shell find lib/src -name '*.c')
OBJS := $(patsubst lib/src/%.c, lib/bin/%.o, $(SRCS))
SRCS_CUDA := $(shell find lib/src -name '*.cu')
OBJS_CUDA := $(patsubst lib/src/%.cu, lib/bin/%.o, $(SRCS_CUDA))
SRCS_ASM := $(shell find lib/src -name '*.asm')
OBJS_ASM := $(patsubst lib/src/%.asm, lib/bin/%.o, $(SRCS_ASM))

############
# Build App
############
app: runApp

runApp: app/build/app
	@./$<

app/build/app: app/bin/app.o $(OBJS) $(OBJS_CUDA) $(OBJS_ASM)
	@mkdir -p $(dir $@)
	@$(NVCC) $^ -o $@

app/bin/app.o: app/src/app.c
	@mkdir -p $(dir $@)
	@$(CC) $(FLAG) -I$(INC) -c $< -o $@

############
# Build Lib
############
lib: $(OBJS) $(OBJS_CUDA) $(OBJS_ASM)

# % matches the full subpath, e.g. ALLOCATOR/cf_alloc
lib/bin/%.o: lib/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(FLAG) -I$(INC) -c $< -o $@

lib/bin/%.o: lib/src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAG_CUDA) -I$(INC) -c $< -o $@

lib/bin/%.o: lib/src/%.asm
	@mkdir -p $(dir $@)
	$(ASM) $(FLAG_ASM) $< -o $@

############
# Run Tests
############
test: runTests

runTests: tests/build/test
	@./$< > public/doc/test.result.txt 2>&1

tests/build/test: tests/bin/test.o $(OBJS) $(OBJS_CUDA)
	@mkdir -p $(dir $@)
	@$(NVCC) $^ -o $@

tests/bin/test.o: tests/src/test.c
	@mkdir -p $(dir $@)
	@$(CC) $(FLAG) -I$(INC) -c $< -o $@

############
# Utility
############

check-nasm: 
	@if ! command -v nasm >/dev/null 2>&1; then\
		sudo apt update && sudo apt install -y nasm;\
	fi
	@nasm -v

clean:
	rm -rf lib/bin app/bin app/build tests/bin tests/build

.PHONY: app runApp lib test runTests check-nasm clean 
