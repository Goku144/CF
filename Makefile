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

INC := -Ipublic/inc -I/usr/local/cuda/include -I/usr/include

###############
# NVCC OPTIONS 
###############

NVCC ?= $(shell command -v nvcc 2>&1)
CUDA_AVAILABLE := $(if $(NVCC),1,0)
FLAG_CUDA := -O3 -Wno-deprecated-gpu-targets -arch=sm_75 -diag-suppress 550
LIBS_CUDA := -lcudnn -lcusparse -lcusolver -lcurand -lcublasLt -lcublas -lcudart
SRCS_CUDA := $(shell find lib/src -name '*.cu')
OBJS_CUDA := $(patsubst lib/src/%.cu, lib/bin/%.o, $(SRCS_CUDA))

#############
# CONDITIONS
#############

ifeq ($(CUDA_AVAILABLE),0)
$(info visit the website to install: https://developer.nvidia.com/cuda/toolkit)
$(error because CUDA is not available compiling .cu sources)
endif

############
# Build App
############

app: runApp

runApp: app/build/app
	@mkdir -p public/checkpoints
	@./$< 2 public/checkpoints

app/build/app: app/bin/app.o $(OBJS_CUDA)
	@mkdir -p $(dir $@)
	@$(NVCC) $^ $(LIBS_CUDA) -o $@

app/bin/app.o: app/src/app.cu
	@mkdir -p $(dir $@)
	@$(NVCC) $(FLAG_CUDA) $(INC) -c $< -o $@

############
# Build Lib
############

lib: $(OBJS_CUDA)

lib/bin/%.o: lib/src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAG_CUDA) $(INC) -c $< -o $@

############
# Utility
############

clean:
	rm -rf lib/bin app/bin app/build

.PHONY: app runApp lib clean