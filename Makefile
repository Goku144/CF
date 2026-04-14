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
INC := public/inc
FLAG := -Wall -Wextra -Wpedantic -Werror
SRCS := $(shell find lib/src -name '*.c')
OBJS := $(patsubst lib/src/%.c, lib/bin/%.o, $(SRCS))

############
# Build App
############
app: runApp

runApp: app/build/app
	./$<

app/build/app: app/bin/app.o $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $^ -o $@

app/bin/app.o: app/src/app.c
	@mkdir -p $(dir $@)
	$(CC) $(FLAG) -I$(INC) -c $< -o $@

############
# Build Lib
############
lib: $(OBJS)

# % matches the full subpath, e.g. ALLOCATOR/cf_alloc
lib/bin/%.o: lib/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(FLAG) -I$(INC) -c $< -o $@

############
# Run Tests
############
test: runTests

runTests: tests/build/test
	./$< > public/doc/test.result.txt 2>&1

tests/build/test: tests/bin/test.o $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $^ -o $@

tests/bin/test.o: tests/src/test.c
	@mkdir -p $(dir $@)
	$(CC) $(FLAG) -I$(INC) -c $< -o $@

############
# Utility
############

clean:
	rm -rf lib/bin app/bin app/build tests/bin tests/build

.PHONY: app runApp lib test runTests clean