CC := gcc
LIB :=
INC := public/inc
FLAG := -Wall -Wextra -Wpedantic -Werror
SRCS := $(wildcard lib/src/*.c)
OBJS := $(patsubst lib/src/%.c,lib/bin/%.o,$(SRCS))











############
# Build Lib
############
lib: $(OBJS)

lib/bin/%.o: lib/src/%.c
	$(CC) $(CFLAGS) -I$(INC) -c $< -o $@

############
# Run Tests
############
test: runTests

runTests: tests/build/test
	./$<

tests/build/test: tests/bin/test.o $(OBJS)
	$(CC) $^ -o $@

tests/bin/test.o: tests/src/test.c $(INC)/cf_types.h
	$(CC) $(FLAG) -I$(INC) -c $< -o $@