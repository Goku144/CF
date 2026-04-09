CC := gcc
LIB :=
INC := public/inc
FLAG := -Wall -Wextra -Wpedantic -Werror













############
# Run Tests
############
test: runTests

runTests: tests/build/test
	./$<

tests/build/test: tests/bin/test.o
	$(CC) $^ -o $@

tests/bin:
	mkdir -p $@

tests/build:
	mkdir -p $@

tests/bin/test.o: tests/src/test.c $(INC)/cf_types.h tests/bin tests/build
	$(CC) $(FLAG) -I$(INC) -c $< -o $@