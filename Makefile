CC := gcc
LIB :=
INC := public/inc
FLAG := -Wall -Wextra -Wpedantic -Werror













############
# Run Tests
############
test: runTests

runTests: tests/build/test
	./tests/build/test

tests/build/test: tests/bin/test.o
	$(CC) tests/bin/test.o -o tests/build/test

tests/bin/test.o: tests/src/test.c
	$(CC) $(FLAG) -I$(INC)/* -c tests/src/test.c -o tests/bin/test.o