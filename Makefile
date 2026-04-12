CC := gcc
LIB :=
INC := public/inc
FLAG := -Wall -Wextra -Wpedantic -Werror
SRCS := $(wildcard lib/src/*.c)
OBJS := $(patsubst lib/src/%.c,lib/bin/%.o,$(SRCS))

############
# Build App
############

app: runApp

runApp: app/build/app
	./$<

app/build/app: app/bin/app.o $(OBJS)
	$(CC) $^ -o $@

app/bin/app.o: app/src/app.c $(INC)/cf_types.h
	$(CC) $(FLAG) -I$(INC) -c $< -o $@

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
	./$< > public/doc/test.result.txt 2>> public/doc/test.result.txt

tests/build/test: tests/bin/test.o $(OBJS)
	$(CC) $^ -o $@

tests/bin/test.o: tests/src/test.c $(INC)/cf_types.h
	$(CC) $(FLAG) -I$(INC) -c $< -o $@