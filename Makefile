CC = gcc

CFLAGS = -O2 -mavx -mfma -Wall -fopenmp
LDFLAGS_OPENBLAS = -lopenblas

SRCS = main.c
TARGETS = main

all: $(TARGETS)

main: main.c
	$(CC) $(CFLAGS) main.c -o main $(LDFLAGS_OPENBLAS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
