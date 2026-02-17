CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -ldl -lm

proof: proof.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f proof

.PHONY: clean
