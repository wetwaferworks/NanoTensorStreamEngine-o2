CC = gcc
CFLAGS = -O2 -nostdlib -static -ffreestanding -fno-stack-protector -fno-builtin

all: vq-0.4 vq-0.2 vq-0.02

vq-0.4: examples/vq-0.4.c
	$(CC) $(CFLAGS) -o $@ $<

vq-0.2: examples/vq-0.2.c
	$(CC) $(CFLAGS) -o $@ $<

vq-0.02: examples/vq-0.02.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f vq-0.4 vq-0.2 vq-0.02

.PHONY: all clean
