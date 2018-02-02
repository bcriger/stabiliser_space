CC=gcc
CFLAGS= -Wall -O3 -fPIC
RM=rm -f
.PHONY: all clean

all: gf2_mat.so
clean:
	$(RM) *.o *.so

gf2_mat.so: gf2_mat.o
	$(LINK.c) -shared $< -o $@

gf2_mat.o: gf2_mat.c