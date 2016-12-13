
CC=gcc
CFLAGS=-Wall -g -O2

LIBS = -lm -lrt -lOpenCL
LDFLAGS = ${LIBS}

all: vggnet

.PHONY: all vggnet

vggnet: vggnet.o main.o
	${CC} $^ -o $@ ${LDFLAGS}

clean:
	rm -f vggnet.o main.o vggnet task*
