
CC=gcc
CFLAGS=-Wall -O3 -std=gnu99 -march=native -msse -mfpmath=sse -funroll-loops -ftree-vectorize

LIBS=-lm -lrt -lOpenCL
LDFLAGS=${LIBS}

all: vggnet

.PHONY: all vggnet

vggnet: vggnet.o main.o
	${CC} $^ -o $@ ${LDFLAGS}

run: vggnet
	thorq --add --device cpu ./vggnet image_list.txt

clean:
	rm -f vggnet.o main.o vggnet task*
