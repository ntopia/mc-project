
CC=mpicc
CFLAGS=-Wall -g -O3 -std=gnu99 -march=native -msse -mfpmath=sse -funroll-loops -ftree-vectorize

LIBS=-lm -lrt -lOpenCL -lpthread
LDFLAGS=${LIBS}

all: vggnet

.PHONY: all vggnet

vggnet: vggnet.o main.o
	${CC} $^ -o $@ ${LDFLAGS}

run: vggnet
	thorq --add --mode mpi --nodes 4 --device gpu ./vggnet image_list.txt

clean:
	rm -f vggnet.o main.o vggnet task*
