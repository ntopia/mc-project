
CC=mpicc
CFLAGS=-Wall -g -O2 -std=gnu99 -march=native -msse -mfpmath=sse -funroll-loops -ftree-vectorize -I$(SNUCLROOT)/inc

LIBS=-lm -lrt -lpthread -L${SNUCLROOT}/lib -lsnucl_cluster
LDFLAGS=${LIBS}

all: vggnet

.PHONY: all vggnet

vggnet: vggnet.o main.o
	${CC} $^ -o $@ ${LDFLAGS}

run: vggnet
	thorq --add --mode snucl --nodes 4 --device gpu ./vggnet image_list.txt

clean:
	rm -rf vggnet.o main.o vggnet task*
