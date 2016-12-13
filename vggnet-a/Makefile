
CC=gcc
CFLAGS=-Wall -g -O2 -std=gnu99

LIBS = -lm -lrt -lOpenCL
LDFLAGS = ${LIBS}

all: vggnet

.PHONY: all vggnet

vggnet: vggnet.o main.o
	${CC} $^ -o $@ ${LDFLAGS}

run: vggnet
	thorq --add --device cpu ./vggnet image_list.txt

clean:
	rm -f vggnet.o main.o vggnet task*
