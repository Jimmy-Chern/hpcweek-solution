TARGET = compress
SRC = compress.c

CC = gcc
CFLAGS = -Wall -Wextra -O3 -fopenmp -lpthread
LDFLAGS = -lhts -lpthread

DIR=/hpcweek/home/s3240106190/htslib_install
LIB=$(DIR)/lib
INCLUDE=$(DIR)/include

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -I$(INCLUDE) -L$(LIB)

clean:
	rm -f $(TARGET)

.PHONY: all clean
