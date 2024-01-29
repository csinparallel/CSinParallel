PROG = calcPiMPI
CC = mpic++
CFLAGS = -Wall -std=c++11
LFLAGS = -o $(PROG)
INCLUDE=/usr/local/include/trng
LIB=trng4

$(PROG): $(PROG).C
	$(CC) $(CFLAGS) $(PROG).C $(LFLAGS) -I$(INCLUDE) -l$(LIB)

clean:
	rm -f $(PROG) a.out *~ *# *.o

