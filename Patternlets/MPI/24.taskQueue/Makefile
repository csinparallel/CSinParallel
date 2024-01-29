PROG   = taskQueue
CC     = mpicc
CFLAGS = -Wall -pedantic -std=gnu99
LFLAGS = -o $(PROG) -lm

$(PROG): $(PROG).c
	$(CC) $(CFLAGS) $(PROG).c $(LFLAGS)

clean:
	rm -f $(PROG) a.out *~ *# *.o
