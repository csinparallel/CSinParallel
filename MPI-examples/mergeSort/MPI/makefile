OBJ    = mergeSortMPI
CC     = mpicc
CFLAGS = -Wall -ansi -pedantic -std=c99 
LFLAGS = -lm -o $(OBJ)

$(OBJ): $(OBJ).c
	$(CC) $(CFLAGS) $(OBJ).c $(LFLAGS)

clean:
	rm -f $(OBJ)
