
PROGS = ttt-serial ttt-omp

all: $(PROGS)

ttt-serial: ttt-serial.c
	gcc -o ttt-serial ttt-serial.c

ttt-omp: ttt-omp.c
	gcc -o ttt-omp ttt-omp.c -fopenmp

clean:
	rm -f $(PROGS)
	