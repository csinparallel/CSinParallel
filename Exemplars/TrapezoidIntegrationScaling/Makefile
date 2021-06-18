CC=gcc
OMP=-fopenmp

all: trap-seq trap-omp

trap-seq: trap-seq.c
	${CC} ${OMP} -o trap-seq trap-seq.c -lm

trap-omp: trap-omp.c
	${CC} ${OMP} -o trap-omp trap-omp.c -lm

clean:
	rm -f trap-omp trap-seq
