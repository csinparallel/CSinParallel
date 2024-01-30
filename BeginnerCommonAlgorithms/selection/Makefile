
PROGS = selection_seq selectionOMP_critical selectionOMP_lock selectionOMP_reduction selectionMPI1 selectionMPI2

all: $(PROGS)

selection_seq: selection_seq.c
	gcc -o selection_seq selection_seq.c

selectionOMP_critical: selectionOMP_critical.c
	gcc -o selectionOMP_critical selectionOMP_critical.c -fopenmp

selectionOMP_lock: selectionOMP_lock.c
	gcc -o selectionOMP_lock selectionOMP_lock.c -fopenmp

selectionOMP_reduction: selectionOMP_reduction.c
	gcc -o selectionOMP_reduction selectionOMP_reduction.c -fopenmp

selectionMPI1: selectionMPI1.c
	mpicc -o selectionMPI1 selectionMPI1.c

selectionMPI2: selectionMPI2.c
	mpicc -o selectionMPI2 selectionMPI2.c

clean:
	rm -f $(PROGS)
