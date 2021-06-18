# makefile for drug design, for Raspberry Pi

all: dd_serial dd_omp_static dd_omp_dynamic dd_threads dd_mpi

dd_serial:  dd_serial.cpp
	g++ -o dd_serial dd_serial.cpp

dd_omp_static: dd_omp_static.cpp
	g++ -o dd_omp_static dd_omp_static.cpp -lm -fopenmp -std=c++17

dd_omp_dynamic: dd_omp_dynamic.cpp
	g++ -o dd_omp_dynamic dd_omp_dynamic.cpp -lm -fopenmp -std=c++17

dd_threads: dd_threads.cpp
	g++ -o dd_threads dd_threads.cpp -lm -std=c++11 -pthread

dd_mpi:  dd_mpi.cpp
	mpiCC -std=c++11 -o dd_mpi dd_mpi.cpp

clean:
	rm -f dd_serial dd_omp_static dd_omp_dynamic dd_threads dd_mpi

