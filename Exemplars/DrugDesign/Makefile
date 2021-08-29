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

Pyjama.jar:
	wget https://www.csc.tntech.edu/pdcincs/resources/modules/tools/updated/Pyjama.zip
	unzip -o Pyjama.zip
	cp Pyjama/Pyjama*.jar Pyjama.jar
	rm -r -f Pyjama.zip Pyjama

DDPyjama.class:	DDPyjama.java Pyjama.jar
	java -jar Pyjama.jar DDPyjama.java

DDPyjama:	DDPyjama.class
	java -cp .:Pyjama.jar DDPyjama $(ARGS)

clean:
	rm -f dd_serial dd_omp_static dd_omp_dynamic dd_threads dd_mpi Pyjama.jar DDPyjama*class

