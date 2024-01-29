CPP=g++
OMP=-fopenmp
INCLUDE=/usr/local/include/trng
LIB=trng4
TARGETS = roulette_omp roulette_seq

all: $(TARGETS)

getCommandLine.o: ./utils/getCommandLine.c ./utils/getCommandLine.h
	$(CPP) -c -o getCommandLine.o ./utils/getCommandLine.c

roulette_seq: roulette_seq.cpp getCommandLine.o
	$(CPP) $(OMP) -o roulette_seq getCommandLine.o roulette_seq.cpp -I$(INCLUDE) -l$(LIB) 

roulette_omp: roulette_omp.cpp getCommandLine.o
	$(CPP) $(OMP) -o roulette_omp getCommandLine.o roulette_omp.cpp -I$(INCLUDE) -l$(LIB) 

clean:
	rm -f $(TARGETS) *.o
