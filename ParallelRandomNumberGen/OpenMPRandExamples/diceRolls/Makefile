CPP=g++
OMP=-fopenmp
INCLUDE=/usr/local/include/trng
LIB=trng4
TARGETS = diceSim_omp 

all: $(TARGETS)

getCommandLine.o: ./utils/getCommandLine.c ./utils/getCommandLine.h
	$(CPP) -c -o getCommandLine.o ./utils/getCommandLine.c

diceSim_omp: diceSim_omp.cpp getCommandLine.o
	$(CPP) $(OMP) -o diceSim_omp getCommandLine.o diceSim_omp.cpp -I$(INCLUDE) -l$(LIB) 

clean:
	rm -f $(TARGETS) getCommandLine.o
