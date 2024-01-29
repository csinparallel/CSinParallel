CPP=g++
OMP=-fopenmp
INCLUDE=/usr/local/include/trng
LIB=trng4
TARGETS = coinFlip_omp coinFlip_seq

all: $(TARGETS)

getCommandLine.o: ./utils/getCommandLine.c ./utils/getCommandLine.h
	$(CPP) -c -o getCommandLine.o ./utils/getCommandLine.c

coinFlip_seq: coinFlip_seq.cpp getCommandLine.o
	$(CPP) $(OMP) -o coinFlip_seq getCommandLine.o coinFlip_seq.cpp -I$(INCLUDE) -l$(LIB) 

coinFlip_omp: coinFlip_omp.cpp getCommandLine.o
	$(CPP) $(OMP) -o coinFlip_omp getCommandLine.o coinFlip_omp.cpp -I$(INCLUDE) -l$(LIB) 

clean:
	rm -f $(TARGETS) getCommandLine.o
