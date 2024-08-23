CPP=g++
OMP=-fopenmp
INCLUDE=/usr/local/include/trng
LIB=trng4
TARGETS = nestedLoopsMC

all: $(TARGETS)

getCommandLine.o: getCommandLine.c getCommandLine.h
	$(CPP) -c -o getCommandLine.o getCommandLine.c

nestedLoopsMC: nestedLoopsMC.cpp getCommandLine.o
	$(CPP) $(OMP) -o nestedLoopsMC getCommandLine.o nestedLoopsMC.cpp -I$(INCLUDE) -l$(LIB) 

clean:
	rm -f $(TARGETS) getCommandLine.o
