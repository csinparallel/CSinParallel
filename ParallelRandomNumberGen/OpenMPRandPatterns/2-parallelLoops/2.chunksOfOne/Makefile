CPP=g++
OMP=-fopenmp
INCLUDE=/usr/local/include/trng
LIB=trng4
TARGETS = loopChunksOfOne

all: $(TARGETS)

getCommandLine.o: ../utils/getCommandLine.c ../utils/getCommandLine.h
	$(CPP) -c -o getCommandLine.o ../utils/getCommandLine.c

loopChunksOfOne: loopChunksOfOne.cpp getCommandLine.o
	$(CPP) $(OMP) -o loopChunksOfOne getCommandLine.o loopChunksOfOne.cpp -I$(INCLUDE) -l$(LIB) 

clean:
	rm -f $(TARGETS) getCommandLine.o
