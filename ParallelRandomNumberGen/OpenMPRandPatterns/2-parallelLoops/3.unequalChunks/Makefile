CPP=g++
OMP=-fopenmp
INCLUDE=/usr/local/include/trng
LIB=trng4
TARGETS = loopUnEqualChunks arrayLoopUnequalChunks

all: $(TARGETS)

getCommandLine.o: ../utils/getCommandLine.c ../utils/getCommandLine.h
	$(CPP) -c -o getCommandLine.o ../utils/getCommandLine.c

loopUnEqualChunks: loopUnEqualChunks.cpp getCommandLine.o ../utils/chunks.h
	$(CPP) $(OMP) -o loopUnEqualChunks getCommandLine.o loopUnEqualChunks.cpp -I$(INCLUDE) -l$(LIB) 

arrayLoopUnequalChunks: arrayLoopUnequalChunks.cpp getCommandLine.o ../utils/chunks.h
	$(CPP) $(OMP) -o arrayLoopUnequalChunks getCommandLine.o arrayLoopUnequalChunks.cpp -I$(INCLUDE) -l$(LIB)

clean:
	rm -f $(TARGETS) getCommandLine.o
