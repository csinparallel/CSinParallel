CPP=g++
OMP=-fopenmp
INCLUDE=/usr/local/include/trng
LIB=trng4
TARGETS = loopEqualChunks

all: $(TARGETS)

getCommandLine.o: ../utils/getCommandLine.c ../utils/getCommandLine.h
	$(CPP) -c -o getCommandLine.o ../utils/getCommandLine.c

loopEqualChunks: loopEqualChunks.cpp getCommandLine.o
	$(CPP) $(OMP) -o loopEqualChunks getCommandLine.o loopEqualChunks.cpp -I$(INCLUDE) -l$(LIB) 

clean:
	rm -f $(TARGETS) getCommandLine.o
