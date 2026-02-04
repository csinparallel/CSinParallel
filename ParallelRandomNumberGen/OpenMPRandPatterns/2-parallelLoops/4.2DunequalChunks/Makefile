CPP=g++
OMP=-fopenmp
INCLUDE=/usr/local/include/trng
LIB=trng4
TARGETS = 2DArrayNestedLoop 

all: $(TARGETS)

getCommandLine.o: ../utils/getCommandLine.c ../utils/getCommandLine.h
	$(CPP) -c -o getCommandLine.o ../utils/getCommandLine.c

2DArrayNestedLoop: 2DArrayNestedLoop.cpp getCommandLine.o ../utils/chunks.h grid.h
	$(CPP) $(OMP) -o 2DArrayNestedLoop getCommandLine.o 2DArrayNestedLoop.cpp -I$(INCLUDE) -l$(LIB) 

clean:
	rm -f $(TARGETS) getCommandLine.o
