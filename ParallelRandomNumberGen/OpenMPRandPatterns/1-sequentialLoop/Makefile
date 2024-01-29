CPP=g++
INCLUDE=/usr/local/include/trng
LIB=trng4
TARGETS = loopRepetitions

all: $(TARGETS)

getCommandLine.o: getCommandLine.c getCommandLine.h
	$(CPP) -c -o getCommandLine.o getCommandLine.c

loopRepetitions: loopRepetitions.cpp getCommandLine.o
	$(CPP) -o loopRepetitions getCommandLine.o loopRepetitions.cpp -I$(INCLUDE) -l$(LIB) 

clean:
	rm -f $(TARGETS) getCommandLine.o
