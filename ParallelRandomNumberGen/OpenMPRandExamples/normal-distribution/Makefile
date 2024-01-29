CPP=g++ -Ofast
OMP=-fopenmp
INCLUDE=/usr/local/include/trng
LIB=trng4
TARGETS = histogram_omp histogram_seq

all: $(TARGETS)

getCommandLineSeq.o: ./utils/getCommandLineSeq.c ./utils/getCommandLineSeq.h
	$(CPP) -c -o getCommandLineSeq.o ./utils/getCommandLineSeq.c

getCommandLinePar.o: ./utils/getCommandLinePar.c ./utils/getCommandLinePar.h
	$(CPP) -c -o getCommandLinePar.o ./utils/getCommandLinePar.c

######### Sequential
histogram_seq: histogram_seq.cpp getCommandLineSeq.o
	$(CPP) ${OMP} -o histogram_seq getCommandLineSeq.o histogram_seq.cpp -I$(INCLUDE) -l$(LIB)

######### OpenMP
histogram_omp: histogram_omp.cpp getCommandLinePar.o
	$(CPP) ${OMP} -o histogram_omp getCommandLinePar.o histogram_omp.cpp -I$(INCLUDE) -l$(LIB)

######### clean
clean:
		rm -f $(TARGETS) *.o
