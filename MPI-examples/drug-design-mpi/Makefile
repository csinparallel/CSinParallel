# which compilers to use
CC        = mpicc 
CPP       = mpicxx         # C++ version

MPIEQ_OBJS  = dd_functions.o dd_mpi_equal_chunks.o
MPIEQ_EXEC  =  dd_mpi_equal_chunks
MPIDYN_OBJS = dd_functions.o dd_mpi_dynamic.o
MPIDYN_EXEC = dd_mpi_dynamic

# valid file suffixes 
.SUFFIXES: .c .o .cpp

# command to build .o files from .c files
.cpp.o: 
	$(CPP) -c $(CFLAGS) $<

all: $(MPIEQ_EXEC) $(MPIDYN_EXEC)

dd_functions.cpp: dd_functions.h

$(MPIEQ_EXEC): $(MPIEQ_OBJS)
	$(CPP) $(MPIEQ_OBJS) -o $(MPIEQ_EXEC)

$(MPIDYN_EXEC): $(MPIDYN_OBJS)
	$(CPP) $(MPIDYN_OBJS) -o $(MPIDYN_EXEC)

clean:
	rm -f *.o $(MPIEQ_EXEC) $(MPIDYN_EXEC) *~ *# a.out
