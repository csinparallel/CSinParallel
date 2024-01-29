MPI_CC  = mpicc
CC      = gcc
#CFLAGS  = -ansi -pedantic -std=c99 
CFLAGS = -std=gnu99 
WFLAGS = -Wno-format-overflow -Wno-unused-result
# -Wimplicit-function-declaration  -Wno-maybe-unitialized
# source files for larger dynamically sized plates
SRCS1      = mpi_heat2D.c display.c gnuplot_i.c getCommandLine.c
# object files from source files
OBJS1      = $(SRCS1:.c=.o)

# source files for small hard-coded size of plate and number of iterations
##SRCS2      = mpi_heat2D_big.c display.c gnuplot_i.c getCommandLine.c
# object files from source files
OBJS2      = $(SRCS2:.c=.o)

PROGS = mpi_heat2D 

all: mpi_heat2D 

# valid file suffixes 
.SUFFIXES: .c .o .cpp

# command to build .o files from .c files
.c.o: 
	$(MPI_CC) -c $(CFLAGS) $(WFLAGS) $<


mpi_heat2D: $(OBJS1)
	$(MPI_CC) $(CFLAGS) -o mpi_heat2D $(OBJS1) -lm

# mpi_heat2D_big: $(OBJS2)
# 	$(MPI_CC) $(CFLAGS) -o mpi_heat2D_big $(OBJS2)


clean:
	/bin/rm -f *.o $(PROGS) initial.dat final.dat *~ *# a.out