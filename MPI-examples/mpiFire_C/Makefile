

# names of the binaries
ONEFIRE = oneFire
SEQSIM = seqFireSim
MPISIM = mpiFireSim

PROGS = $(ONEFIRE) $(SEQSIM) $(MPISIM)

# source files for single fire
SRCS1      = fire_functions.c oneFire.c fireSimPlot.c gnuplot_i.c 
# object files from source files
OBJS1      = $(SRCS1:.c=.o)

# source files for sequential fire simulation
SRCS      = fire_functions.c seqFireSim.c fireSimPlot.c gnuplot_i.c 
# object files from source files
OBJS      = $(SRCS:.c=.o)

# source files for mpi version of fire simulation
SRCSM      = fire_functions.c mpiFireSim.c fireSimPlot.c gnuplot_i.c 
# object files from source files
OBJSM      = $(SRCSM:.c=.o)

# which compiler to use
CC        = mpicc 

# compiler flags for warnings that can be ignored
CFLAGS = -Wno-unused-result -Wno-format-overflow -Wno-maybe-unitialized


# valid file suffixes 
.SUFFIXES: .c .o .cpp

# command to build .o files from .c files
.c.o: 
	$(CC) -c $(CFLAGS) $<

all: $(SEQSIM) $(ONEFIRE) $(MPISIM)

# commands to link object files into binaries
$(SEQSIM): $(OBJS)
	$(CC) $(OBJS) -o $(SEQSIM)

$(ONEFIRE): $(OBJS1)
	$(CC) $(OBJS1) -o oneFire

$(MPISIM): $(OBJSM)
	$(CC) $(OBJSM) -o mpiFireSim

# other dependencies
fire_functions.o: fire_functions.h
fireSimPlot.o: fireSimPlot.h
gnuplot_i.o:  gnuplot_i.h


clean:
	/bin/rm -f *.o $(PROGS) tmpout.dat *~ *# a.out

