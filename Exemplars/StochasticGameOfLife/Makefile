CC = gcc
CXX=g++
OMP=-fopenmp
INCLUDE=/usr/local/include/trng
LIB=trng4
OPTFLAGS = -O3

OBJ_DIR = object_files

OBJS= $(OBJ_DIR)/getCommandLine.o $(OBJ_DIR)/display.o $(OBJ_DIR)/gnuplot_i.o 

COMMON_SRC_FILES = grid_common.cpp gol_rules.cpp
COMMON_HEADER_FILES = grid_common.hpp gol_rules.hpp initGrid.hpp calcNewGrid.hpp

# Directories containing source files specific to the random number generation method
GRID_SEQ_CPPRAND_DIR = grid_seq_rand_cpp
GRID_OMP_CPPRAND_DIR = grid_omp_rand_cpp
GRID_SEQ_TRNG_DIR = grid_seq_rand_trng
GRID_OMP_TRNG_DIR = grid_omp_rand_trng

# source files specific to the random number generation method
CPPRAND_SEQ_SRC_FILES = $(GRID_SEQ_CPPRAND_DIR)/cpprandom_calcNewGrid.cpp $(GRID_SEQ_CPPRAND_DIR)/cpprandom_initGrid.cpp
CPPRAND_OMP_SRC_FILES = $(GRID_OMP_CPPRAND_DIR)/cpprandom_calcNewGrid.cpp $(GRID_OMP_CPPRAND_DIR)/cpprandom_initGrid.cpp
TRNG_SEQ_SRC_FILES = $(GRID_SEQ_TRNG_DIR)/trng_calcNewGrid.cpp $(GRID_SEQ_TRNG_DIR)/trng_initGrid.cpp
TRNG_OMP_SRC_FILES = $(GRID_OMP_TRNG_DIR)/trng_calcNewGrid.cpp $(GRID_OMP_TRNG_DIR)/trng_initGrid.cpp

CPPRAND_SEQ_SRC = gol_seq.cpp $(COMMON_SRC_FILES) $(CPPRAND_SEQ_SRC_FILES)
CPPRAND_OMP_SRC = gol_omp.cpp $(COMMON_SRC_FILES) $(CPPRAND_OMP_SRC_FILES)
TRNG_SEQ_SRC = gol_seq.cpp $(COMMON_SRC_FILES) $(TRNG_SEQ_SRC_FILES)
TRNG_OMP_SRC = gol_omp.cpp $(COMMON_SRC_FILES) $(TRNG_OMP_SRC_FILES)

# profiling options
PROF_OPTS = -pg -g -O0

# These use randomness when creating a starting grid
PROGS =  cpprand_gol_omp cpprand_gol_seq cpprand_gol_seq_prof
# Initially, these do not use trng- it's an excresise to add it
PROGS_TRNG =  trng_gol_omp trng_gol_seq
# These use randomness when deciding on the hardiness of a live cell
STOCHASTIC_PROGS = cpprand_stgol_omp
STOCHASTIC_PROGS_TRNG = trng_stgol_omp

all: $(PROGS) $(STOCHASTIC_PROGS) $(PROGS_TRNG) $(STOCHASTIC_PROGS_TRNG)

#################### classic rules versions
## sequential w/c++ random nums for initialization
cpprand_gol_seq: $(CPPRAND_SEQ_SRC) $(COMMON_HEADER_FILES) $(OBJS) 
	$(CXX) ${OPTFLAGS} -o $@ $(OBJS) $(CPPRAND_SEQ_SRC) ${OMP}

## profiling version of sequential w/c++ random nums
cpprand_gol_seq_prof: $(CPPRAND_SEQ_SRC) $(COMMON_HEADER_FILES) $(OBJS) 
	$(CXX) $(PROF_OPTS) -o $@ $(OBJS) $(CPPRAND_SEQ_SRC) ${OMP}

# OpenMP w/cc++ random nums for classic rules version
cpprand_gol_omp: $(CPPRAND_OMP_SRC) $(COMMON_HEADER_FILES) $(OBJS) 
	$(CXX) ${OPTFLAGS} -o $@ $(OBJS) $(CPPRAND_OMP_SRC) ${OMP}

# Sequential w/ trng random nums for initialization as an exercise
trng_gol_seq: $(TRNG_SEQ_SRC) $(COMMON_HEADER_FILES) $(OBJS) 
	$(CXX) ${OPTFLAGS} -o $@ $(OBJS) $(TRNG_SEQ_SRC) ${OMP} -I$(INCLUDE) -l$(LIB)

# OpenMP w/ trng random nums for initialization as an exercise
trng_gol_omp: $(TRNG_OMP_SRC) $(COMMON_HEADER_FILES) $(OBJS) 
	$(CXX) ${OPTFLAGS} -o $@ $(OBJS) $(TRNG_OMP_SRC) ${OMP} -I$(INCLUDE) -l$(LIB)

##################################### Stchastic versions
cpprand_stgol_omp: $(CPPRAND_OMP_SRC) $(COMMON_HEADER_FILES) $(OBJS) 
	$(CXX) ${OPTFLAGS} -DSTOCHASTIC -o $@ $(OBJS) $(CPPRAND_OMP_SRC) ${OMP}

# OpenMP w/ trng random nums for initialization as an exercise
trng_stgol_omp: $(TRNG_OMP_SRC) $(COMMON_HEADER_FILES) $(OBJS) 
	$(CXX) ${OPTFLAGS} -DSTOCHASTIC -o $@ $(OBJS) $(TRNG_OMP_SRC) ${OMP} -I$(INCLUDE) -l$(LIB)

####################### Object files
# Utility functions are compiled into object files
# For command line arguments
$(OBJ_DIR)/getCommandLine.o: command_line/getCommandLine.cpp command_line/getCommandLine.hpp
	$(CC) -c -o $@ $<
# For graphics display based on gnuplot
$(OBJ_DIR)/display.o: graphics/display.c graphics/display.h
	$(CC) -c -o $@ $<
# For low-level gnuplot interface
$(OBJ_DIR)/gnuplot_i.o: graphics/gnuplot_i.c graphics/gnuplot_i.h
	$(CC) -Wno-format-overflow -Wno-error=format-overflow -c -o $@ $<


######### clean
clean:
		rm -f $(OBJS) $(PROGS) $(STOCHASTIC_PROGS) $(PROGS_TRNG) $(STOCHASTIC_PROGS_TRNG)

clean_movie:
		rm -f  intermediate.dat movie_images/*.dat movie_images/*.png movie_images/*.mp4

clean_data:
	rm -f initial.dat final.dat result.png

clean_profile:
	rm -f gmon.out profile.txt trimmed_profile.txt

clean_all: clean clean_movie clean_data clean_profile
