CC = nvcc
ARCH = -arch=native   # detects the architecture of your GPU(s)

PROGS = printThreads-to-2DArray

all: $(PROGS)

printThreads-to-2DArray: printThreads-to-2DArray.cu
	nvcc $(ARCH) -o printThreads-to-2DArray printThreads-to-2DArray.cu

clean:
	rm -f $(PROGS)
