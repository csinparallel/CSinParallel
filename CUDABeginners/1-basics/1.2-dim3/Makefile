CC = nvcc
ARCH = -arch=native   # detects the architecture of your GPU(s)

PROGS = dim3Demo dim3Demo1D1D dim3Demo1D2D dim3Demo2D2D

all: $(PROGS)


dim3Demo: dim3Demo.cu
	$(CC) $(ARCH) -o dim3Demo dim3Demo.cu

dim3Demo1D1D: dim3Demo1D1D.cu
	$(CC) $(ARCH) -o dim3Demo1D1D dim3Demo1D1D.cu

dim3Demo1D2D: dim3Demo1D2D.cu
	$(CC) $(ARCH) -o dim3Demo1D2D dim3Demo1D2D.cu

dim3Demo2D2D: dim3Demo2D2D.cu
	$(CC) $(ARCH) -o dim3Demo2D2D dim3Demo2D2D.cu

clean:
	rm -f $(PROGS)
