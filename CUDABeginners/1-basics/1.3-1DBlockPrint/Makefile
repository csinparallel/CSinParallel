CC = nvcc
ARCH = -arch=native   # detects the architecture of your GPU(s)


PROGS = print1Block print2Blocks

all: $(PROGS)

print1Block: print1Block.cu
	$(CC) $(ARCH) -o print1Block print1Block.cu

print2Blocks: print2Blocks.cu
	$(CC) $(ARCH) -o print2Blocks print2Blocks.cu

clean:
	rm -f $(PROGS)













