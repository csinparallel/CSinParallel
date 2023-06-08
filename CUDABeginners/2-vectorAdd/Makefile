CC = nvcc
ARCH = -arch=native   # detects the architecture of your GPU(s)


PROGS = vectorAdd 

all: $(PROGS)

vectorAdd: vectorAdd.cu 
	$(CC) $(ARCH) -o vectorAdd vectorAdd.cu

clean:
	rm -f $(PROGS)
