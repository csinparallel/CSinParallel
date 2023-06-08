CC = nvcc
ARCH = -arch=native   # detects the architecture of your GPU(s)

PROGS = device_info 

all: $(PROGS)

device_info: device_info.cu 
	$(CC) -o device_info device_info.cu

clean:
	rm -f $(PROGS)
