
import sys
import subprocess
from math import sqrt

def run_test(width, length, threads):

        # Construct the command
        command = f"./trng_stgol_omp -i 200 -w {width} -l {length} -t {threads} -e"
        
        # Execute the command
        process = subprocess.Popen(command, shell=True)
        process.wait()

def main():

    # Check if the correct number of arguments is provided: 
    # length and width, number of speedup lines, number of trials
    if len(sys.argv) != 5:
        print("Usage: python3 runStrongTest.py <length> <width>")
        sys.exit(1)

    # Get initial parameters from command line arguments
    length = int(sys.argv[1])
    width = int(sys.argv[2])
    num_lines = int(sys.argv[3])
    num_trials = int(sys.argv[4])

    # This line is different than the shell scripts so that you know
    # what the width and length are at the start
    print(f"Initial Width: {width}, Initial Length: {length}")

    # print header
    print(f"Trial\t#th\t", end='', flush=True)
    sizes = [(0, 0)] * num_lines
    for i in range(num_lines):
        sizes[i] = (width, length)
        num_cells = width * length
        print(f"{num_cells}\t", end='\t', flush=True)

        if width == length:
            width = int(sqrt(width * length * 2))
            length = width
        else:
            length *= 2
    print()
    
    # run given number of trials for each number of threads and set of grid sizes
    threads_list = [1, 2, 4, 6, 8, 12, 16]
    for threads in threads_list:  
        for trial in range(num_trials):  
            print(f"{trial+1}\t{threads}\t", end='', flush=True)
            for size in sizes:
                run_test(size[0], size[1], threads)

            print()
        print()

if __name__ == "__main__":
    main()

