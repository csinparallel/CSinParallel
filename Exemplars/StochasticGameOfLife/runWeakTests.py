import subprocess
"""
Run a series of 3 tests for the Stochastic Game of Life simulation.
initial width and length provided as parameters and adjusts after each execution. 
The function prints a blank line between tests for clarity.
    initial_width (int): The initial width of the simulation grid.
    initial_length (int): The initial length of the simulation grid.

The series of tests that is run starts with a width and length provided as parameters. 
Then the size of the grid is doubled for the next test. If the width and length are equal,
the width and height are adjusted to maintain a square grid. The number of threads (t_value) 
is doubled.

This is done 3 times, where the first test is the original size, the second test is double
the size, and the third test is double the size again.
The function prints the parameters used for each execution and a blank line between 
the series of tests.
"""

def run_tests(initial_width, initial_length, t_value):
    for test in range(3):
        run_commands(initial_width, initial_length, t_value)
        # Reset width, length, and t_value for the next test
        if initial_length == initial_width:
            initial_width = int(sqrt(initial_width * initial_length * 2))
            initial_length = initial_width
        else:
            initial_width *= 2
            initial_length = initial_length

        t_value = 2
        print()  # Print a blank line between tests

## LS Note: I asked ChatGPT to write this documentation for me. What needs work?
"""
Run a series of commands to execute a parallelized version of the Stochastic Game of Life simulation.
This function executes a command multiple times, varying the dimensions of the simulation grid 
(width and length) and the number of threads (t_value) used for execution. It starts with the 
initial width and length provided as parameters and doubles the number of threads after each 
execution. The dimensions are adjusted based on whether they are equal or not, and the function 
prints the parameters used for each execution.
Parameters:
    width (int): The initial width of the simulation grid.
    length (int): The initial length of the simulation grid.
    t_value (int): The initial number of threads to use for the simulation.
The function will run the command four times, adjusting the parameters after each run, and 
printing the results in a tab-separated format.
"""
import sys
from math import sqrt

def run_commands(width, length, t_value):
    # Loop to run the command 4 times, 2, 4, 8, 16 threads
    for i in range(4):
        print(width, length, t_value, sep='\t', end='\t', flush=True)

        # Construct the command
        command = f"./trng_stgol_omp -i 200 -w {width} -l {length} -t {t_value} -e"
        
        # Execute the command
        process = subprocess.Popen(command, shell=True)
        process.wait()
        
        # Update width and length
        if width == length:
            width = int(sqrt(width * length * 2))
            length = width
        else:
            length *= 2
        
        # double number of threads
        t_value *= 2

        print()

def main():
    
    # Check if the correct number of arguments is provided: length and width
    if len(sys.argv) != 3:
        print("Usage: python3 runWeakTest.py <length> <width>")
        sys.exit(1)

    # Get initial parameters from command line arguments
    length = int(sys.argv[1])
    width = int(sys.argv[2])

    initial_length = length
    initial_width = width
    t_value = 2  # Starting t value
    print("Width\tLength\tThreads\tTime")

    run_tests(initial_width, initial_length, t_value)

if __name__ == "__main__":
    main()


