#!/bin/sh

echo "===== Testing command line arguments for diceSim_omp ===="	# This is a comment, too!

echo " -t 3 -n 12 -c should roll 2 dice 12 times using 3 threads."
./diceSim_omp -t 3 -n 12 -c -v
echo " -t 4 -n 12 -c should roll 2 dice 12 times using 4 threads with same result as above."
./diceSim_omp -t 4 -n 12 -c -v
