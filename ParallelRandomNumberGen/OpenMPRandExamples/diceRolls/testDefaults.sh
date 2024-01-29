#!/bin/sh

echo "===== Testing command line arguments for diceSim_omp ===="	# This is a comment, too!

echo " -v only should use default values and produce verbose output."
./diceSim_omp -v
echo " -n 5 -v should indicate 5 rolls of the default number of dice."
./diceSim_omp -n 5 -v

echo " -d 6 -n 3 -v should roll 6 dice 3 times."
./diceSim_omp -d 6 -n 3 -v


