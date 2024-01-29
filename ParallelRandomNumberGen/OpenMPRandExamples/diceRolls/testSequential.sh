#!/bin/sh

echo "===== Testing command line arguments for diceSim_omp ===="	# This is a comment, too!

echo "NOTE: This test should be used on original version with pragmas and call to jump commented out."
echo " -n 5 -c -v should indicate 5 rolls repeated of the default number of dice."
echo " We run this one 2 times to verify same output with 1 thread"
./diceSim_omp -n 5 -c -v -t 1
./diceSim_omp -n 5 -c -v -t 1


