#!/bin/sh
# This is a comment!
echo "===== Testing command line arguments for diceSim_omp ===="	# This is a comment, too!
echo " -h should produce a usage and exit."
./diceSim_omp -h

echo " -n should indicate missing argument."
./diceSim_omp -n

echo " -d should indicate missing argument."
./diceSim_omp -d

echo " -s should fail as an unrecognized argument."
./diceSim_omp -s
