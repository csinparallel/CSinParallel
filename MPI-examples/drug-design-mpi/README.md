# Dynamic vs. Equal Loads with the drug design example
Libby Shoop, Macalester College

This folder contains two separate distributed versions of the contrived drug design example. In one case the work is distributed equally among worker processes (dd_mpi_equal_chunks.cpp) and in another case each worker process gets its work one at a time after it finishes one task (dd_mpi_dynamic.cpp). 

When the time it takes each process to finish one unit of work (in this case matching a ligand string to a protein) may vary quite a bit from case to case, it is very often the case that dynamic assignment of the work can perform better. You can run this code to observe whether that is true here.

To make this code a bit easier to read (and in truth easier to write), it was written using features of C++, in particular vectors and strings. If you are more familiar with C than C++, you still should be able to follow it.

## Run slightly differently than on Pi clusters

You can use the instructions for running the code found in (The online book for using MPI on the Raspberry Pi clusters, Ch. 8)[https://www.learnpdc.org/RaspberryPi-mpi/09Exemplars/drug_design.html] as a guide for running the code on our department machine. However, **eliminate the -hostfile option** and run like this example:

        mpirun -np 4 ./dd_mpi_equal_chunks -n 18 --verbose

