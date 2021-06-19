# Drug design exemplar code for various platforms.  


The Drug Design Exemplar CSinParallel module https://csinparallel.org/csinparallel/modules/drugDesign.html considers the application of scoring candidate drug "ligands" against a protein, using a simple representative scoring computation.  Implementations are provided for numerous PDC platforms based on a map-reduce structural pattern.  The range of platforms enables this module to be incorporated in a wide variety of course or extracurricular contexts.  

## Material designed for exploration

This repository contains the collection of code examples. We present the problem and the code implementations in [a CSinParallel descriptive module](http://selkie.macalester.edu/csinparallel/modules/DrugDesignInParallel/build/html/index.html). This can be used with the code provided here to enable a dive into various parallel implementations and explains some of the parallel patterns used in the code.

**Note:** The dd_omp.cpp version here does not use threaded building block (TBB) thread-safe collection classes as described in the above-linked descriptive module.

 Even without students examining the code itself, we have found them useful as "black boxes" for exploring the nature of PDC computation in elementary courses and in workshop presentations, where participants encounter issues such as speedup vs. number of processing elements, the effects of variable length cpu-intensive computational tasks, and scheduling of parallel tasks.  

## Compiling the examples

A simple example Makefile is provided for compiling the various code versions (where appropriate- the golang, or go version can simply be run). 

Simply typing

    make

should build each one, if you have g++ and mpiCC installed on a linux system.

## Running the examples

Here we describe some typical ways that we have run these code examples to explore how the code scales when running in parallel using multiple threads or processes.

### Sequential C++ version

The serial, or sequential version that does not use parallel threads can be run like this:

    ./dd_serial 7 120

This is for a maximum ligand length of 7 and 120 ligands generated to search against the protein. Be patient while this takes around 30 seconds to run on most servers. This is a baseline time that parallelism will improve.

### OpenMP versions

With OpenMP, we are using shared memory and creating multiple **threads** to work on a subset of the ligands. In one version of the code in the file dd_omp_dynamic.cpp, dynamic scheduling is used to have each thread work on one ligand at a time, starting on a new one as soon as finishes. This is set up by this OpenMP pragma ahead of the loop that works on each ligand:

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)

In a second version of the code in the file dd_omp_static.cpp, static scheduling is used by default because the schedule clause is not used. his assigns an equal number of ligands per thread.

In these code examples, the time to execute the Map and Reduce portions of the code have been determined and printed. When running the following examples, note how the dynamic scheduling improves the time.

First, maximum ligand length of 7 and 120 ligands total, static vs. dynamic:

    ./dd_omp_static 7 120 2
    ./dd_omp_dynamic 7 120 2
    ./dd_omp_static 7 120 3
    ./dd_omp_dynamic 7 120 3
    ./dd_omp_static 7 120 4
    ./dd_omp_dynamic 7 120 4
    ./dd_omp_static 7 120 8
    ./dd_omp_dynamic 7 120 8
    ./dd_omp_static 7 120 16
    ./dd_omp_dynamic 7 120 16
    ./dd_omp_static 7 120 32
    ./dd_omp_dynamic 7 120 32

Do you see any improvement from 16 to 32 threads in either case?

Note that on a server with more cores and a faster processor we have been able to try a maximum ligand length of 7 and we can also increase how many ligands we generate, like this, using dynamic scheduling:

    ./dd_omp_dynamic 7 240 4
    ./dd_omp_dynamic 7 240 8
    ./dd_omp_dynamic 7 240 16

### C++11 threads version

This version in the file dd_threads.cpp uses C++11 threads using a "thread pool" parallel pattern. It is inherently computing each ligand match in a dynamic manner on each thread in the "pool", or group of threads that get created for the program to use.

You can compare this to the dd_omp_dynamic version to see that the times are comparable:

    ./dd_threads 7 120 4
    ./dd_omp_dynamic 7 120 4

### Go version

The go version currently defaults to a max ligand size of 7, 120 ligands, and 1 CPU core. Run the default sequential version like this:

    go run dd_go.go

Be patient while one CPU chugs through the 120 ligands. Run fewer ligands like this:

    go run dd_go.go -n 60

To use more cores and the default 120 ligands, try these examples and note how the time decreases.

    go run dd_go.go -cpu 2
    go run dd_go.go -cpu 4
    go run dd_go.go -cpu 2
    go run dd_go.go -cpu 8
    go run dd_go.go -cpu 16
    go run dd_go.go -cpu 32

The Go version is also inherently computing each ligand match in a dynamic manner on each thread. How it is creating the ligands is using a different random number generator, so the ligands being used are different. It is still quite interesting to compare this version's time to the C++ OpenMP dynamic version. Try this:

    go run dd_go.go -cpu 4
    ./dd_omp_dynamic 7 120 4

You can also try using the -len flag to change the maximum ligand length. Recall that the time to find a match increases quite a bit as the length of the ligand increases.

## C++ MPI version

We also have a version that uses multiple **processes** rather than threads and uses message passing as the way that processes communicate. This is written using the MPI library and requires that you have a version of MPI installed on your system. Though often though of for use in distributed clusters of machines, MPI can be used on shared memory machines also.

