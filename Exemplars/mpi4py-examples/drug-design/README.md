# Drug Design example using Python and Message Passing

This is a version of the drug design problem using Python and the mpi4py library.

## Running examples

With the combination of Python and message passing between processes, the times are typically slower than the C++ shared memory versions. However, we can get better results (more ligands) on a server class machine with many cores than on the 4-core Raspberry Pi.

There are two versions here, one that distributes work dynamically to each worker process, and one that distributes work equally between all processes (each worker gets the same number of ligands to match).

The default maximum ligand length in the code is 5. You can try 48 ligands using 1 master and 4 workers like this:

    mpirun -np 5 python dd_mpi_dynamic.py 48
    mpirun -np 5 python dd_mpi_equal_chunks.py 48

To observe scalability, you can try either program with a fixed number of ligands, varying the number of processes like this: 2 (1 worker), 3 (2 workers), 5 (4 workers), 9 (8 workers), 17 (16 workers). Try it with 60 ligands:

| -np | #ligands | # workers | equal chunks time | dynamic time |
|-----|----------|-----------|-------------------|--------------|
| 3   | 60       |     2     |                   |              |
| 5   | 60       |     4     |                   |              |
| 9   | 60       |     8     |                   |              |
| 17  | 60       |    16     |                   |              |

Ideally, as you double the number of workers, the time should be cut in half. This is called **strong scalability**. But there is some overhead from the message passing, so we don't see perfect strong scalability.

You could try the dynamic version with a larger number of ligands (80, 100, 120) and 4 or more workers (-np 5, 9, 17) to see how well it scales.