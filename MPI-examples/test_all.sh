cd heat2D
make clean
make
shelltest -c heat2D.test
make clean
cd ../monteCarloPi/MPI
# make and make clean are part of the test files for this
shelltest -c calcPiMPI.test
cd ../Seq
# make and make clean are part of the test files for this
shelltest -c calcPiSeq.test
cd ../..
