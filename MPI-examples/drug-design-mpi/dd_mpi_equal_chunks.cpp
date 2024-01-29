/*
 * dd_mpi_equal_chunks.cpp
 * Libby Shoop
 *
 * Contrived example of ligand-protein matching in drug design.
 * In this example, the work of testing a set of potential ligand 
 * strings to a protein string is divided as equally as possible among
 * each worker process. Process 0, the conductor (or coordinator, if you
 * prefer) generates the ligand strings and sends them to the workers
 * for scoring.
 *
 * For usage of all the command line options, type:
 *        mpirun -np 2 ./dd_mpi_equal_chunks -h
 * or
 *        mpirun -np 2 ./dd_mpi_equal_chunks --help
 *
 * Note when running actual examples you should use more than two proceses.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cstring>

#include "dd_functions.h"

#include "mpi.h"

//#define DEBUG 1

int main(int argc, char ** argv) {
  
  // initial default conditions and variable definitions
  bool verbose = false;       //C++ has bool type
  int nLigands = 120;         // number of ligands to match to protein
  int maxLigand = 5;          // maximum ligand length
  std::string protein = "How razorback-jumping frogs can level six piqued gymnasts";
  bool help = false;

  // alternate protein strings you could try:
  // shorter takes less time
  // string protein = "Cwm fjord bank glyphs vext quiz";
  // more repetition of certain letters
  // string protein = "the cat in the hat wore the hat to the cat hat party";
  
  // new values for MPI
  int numtasks;   // total processes
  int taskid;     // id of task running a process
  int numworkers; // number of tasks that will report to conductor
  MPI_Status status;     // status of MPI send/recv

  /* First, find out my taskid and how many tasks are running */
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
  numworkers = numtasks-1;

  if (numworkers < 1) {
    printf("Please use at least two processes. Quitting.\n");
    MPI_Finalize();
    exit(1);
  }

  // check command line arguments
  // note  each process does this
  help = getCommandLineArgs(argc, argv,
			       &nLigands, &maxLigand,
			       &protein, &verbose);

  if (taskid == 0) { // conductor
    if (help) {
      usage(argv[0]);
    }

    if (verbose) {
      printf("Verbose output chosen.\n");
      printf("Values to be used:\n");
      printf("%d Ligands\n%d max Ligand length\nProtein: %s\n",
	   nLigands, maxLigand, protein.c_str());
    }
  }

  // If chose help on commnad line, then all processes stop.
  if (help) {
    MPI_Finalize();
    return(0);
  }
  ///////////////////// end handling command line

  // Special test case for illustration uses a max ligand length of 6,
  // so override the default or the command line.
  if (nLigands <= 18) {
    maxLigand = 6;
  }

  // for timing
  double startTime, finishTime;

  startTime = MPI_Wtime();
  
  ////////////////////////////////////////////////////////
  // conductor's job:
  // 1. make the ligands to be matched
  // 2. decide amount of work for each worker,
  // 3. send sublist of ligands to each worker,
  // 4. wait for computation results, keeping track of maximum score.
  /////////////////////////////////////////////////////////
  if (taskid == 0) {

    std::vector<std::string> ligandList;
    ligandList = genLigandList(nLigands, maxLigand);    // 1.

    if (verbose) {
      printf("ligands to try: "); 
      for (std::string s: ligandList) {
	printf("%s, ", s.c_str());
      }
      printf("\n------------------\n");
    }

    std::vector<std::string> maxScoringLigandList;
    int maxScore = 0;

    // 2.
    // conductor determines number of ligands for each worker process,
    // ensuring work for each is no more than 1 above any other--
    // this is the 'equal chunks' concept of dividing the work.
    //
    int trunc_num_nLigands = nLigands/numworkers;
    int extra = nLigands%numworkers;
    int nLigands_per_worker = 0;
    
    int start =0;  // start of 'slice' of vector of ligands to be sent
    int listStartPerWorker[numtasks];
    
    for (int id=1; id<=numworkers; id++) {
      // compute 'chunk' of work that is a most 1 more than another worker      
      if (id <= extra) {
	nLigands_per_worker = trunc_num_nLigands + 1;
      } else {
	nLigands_per_worker = trunc_num_nLigands;
      }
      // send number of ligands to worker id
      MPI_Send(&nLigands_per_worker, 1, MPI_INT, id, 1, MPI_COMM_WORLD);

      // 3.
      // Send the 'slice' of vector of ligands to the worker,
      // one string at a time as C strings (MPI can only send C char arrays)
      // first time through, start is 0 for first worker

      for (int i_ligand = 0; i_ligand < nLigands_per_worker; i_ligand++) {
	
	if (verbose) {
	  printf("conductor sends to worker %d ligand: %s\n", id, ligandList[start + i_ligand].c_str());
	}
	
	int len = ligandList[start + i_ligand].length();

	MPI_Send(ligandList[start + i_ligand].c_str(), // the string
		 len+1,      // its length
		 MPI_CHAR,   // its type
		 id,         // worker id to send to
		 2,          // tag for ligand send
		 MPI_COMM_WORLD);  //communicator
      }

      listStartPerWorker[id] = start;
      
      // ready the start for the next worker
      start = start + nLigands_per_worker;
    }

#ifdef DEBUG
    for (int id=1; id<=numworkers; id++) {
      printf("DEBUG task %d start in ligand list: %d\n", id, listStartPerWorker[id]);
    }
#endif
    
    // set up for receiving results
    int recvCount = 0;
    int nextScore =0;
    int listOffsetPerWorker[numtasks];
    int currentWorker;
    int nextListOffset;

    // initialize counter for scores completed per worker
    for (int id=0; id<=numworkers; id++) {
      listOffsetPerWorker[id] = 0;
    }

    // 4.
    // while there are more ligands to process,
    // conductor waits to get each score one at a time
    // and updates the maximum score found and from which ligands
    while (recvCount < nLigands) {
      MPI_Recv(&nextScore, 1, MPI_INT,
	     MPI_ANY_SOURCE,            // from any worker
	     3,                         // tag for scores
	     MPI_COMM_WORLD, &status);

      if (verbose) {
	printf("conductor recv score %d from task %d\n",
	       nextScore, status.MPI_SOURCE);
      }

      currentWorker = status.MPI_SOURCE;
      nextListOffset = listStartPerWorker[currentWorker] +
	               listOffsetPerWorker[currentWorker];

#ifdef DEBUG
      printf("DEBUG index %d corresponds to ligand %s\n",
	     nextListOffset, ligandList[nextListOffset].c_str());
#endif
      
      updateMaxScore(nextScore,
		     &maxScore,
		     ligandList,
		     maxScoringLigandList,
		     nextListOffset);

#ifdef DEBUG
      printf("DEBUG current max score : %d from ligands:", maxScore);
      printLigandList(maxScoringLigandList);
#endif
      
      // update counts before next message
      listOffsetPerWorker[currentWorker] ++;
      recvCount ++;
    }

    finishTime = MPI_Wtime();
    printf("Total running time of process %d : %f seconds\n",
	   taskid, finishTime - startTime);
    // report results
    printf("Final max score : %d\nAchieved by ligand(s): ", maxScore);
    printLigandList(maxScoringLigandList);

    
  } else {  // workers compute score for each ligand it receives
    ///////////////////////////////////////////////////////////////
    
    // receive count of ligands this worker will do from conductor
    // tag of this message is 1
    int my_nLigands = 0;
    MPI_Recv(&my_nLigands, 1, MPI_INT,
	     0,                         // from conductor
	     1,                         // tag
	     MPI_COMM_WORLD, &status);

    int my_nextLen; // length of next ligand string message
    int myCount;    // accumulator for ligand counter
    char my_nextStr[maxLigand+1];   // for each ligand message
    std::vector<std::string> my_ligandList; // list of strings for each ligand
    
    // get ligand C strings one at a time
    // MPI can only accomodate C style strings
    //
    for (myCount = 0; myCount < my_nLigands; myCount++) {
      strcpy(my_nextStr, ""); // clear out to reuse
      
      // We need to know how long the next string is, so use probe.
      // Process 0 is conductor, tag 2 for next ligand string message
      MPI_Probe(0, 2, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_CHAR, &my_nextLen);

      // Receive the next ligand C string
      MPI_Recv(my_nextStr,   // next ligand string
	       my_nextLen,   // its length
	       MPI_CHAR,     // type
	       0,            // recv from conductor
	       2,            // tag
	       MPI_COMM_WORLD, &status);

 #ifdef DEBUG
      printf("DEBUG worker %d next ligand received: %s\n", taskid, my_nextStr);
#endif
      
      // place the received string back onto C++ vector of C++ strings
      // use C++ string constructor for the received ligand string
      std::string my_nextLigand(my_nextStr);
      my_ligandList.push_back(my_nextLigand);

    }
    
    // once worker has its ligands to work on, score each one
    // and send results to conductor one at a time.
    int my_nextScore;
    for (std::string lig: my_ligandList) {
      my_nextScore = score(lig, protein);
      
      if (verbose) {
	printf("worker %d scores ligand %s as %d\n",
	       taskid, lig.c_str(), my_nextScore);
      }
      
      MPI_Send(&my_nextScore, 1, MPI_INT, 0, // one int to task 0
	       3,                            // tag for scores
	       MPI_COMM_WORLD);

      
    }
    // potential improvement: send a block of scores in an integer array

    finishTime = MPI_Wtime();
    printf("Total running time of process %d : %f seconds\n",
	   taskid, finishTime - startTime);

  }   ///////// end of worker
  
  ////////////////////////////// all done
  MPI_Finalize();
  return(0);

}
