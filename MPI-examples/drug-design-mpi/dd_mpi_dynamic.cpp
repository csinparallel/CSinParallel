/*
 * dd_mpi_dynamic.cpp
 * Libby Shoop
 *
 * Contrived example of ligand-protein matching in drug design.
 * In this example, the work of testing a set of potential ligand 
 * strings to a protein string is handed out dynamically among
 * each worker process: each worker is given one ligand at a time
 * by Process 0, the conductor (or coordinator, if you prefer).
 * After receiving a score from a worker, the conductor sends another
 * ligand to that worker for scoring.
 *
 * For usage of all the command line options, type:
 *        mpirun -np 2 ./dd_mpi_dynamic -h
 * or
 *        mpirun -np 2 ./dd_mpi_dynamic --help
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

#define WORKTAG 1
#define DIETAG 2
#define SCORETAG 3


int main(int argc, char ** argv) {
  
  // initial default conditions and variable definitions
  bool verbose = false;       //C++ has bool type
  int nLigands = 120;         // number of ligands to match to protein
  int maxLigand = 5;          // maximum ligand length
  std::string protein = "How razorback-jumping frogs can level six piqued gymnasts";
  bool help = false;          // user chose help on command line
  bool problem = false;       // problem with chosen arguments

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

  // checks for poor choices of input values
  if (nLigands < numworkers) {
    if (taskid == 0) {
      printf("Please choose the number of ligands with -n \n");
      printf("to be greater than the number of workers (-np value - 1).\n");
      printf("Quitting.\n");
    }
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
  // 2. hand out a ligand to each worker process to start
  // 3. while there are ligands left,
  //    receive score result from a worker and send next ligand to it
  // 4. when work is done, tell each worker to stop and
  //    report max scoring ligands
  //
  if (taskid == 0) {
    std::vector<std::string> ligandList;
    ligandList = genLigandList(nLigands, maxLigand);       // 1.
    
    int ligLen;             // length of next ligand sent or
                            //  whose score was received

    if (verbose) {
      printf("ligands to try: "); 
      for (std::string s: ligandList) {
	printf("%s, ", s.c_str());
      }
      printf("\n------------------\n");
    }
 
    std::vector<std::string> maxScoringLigandList;
    int maxScore = 0;

    int workCount = 0;     //count of ligands sent for scoring
    // holds index of ligand in ligandList that process was last given
    int currentLigandIndex[numtasks];
    
    for (int i=0; i < numtasks; i++) {
      currentLigandIndex[i] = 0;
    }

    if (verbose)
      printf("Conductor sending out first single tasks to each worker.\n");
    
    // 2. give one ligand to each worker
    for (int i=0; i < numworkers; i++) {
      if (verbose) {
	printf("conductor sends %s to worker %d\n",
	       ligandList[i].c_str(), i+1);
      }
      ligLen = ligandList[i].length();
      // send ligand as C string
      MPI_Send(ligandList[i].c_str(), // The string
	       ligLen+1,   // its length, account for trailing \0
	       MPI_CHAR,   // its type
	       i+1,        // worker id to send to
	       WORKTAG,          // tag for ligand send
	       MPI_COMM_WORLD);  //communicator
      
      currentLigandIndex[i+1] = workCount; // set first ligand worker is on
      workCount ++;

    }
    
    // prepare for getting scores
    int recvCount = 0;      // num scores received
    int currentWorker;      // which worker recevied from
    int workerLigandIndex;  // index into ligandList for ligand scored
    int nextScore = 0;      // score received from a worker
    std::string nextLigand; // next ligand received or to send
    
    // 3. recieve a score and give out one ligand at a time until
    // all work has been sent.
    while (workCount < nLigands) {
      // receive a score
      
      recvCount++;
      MPI_Recv(&nextScore, 1, MPI_INT,
	       MPI_ANY_SOURCE,            // from any worker
	       SCORETAG,                  // tag for scores
	       MPI_COMM_WORLD, &status);
#ifdef DEBUG
      printf("conductor recv score %d from task %d\n",
	     nextScore, status.MPI_SOURCE);
#endif
      
      // which worker?
      currentWorker = status.MPI_SOURCE;
      
      // what was worker's last ligand list index?
      workerLigandIndex = currentLigandIndex[currentWorker];
      
      // get that ligand from the list
      nextLigand = ligandList[workerLigandIndex];
      ligLen = nextLigand.length();


#ifdef DEBUG
      printf("Ligand for that score: %s, count %d\n",
	     nextLigand.c_str(),
	     recvCount);
#endif
      
      // handle whether a max score
      updateMaxScore(nextScore,
		     &maxScore,
		     ligandList,
		     maxScoringLigandList,
		     workerLigandIndex);
#ifdef DEBUG
      printf("current max score : %d from ligands:", maxScore);
      printLigandList(maxScoringLigandList);
#endif

      // set next ligand worker is on
      currentLigandIndex[currentWorker] = workCount;
      nextLigand = ligandList[workCount];
      ligLen = nextLigand.length();

      if (verbose) {
	printf("conductor sends %s to worker %d\n",
	       nextLigand.c_str(), currentWorker);
      }

      // send that next ligand to the worker
      MPI_Send(nextLigand.c_str(), // the string
	       ligLen+1,   // its length, account for trailing \0
	       MPI_CHAR,   // its type
	       currentWorker,    // worker id to send to
	       WORKTAG,          // tag for ligand send
	       MPI_COMM_WORLD);  //communicator
      

      workCount++;
    }

      

    // receive the last of the work handed out
    while (recvCount < nLigands) {
      recvCount++;
      MPI_Recv(&nextScore, 1, MPI_INT,
	       MPI_ANY_SOURCE,            // from any worker
	       SCORETAG,                  // tag for scores
	       MPI_COMM_WORLD, &status);

      if (verbose) {
	printf("conductor recv score %d from task %d\n",
	       nextScore, status.MPI_SOURCE);
      }
      
      // which worker?
      currentWorker = status.MPI_SOURCE;
      
      // what was worker's last ligand list index?
      workerLigandIndex = currentLigandIndex[currentWorker];
      
      // get that ligand from the list
      nextLigand = ligandList[workerLigandIndex];
      ligLen = nextLigand.length();

#ifdef DEBUG
      printf("Ligand for that score: %s, count %d\n",
	     nextLigand.c_str(), recvCount);
#endif
      
      // handle whether a max score
      updateMaxScore(nextScore,
		     &maxScore,
		     ligandList,
		     maxScoringLigandList,
		     workerLigandIndex);
#ifdef DEBUG
      printf("current max score : %d from ligands:", maxScore);
      printLigandList(maxScoringLigandList);
#endif

    }

    int dieMsg = 1;
    
    // 4.
    // send separate 'die' message to each process using a special tag
    for (int id=1; id <= numworkers; id++) {
      MPI_Send(&dieMsg, 1, MPI_INT, id, // one int to task id
	       DIETAG,                  // tag for dying
	       MPI_COMM_WORLD);

    }

    finishTime = MPI_Wtime();
    printf("Total running time of process %d : %f seconds\n",
	   taskid, finishTime - startTime);
    // report results
    printf("Final max score : %d\nAchieved by ligand(s): ", maxScore);
    printLigandList(maxScoringLigandList);

    //////////////////////////////////////////////////////
  } else {  // worker
    ///////////////////////////////////////////////////////

    int myLast = 0; // small 'message' indicating no more work
    bool done = false;
    int my_nextLen; // length of next ligand string message
    int myCount = 0;    // accumulator for ligand counter
    char my_nextStr[maxLigand+1];   // for each ligand message
    int my_nextScore = 0;
    
    
    // keep getting next message, regardless of tag
    while (!done) {
    
      // We need to know what type of message is coming, so use probe.
      // Process 0 is conductor,
      // either tag 1 for next ligand string message
      // or tag for the 'die' message
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      // if tag from probe is die, recv and exit, otherwise recv next ligand
      if (status.MPI_TAG == DIETAG) {
	MPI_Recv(&myLast,   // no more work
		 1,   // its length
		 MPI_INT,     // type
		 0,            // recv from conductor
		 DIETAG,            // tag
		 MPI_COMM_WORLD, &status);

	// TODO as an exercise, send back how many this worker did

	if (verbose) printf("worker %d stopping.\n", taskid);
	
	done = true;   // get out of while loop
      
      } else {
	myCount++;
	// get a ligand to work on
	// first check the length of the string in the message
	MPI_Get_count(&status, MPI_CHAR, &my_nextLen);

	// Receive the next ligand C string
	MPI_Recv(my_nextStr,   // next ligand string
		 my_nextLen,   // its length
		 MPI_CHAR,     // type
		 0,            // recv from conductor
		 WORKTAG,      // tag
		 MPI_COMM_WORLD, &status);

	// create a C++ string and score it
	std::string my_nextLigand(my_nextStr);
	my_nextScore = score(my_nextLigand, protein);
	
	if (verbose) {
	      printf("worker %d scored ligand %s as %d\n",
		     taskid, my_nextLigand.c_str(), my_nextScore);
	}
	MPI_Send(&my_nextScore, 1, MPI_INT, 0, // one int to task 0
		 SCORETAG,                            // tag for scores
		 MPI_COMM_WORLD);

      }
    } // end while
    finishTime = MPI_Wtime();
    printf("Total running time of process %d : %f seconds\n",
	   taskid, finishTime - startTime);

  }  //////// end worker

  MPI_Finalize();
  return(0);

}



