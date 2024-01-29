#include "mpi.h"


#include <getopt.h>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <string>


void usage(char *program)
{
    std::cout <<
      "mpirun -np <np> " << program << " [options]\n where options are:\n" <<
            "--nLigands <-n>:        Number of ligands to test\n"
            "--maxLigandLength <n>: Ligands will be this length or less\n"
            "--protein <string>:    Protein string to match ligands to\n"
            "--verbose:             Display results as ligands get matched\n"
            "--help:                Show help\n";
}

// Returns true if -h or --help was requested on command line.
// Other optional arguments passed by reference and set if provided
// on the command line.
bool getCommandLineArgs(int argc, char **argv,
			int * nLigands, int * maxLigand,
		        std::string * protein,
			bool * verbose
		       )
{
  bool help = false; // default
  
  // Using the features of getopt
  const char* const short_opts = "n:m:p:hv";
  const option long_opts[] = {
	      {"nLigands", optional_argument, nullptr, 'n'},
	      {"maxLigandLength", optional_argument, nullptr, 'm'},
	      {"protein", optional_argument, nullptr, 'p'},
	      {"verbose", optional_argument, nullptr, 'v'},
	      {"help", 0, nullptr, 'h'},
	      {nullptr, 0, nullptr, 0}
  };

  while (true) {
    const auto opt = getopt_long(argc, argv,
				 short_opts, long_opts, nullptr);

    if (-1 == opt)
      break;

    switch (opt) {
    case 'n':
      *nLigands = std::stoi(optarg);
      break;

    case 'm':
      *maxLigand = std::stoi(optarg);
      break;

    case 'p':
      *protein = optarg;
      break;

    case 'v':
      *verbose = true;
      break;

    case 'h': // -h or --help
    case '?': // Unrecognized option
    default:
      //usage(argv[0]);
      help = true;
      break;
    }
  }
    
  return(help);
}

// Keep track of the maximum score and ligand that achieved it
//updateMaximum(score, lig, maxScore, maxScoreLigands):

// return an array of int values representing random lengths,
// following a gamma distribution, where alpha is 4.2 and beta is 0.8.
// This is so we get more shorter lengths than longer ones, simply
// to make the simulation run a bit faster and have uneven distibution
// of lengths.
// Aprroximately one in 10 is length 5, with an occasional length 6.
//
void genLigandLengths(int nLigands, int maxLength, int *ligandLengths) {
  std::default_random_engine generator;
  std::gamma_distribution<double> distribution(4.2, 0.8);

  for (int i = 0; i < nLigands; i++) {
    double number = distribution(generator);
    int len = (int) number;

    if (len < 2) {
      len = 2;
    } else if (len > maxLength) {
      len = maxLength;
    }

    ligandLengths[i] = len;
  }
  return;
}

// Generate and return a vector of lowercase strings, each string repesenting 
// a random length from an input array.
//      nLigands is the number of ligands to make
//      ligandLengths is an arrary of ints for the length of each ligand
//
std::vector<std::string> genRandomLigands(int nLigands,
					  int *ligandLengths){

  int ligLen;                         // length for each new ligand
  std::vector<std::string>ligandList; // list of strings for each ligand
  
  // loop to make nLigands strings
  for (int i=0; i<nLigands; i++) {

    // get length of next ligand
    ligLen = ligandLengths[i];
    // string for the next ligand
    std::string nextStr;
    nextStr = "";

    // each character generated
    char ch;

    // Set up C++ random distribution of ints from 0-25.
    // ASCII a is 1, and z is 26. 
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_int_distribution<int> dist(0, 25);
    
    // for each char over ligand length, create a random character
    // and append to string with random char.
    //
    for (int j=0; j<ligLen; j++) {
      // a + number from 0-25 will create random lowercase value
      ch = 'a' + dist(eng); 
      nextStr.push_back(ch);  // append to string
    }
    ligandList.push_back(nextStr);

  }
  return ligandList;
}

// If nLigands <=18, create and return a pre-determined set of example ligands.
// (A poor case scenario where the longest ligand is six characters
//  and is at the from of the list of ligands.)
//
// Otherwise, create and return a set of ligands whose length randomly varies.
//
std::vector<std::string> genLigandList(int nLigands, int maxLigand) {
  
  std::vector<std::string>ligandList; // to be returned
  
  // demonstrate a poor case scenario where the longest ligand is in
  // the front of the list
  if (nLigands <= 18) {
    ligandList =
      {"razvex", "qudgy", "afrs", "sst", "pgfht", "rt", "id", 
       "how", "aaddh",  "df", "os", "hid", 
       "sad", "fl", "rd", "edp", "dfgt", "spa"
      };

    if (nLigands < 18) {
      // remove the unneeded ones from the end of the list
      ligandList.erase(ligandList.begin()+nLigands, ligandList.end());
    }
  } else { // otherwise generate a random list of ligands
    int ligandLengths[nLigands];
    genLigandLengths(nLigands, maxLigand, ligandLengths);
    
    ligandList = genRandomLigands(nLigands, ligandLengths);
  }
  return ligandList;
  
}

// function score
//   recursive function that creates a score for how well a short ligand
//   string matches a longer protein string.
//
//   2 arguments:  a ligand and a protein sequence
//   return:  int, simulated binding score for ligand arg1 against protein arg2
//
int score(std::string ligand, std::string protein) {
  // bail out if zero-length strings
  if (ligand.length() == 0 || protein.length() == 0) {
    return 0;
  }

  if (ligand.substr(0,1) == protein.substr(0,1)) {
    return 1 + score(ligand.substr(1), protein.substr(1));
  } else {
    return std::max(score(ligand, protein.substr(1)),
	       score(ligand.substr(1), protein));
  }
}

// Keep track of the maximum score found and the ligands that
// have scored it.
// The variable maxScore is passed by reference so it can be updated
// The vector of max scoring ligands, maxScoringLigandList,
//    is also passed by refereence so it can be updated.

void updateMaxScore(int nextScore,
		    int * maxScore,
		    std::vector<std::string> ligandList,
		    std::vector<std::string>& maxScoringLigandList,
		    int nextListOffset) {
      // handle maximum score
      if (nextScore > *maxScore) {
	// clear current vector of highest scoring ligands
	maxScoringLigandList.clear();
	maxScoringLigandList.push_back(ligandList[nextListOffset]);
	*maxScore = nextScore;
      } else if (nextScore == *maxScore) {
	maxScoringLigandList.push_back(ligandList[nextListOffset]);
      }

      /*
      //debug
      printf("current max score : %d from ligands:", *maxScore);
      printLigandList(maxScoringLigandList);
      printf("\n");
      */

}

void printLigandList(std::vector<std::string> ligandList) {
  for (std::string s: ligandList) {
    printf("%s, ", s.c_str());
  }
  printf("\n");
}
