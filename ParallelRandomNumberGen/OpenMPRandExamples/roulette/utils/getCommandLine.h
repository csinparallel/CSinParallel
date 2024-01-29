// functions in getCommandLine.c file

// parallel version has one extra argument, numThreads
void getArguments(int argc, char *argv[],
                  int * N, 
                  int * numThreads, 
                  int * useConstantSeed);
// sequential version has one fewer argument than the parallel one
void getArguments(int argc, char *argv[],
                  int * N, 
                  int * useConstantSeed);

int isNumber(char s[]);
void exitWithError(char cmdFlag, char ** argv, int isPar);
void Usage(char *program, int isPar);
