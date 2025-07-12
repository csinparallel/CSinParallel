// functions in getCommandLine.cpp file
void getArguments(int argc, char *argv[], int * w, int *l, int * iterations, 
                  int * debug, int *graphics, int *animation, int * movie, 
                  int *centerSpore, int *numThreads);
int isNumber(char s[]);
void exitWithError(char cmdFlag, char ** argv);
void Usage(char *program);