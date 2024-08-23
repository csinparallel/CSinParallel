void getArguments(int argc, char *argv[],
                  int * numThreads, int * timeSteps, int * numPlayers, int * useConstantSeed);

int isNumber(char s[]);
void exitWithError(char cmdFlag, char ** argv);
void invalidChoice(char cmdFlag, char ** argv);
void Usage(char *program);
