void getArguments(int argc, char *argv[],
                  int * numThreads, int * N, int * useConstantSeed, int * doleOut);

int isNumber(char s[]);
void exitWithError(char cmdFlag, char ** argv);
void invalidChoice(char cmdFlag, char ** argv);
void Usage(char *program);
