void getArguments(int argc, char *argv[],
                  int * numThreads, int * N, int * useConstantSeed, int * doleOut);

int isNumber(char s[]);
void exitWithError(char cmdFlag, char ** argv);
void invalidChoice(char cmdFlag, char ** argv);
void Usage(char *program);

// 2D grid specific
void getArguments(int argc, char *argv[],
                  int * numThreads, int * w, int * l, int * useConstantSeed, int * doleOut);
void Usage2D(char *program);
