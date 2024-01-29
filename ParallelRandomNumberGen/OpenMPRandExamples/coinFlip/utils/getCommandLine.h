// functions in getCommandLine.c file
void getArgumentsSeq(int argc, char *argv[],
                  int * verbose, int * N, int * useConstantSeed);
void getArguments(int argc, char *argv[],
                  int * verbose, int * N, int *numThreads, int * useConstantSeed);
int isNumber(char s[]);
void exitWithError(char cmdFlag, char ** argv, int isPar);
void Usage();
void UsageSeq(char *program);
void UsagePar(char *program);

