// functions in getCommandLine.c file
void getArguments(int argc, char *argv[], int * nx, int * ny, 
                  int * steps, int * display, int * verbose);
int isNumber(char s[]);
void exitWithError(char cmdFlag, char ** argv);
void Usage(char *program);