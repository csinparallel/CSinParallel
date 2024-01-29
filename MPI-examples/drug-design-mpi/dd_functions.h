// These are the functions in dd_fucntions.cpp that are used
// by the drug design programs

std::vector<std::string> genLigandList(int nLigands, int mxLigand);

bool getCommandLineArgs(int argc, char **argv,
			int * nLigands, int * maxLigand,
		        std::string * protein, bool * verbose
			);
void usage(char *program);

int score(std::string ligand, std::string protein);


void updateMaxScore(int nextScore,
		    int * maxScore,
		    std::vector<std::string> ligandList,
		    std::vector<std::string>& maxScoringLigandList,
		    int nextListOffset);

void printLigandList(std::vector<std::string> ligandList);
