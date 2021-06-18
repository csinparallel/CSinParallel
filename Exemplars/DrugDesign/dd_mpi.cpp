#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <queue>
#include <vector>
#include <chrono>
#include <mpi.h>

#define DEFAULT_max_ligand 6
#define DEFAULT_nligands 120
#define DEFAULT_nthreads 4
#define DEFAULT_protein "the cat in the hat wore the hat to the cat hat party"

#define MAX_BUFF 100
#define VERBOSE 0  // non-zero for verbose output

struct Pair {
  int key;
  std::string val;
  
  Pair(int k, const std::string& v) : key(k), val(v) {}
};

class Help {
public:
  static std::string get_ligand(int max_ligand);
  static int score(const char*, const char*);
};

class MR {
private:
  enum MsgType {
    GET_TASK, // worker request for a fresh ligand to score
    TASK_RESULT, // worker delivery of a score for a ligand
    ACK // protocol acknowledgment message
  };
  
  int max_ligand;
  int nligands;
  int nnodes;
  int rank;
  static const int root = 0;
  std::string protein;
  
  std::queue<std::string> tasks;
  std::vector<Pair> results;
  
  void Generate_tasks(std::queue<std::string>& q);
  //void Map(const std::string& str, std::vector<Pair>& pairs);
  void Sort(std::vector<Pair>& vec);
  int Reduce(int key, const std::vector<Pair>& pairs, int index, 
	     std::string& values);
  
public:
  const std::vector<Pair>& run(int ml, int nl, const std::string& p);
};

int main(int argc, char **argv) {
  int max_ligand = DEFAULT_max_ligand;
  int nligands = DEFAULT_nligands;
  std::string protein = DEFAULT_protein;
  
  if (argc > 1)
    max_ligand = strtol(argv[1], NULL, 10);
  if (argc > 2)
    nligands = strtol(argv[2], NULL, 10);
  if (argc > 3)
    protein = argv[3];
  // command-line args parsed
  
  MPI_Init(&argc, &argv);
  
  MR map_reduce;
  std::vector<Pair> results = map_reduce.run(max_ligand, nligands, protein);
  
  if(results.size()) {
    std::cout << "maximal score is " << results[0].key 
	      << ", achieved by ligands " << std::endl 
	      << results[0].val << std::endl;
  }
  
  MPI_Finalize();
  
  return 0;
}

const std::vector<Pair>& MR::run(int ml, int nl, const std::string& p) {
  max_ligand = ml;
  nligands = nl;
  protein = p;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nnodes);
  
  char buff[MAX_BUFF];
  
  MPI_Status status;
  
  char empty = 0;
  
  if(rank == root) {
    // Only the root will generate the tasks
    Generate_tasks(tasks);
    
    // time the code on root
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

    // Keep track of which workers are working
    std::vector<int> finished;
    for(int i = 0; i < nnodes; ++i) {
      finished.push_back(0);
    }
    finished[root] = 1;  // master task does no scoring
    
    std::vector<Pair> pairs;
    
    // The root waits for the workers to be ready for processing
    // until all workers are done
    while([&](){ 
	for(auto i : finished) { if(!i) return 1; } 
	return 0; }()) {
      
      MPI_Recv(buff, MAX_BUFF, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, 
	       MPI_COMM_WORLD, &status);
      switch(status.MPI_TAG) {

      case GET_TASK:
	// Send the next task to be processed
	if(tasks.empty()) {
	  MPI_Send((void*)&empty, 1, MPI_CHAR, status.MPI_SOURCE, ACK, 
		   MPI_COMM_WORLD);
	  
	  // Mark the worker as finished
	  finished[status.MPI_SOURCE] = 1;
	} else {
	  MPI_Send((void*)tasks.front().c_str(), tasks.front().size() + 1, 
		   MPI_CHAR, status.MPI_SOURCE, ACK, MPI_COMM_WORLD);
	  tasks.pop();
	}
	break;

      case TASK_RESULT: {
	std::string buffstr(buff);
	std::stringstream stream(buffstr);
	std::string task;
	int score;
	
	stream >> task;
	stream >> score;
	pairs.push_back(Pair(score, task));
	if (VERBOSE) 
	  std::cout << rank << ": " << task << " --> " << score << 
	    " (received from " << status.MPI_SOURCE << ")" << std::endl;

      }
	break;

      default:
	break;
      }
    }
    
    // All tasks are done
    Sort(pairs);
    
    int next = 0;
    while(next < pairs.size()) {
      std::string values("");
      int key = pairs[next].key;
      next = Reduce(key, pairs, next, values);
      Pair p(key, values);
      results.push_back(Pair(key, values));
    }

    // finish timing on root node
    std::chrono::time_point<std::chrono::steady_clock> stop = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = stop - start;

    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;

  } else {
    // code for workers
    while(1) {
      
      // Receive the next task
      MPI_Send((void*)&empty, 1, MPI_CHAR, root, GET_TASK, MPI_COMM_WORLD);
      MPI_Recv(buff, MAX_BUFF, MPI_CHAR, root, ACK, MPI_COMM_WORLD, &status);
      
      if(!buff[0]) {
	// No more tasks to process
	break;
      } else {
	// Process the task
	std::string task(buff);
	int score = Help::score(task.c_str(), protein.c_str());
	if (VERBOSE) 
	  std::cout << rank << ": score(" << task.c_str() << 
	    ", ...) --> " << score << std::endl;
	
	// Send back to root, serialized as a stringstream
	std::stringstream stream;
	stream << task << " " << score;
	MPI_Send((void*)stream.str().c_str(), stream.str().size() + 1, MPI_CHAR, root, TASK_RESULT, MPI_COMM_WORLD);
      }
    }
  }
  
  return results;
}

void MR::Generate_tasks(std::queue<std::string> &q) {
  for (int i = 0;  i < nligands;  i++) {
    q.push(Help::get_ligand(max_ligand));
  }
}


void MR::Sort(std::vector<Pair>& vec) {
  std::sort(vec.begin(), vec.end(), [](const Pair& a, const Pair& b) {
      return a.key > b.key;
    });
}

int MR::Reduce(int key, const std::vector<Pair>& pairs, int index, std::string& values) {
  while(index < pairs.size() && pairs[index].key == key) {
    values += pairs[index++].val + " ";
  }
  
  return index;
}

std::string Help::get_ligand(int max_ligand) {
  int len = 1 + rand()%max_ligand;
  std::string ret(len, '?');
  for (int i = 0;	i < len;	i++)
    ret[i] = 'a' + rand() % 26;	
  return ret;
}


int Help::score(const char *str1, const char *str2) {
  if (*str1 == '\0' || *str2 == '\0')
    return 0;
  // both argument strings non-empty
  if (*str1 == *str2)
    return 1 + score(str1 + 1, str2 + 1);
  else // first characters do not match
    return std::max(score(str1, str2 + 1), score(str1 + 1, str2));
}
