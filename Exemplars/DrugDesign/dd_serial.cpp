#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <chrono>


#define DEFAULT_max_ligand 6
#define DEFAULT_nligands 120
#define DEFAULT_protein "the cat in the hat wore the hat to the cat hat party"


using namespace std;


// key-value pairs, used for both Map() out/Reduce() in and for Reduce() out
struct Pair {
  int key;
  string val;
  Pair(int k, const string &v) {key = k;  val = v;}
};


// MR class provides map-reduce structural pattern
class MR {
private:
  int max_ligand;
  int nligands;
  string protein;


  queue<string> tasks;
  vector<Pair> pairs, results;


  void Generate_tasks(queue<string> &q);
  void Map(const string &str, vector<Pair> &pairs);
  void do_sort(vector<Pair> &vec);
  int Reduce(int key, const vector<Pair> &pairs, int index, string &values);
public:
  MR() {}
  const vector<Pair> &run(int ml, int nl, const string& p);
};


// Auxiliary routines
class Help {
public:
  static string get_ligand(int max_ligand);
  static int score(const char*, const char*);
};


// Main program
int main(int argc, char **argv) {
  int max_ligand = DEFAULT_max_ligand;
  int nligands = DEFAULT_nligands;
  string protein = DEFAULT_protein;
  
  if (argc > 1)
    max_ligand = strtol(argv[1], NULL, 10);
  if (argc > 2)
    nligands = strtol(argv[2], NULL, 10);
  if (argc > 3)
    protein = argv[3];
  // command-line args parsed


  MR map_reduce;
  vector<Pair> results = 
    map_reduce.run(max_ligand, nligands, protein);


  cout << "maximal score is " << results[0].key 
       << ", achieved by ligands " << endl 
       << results[0].val << endl;


  return 0;
}


/*  class MR methods */


const vector<Pair> &MR::run(int ml, int nl, const string& p) {
  max_ligand = ml;  nligands = nl;  protein = p;


  Generate_tasks(tasks);
  // assert -- tasks is non-empty

  // time the code
  std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

  while (!tasks.empty()) {
    Map(tasks.front(), pairs);
    tasks.pop();
  }
  
  do_sort(pairs);


  int next = 0;  // index of first unprocessed pair in pairs[]
  while (next < pairs.size()) {
    string values;
    values = "";
    int key = pairs[next].key;
    next = Reduce(key, pairs, next, values);
    Pair p(key, values);
    results.push_back(p);
  }

  // end time
  std::chrono::time_point<std::chrono::steady_clock> stop = std::chrono::steady_clock::now();
	std::chrono::duration<double, std::milli> elapsed = stop - start;

	std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;


  return results;
}


void MR::Generate_tasks(queue<string> &q) {
  for (int i = 0;  i < nligands;  i++) {
    q.push(Help::get_ligand(max_ligand));
  }
}


void MR::Map(const string &ligand, vector<Pair> &pairs) {
  Pair p(Help::score(ligand.c_str(), protein.c_str()), ligand);
  pairs.push_back(p);
}


bool compare(const Pair &p1, const Pair &p2) {
  return p1.key > p2.key;
}


void MR::do_sort(vector<Pair> &vec) {
  sort(vec.begin(), vec.end(), compare);
}
  
int MR::Reduce(int key, const vector<Pair> &pairs, int index, string &values) {
  while (index < pairs.size() && pairs[index].key == key) 
    values += pairs[index++].val + " ";
  return index;
}




/*  class Help methods */


// returns arbitrary string of lower-case letters of length at most max_ligand
string Help::get_ligand(int max_ligand) {
  int len = 1 + rand()%max_ligand;
  string ret(len, '?');
  for (int i = 0;  i < len;  i++)
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
    return max(score(str1, str2 + 1), score(str1 + 1, str2));
}
