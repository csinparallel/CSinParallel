/*
  Original code provided by: 
  Dave Valentine, Slippery Rock University.
  Edited by Libby Shoop, Macalester College.
*/
/*
Simulate American Roulette wheel
	American wheel has 38 slots:
		-2 are 'green' (0 & 00)
			house ALWAYS wins on green
		-18 are red (1..18)
		-18 are black (1..18)

	Our user always bets "red"
	Odds should be:  18/38 or 47.37%
	
*/

// trng YARN (yet another random number) generator class
#include <trng/yarn2.hpp>  
#include <trng/uniform_dist.hpp>   // uniform distribution 
                                   // of the random numbers

#include <iostream>
#include <iomanip>
#include <time.h>
#include <string>
using namespace std;

// limit to number of spins in last simulation.
// first simulation does 1 spin, next 2, next 4, etc.,
// doubling each time until reach this maximum.
const int MAX = 1<<26;	

#include "./utils/getCommandLine.h"

//Function Prototypes
int getNumWins(int numSpins, long unsigned int seed);
void showResults(int numSpins, int numWins);
int spinRed(int bet, unsigned int slot);


/**************************** MAIN **********************/
int main(int argc, char* argv[]) {
	//Variables
	int numSpins;		//#spins per trial
	int	numWins;		//#wins per trial
	clock_t startT, stopT; //wall clock elapsed time


/***** Initialization *****/
// defaults
	int maxSpins = MAX; // set with -n on command line
	// when testing, set this 
	int useConstantSeed = 0; //to 1 on command line with -c

	// command line arguments
	getArguments(argc, argv, &maxSpins, &useConstantSeed);

	// random numbers start from a seed value
  	long unsigned int seed;  
	// note for trng this is long unsigned

	if (useConstantSeed) {
		seed = 503895321;     
	} else {  // variable seed based on computer clock time
		seed = (long unsigned int)time(NULL); 
	}

	startT = clock();     // start the timer

	cout<<"Simulation of an American Roulette Wheel\n" <<
		string(35,'*')<<endl;
	cout<<setw(12)<<"Num Spins" << setw(12) <<"Num Wins" << setw(12) <<"% wins"<<endl;
	numSpins = 1; //we start with 1 spin (all or nothing)

/**************************** Do Simulations ***/		
	while (numSpins < maxSpins) {
		//go spin wheel numSpins times
		numWins = getNumWins(numSpins, seed);	
		showResults(numSpins, numWins);	

		//double spins for next simulation
		numSpins += numSpins;	
	} 

/**************************** Finish Up ********/
	stopT= clock();		//stop our timer & show elapsed time
	cout<<"\nElapsed wall clock time: "<< (double)(stopT-startT)/CLOCKS_PER_SEC<<endl<<endl;

	cout<<"\n\n\n\t\t*** Normal Termination ***\n\n";
	return 0;
} // end main


/*********************** getNumWins ************/ 
// perform one simulation of numSpins spins of the wheel
int getNumWins(int numSpins, long unsigned int seed) {
//always bet 'red' & count wins
	static int wins;//our counter
	int spin;		//loop cntrl var
	int myBet = 10; //amount we bet per spin

	wins = 0;	//clear our counter

	unsigned int nextRandVal;
	unsigned min = 1;
  	unsigned max = 39; // get vals between 1 and 38

	trng::yarn2 rand;
	rand.seed(seed);
	trng::uniform_dist<> uniform(min, max);
	
	for (spin=0; spin<numSpins; spin++){
		nextRandVal = uniform(rand);
		//spinRed returns +/- number (win/lose)
		if (spinRed(myBet, nextRandVal) > 0) //a winner!
			wins++;
	}
	
	return wins;
}  //getNumWins

//spin the wheel, betting on RED
//Payout Rules:
//  0..17 you win (it was red)
// 18..35 you lose (it was black)
// 36..37 house wins (green) - you lose half
int spinRed(int bet, unsigned int slot) {
	int payout;

	if (slot <= 18) //simplify odds: [1..18]==RED
		payout = bet;	//won
	else if (slot <= 36) //spin was 'black'-lose all
		payout = -bet;	//lost
	else //spin was green - lose
	    payout = -bet;	//lost
		// payout = -(bet/2); //half-back alternative
	return payout;
} // spinRed

/*********************** prettyInt *************/
string prettyInt(int n) {
//comma-delimited string made from int
	string s="";	//what we're making
	int digit;		//each digit of n
	int digitCnt=0; //count by 3's for comma insert

	do {
		digit = n % 10;		//get lsd
		n = n/10;			//and chop it from n
		//make digit into numeric char
		char c = (char) ( (int)'0' + digit);
		s.insert(0,1,c);	//insert char to string
		digitCnt++;			//count digits in string
		if ( (digitCnt%3 == 0) && (n>0) )
			s.insert(0,1,',');
	} while (n>0);
	return s;
} //prettyInt


/*********************** showResults ***********/
void showResults(int numSpins, int numWins){
//calc %wins & printout the 3 columns
	double percent = 100.0* (double)numWins/(double)numSpins;
	cout<<setw(12)<<prettyInt(numSpins) << setw(12) << 
		prettyInt(numWins) << setw(12) <<
		setprecision (4) << fixed << percent<< endl;
} //showResults
