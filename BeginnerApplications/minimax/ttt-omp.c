//Tic-tac-toe playing AI. Exhaustive tree-search. WTFPL
//adapted from
//Matthew Steel 2009, www.www.repsilat.com

#include <stdio.h>
#include <stdlib.h>   // malloc
#include <string.h>   // memcopy
#include <omp.h>      // openMP 

#define BOARDSIZE 9

char gridChar(int i) {
    switch(i) {
        case -1:
            return 'X';
        case 0:
            return ' ';
        case 1:
            return 'O';
    }
    return ' ';
}

void draw(int *b) {
    printf(" %c | %c | %c\n",gridChar(b[0]),gridChar(b[1]),gridChar(b[2]));
    printf("---+---+---\n");
    printf(" %c | %c | %c\n",gridChar(b[3]),gridChar(b[4]),gridChar(b[5]));
    printf("---+---+---\n");
    printf(" %c | %c | %c\n",gridChar(b[6]),gridChar(b[7]),gridChar(b[8]));
}

int win(int * board) {
    //determines if a player has won, returns 0 otherwise.
    //kind of inelegant and non-scalable to hard-code wins
    unsigned int wins[8][3] = {{0,1,2},{3,4,5},{6,7,8},{0,3,6},{1,4,7},{2,5,8},{0,4,8},{2,4,6}};
    int i;
    for(i = 0; i < 8; ++i) {
        //for the given current winning board
        // does our board have matching pieces at each of those locations?
        unsigned int curwin0 = wins[i][0];
        unsigned int curwin1 = wins[i][1];
        unsigned int curwin2 = wins[i][2];
        if(board[curwin0] != 0 &&
           board[curwin0] == board[curwin1] &&
           board[curwin0] == board[curwin2])
            return board[curwin2];  //if yes, return the player (-1 or 1) who wins
    }
    return 0;
}


//minimax in a single recursive function
// you call max if it is your move
// and min if it is your opponent's move.
int minimax(int * board, int player) {
    //How is the position like for player (their turn) on board?
    int winner = win(board);   //is the board a win?
    if(winner != 0) return winner*player; //base case

    int curbestmove = -1; //the best move possible
    int curbestscore = -2;//Losing moves are preferred to no move
    int i;
    for(i = 0; i < BOARDSIZE; ++i) {//For all moves,
        if(board[i] == 0) {//If legal,
            board[i] = player;//Try the move
	//    draw(board);   //debug
            int thisScore = -1 * minimax(board, player*-1);
            if(thisScore > curbestscore) {
                curbestscore = thisScore;
                curbestmove = i;
            }//Pick the one that's worst for the opponent
            board[i] = 0;//Reset board after try
        }
    }
    if(curbestmove == -1) return 0;
    return curbestscore;
}

void computerMove(int * board, int nthreads) {
    int bestmove = -1;
    int score = -2;
    int i;
    //printf("computer move:\n");
    //draw(board);
    #pragma omp parallel num_threads(nthreads) 
    {
        int *privateboard = malloc(9*sizeof(int));
        memcpy((void *)privateboard,(void *)board,9*sizeof(int));

        #pragma omp for schedule(dynamic,1)
        for(i = 0; i < BOARDSIZE; ++i) {
            if(privateboard[i] == 0) {
                privateboard[i] = 1;
                int tempScore = -minimax(privateboard, -1);
                privateboard[i] = 0;
                //critical to protect shared variables
                #pragma omp critical
                if(tempScore > score) {
                     score = tempScore;
                     bestmove = i;
                }
            }
        }
    }
    //returns a score based on minimax tree at a given node.
    board[bestmove] = 1;
}

void playerMove(int board[9]) {
    int move = 0;
    do {
        printf("\nInput move ([0..8]): ");
        scanf("%d", &move);
        printf("\n");
    } while ((move >= BOARDSIZE || move < 0) && board[move] == 0);
    board[move] = -1;
}

int main() {
    int board[BOARDSIZE] = {0,0,0,0,0,0,0,0,0};

    int nthreads = 4;
    //computer squares are 1, player squares are -1.
    printf("Computer: O, You: X\nPlay (1)st or (2)nd? ");
    int player=0;
    scanf("%d",&player);
    printf("\n");
    unsigned turn;
    for(turn = 0; turn < BOARDSIZE && win(board) == 0; ++turn) {
        if((turn+player) % 2 == 0)
            computerMove(board,nthreads);
        else {
            draw(board);
            playerMove(board);
        }
    }
    switch(win(board)) {
        case 0:
            printf("A draw. How droll.\n");
            break;
        case 1:
            draw(board);
            printf("You lose.\n");
            break;
        case -1:
            printf("You win. Inconceivable!\n");
            break;
    }
}
