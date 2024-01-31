//Tic-tac-toe playing AI. Exhaustive tree-search. WTFPL
//adapted from
//Matthew Steel 2009, www.www.repsilat.com

#include <stdio.h>

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
           draw(board);
 //        getchar();
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


int main() {
    int board[BOARDSIZE] = {1,-1,1,0,0,0,0,0,0};
    //computer squares are 1, player squares are -1.
    int tempScore = -minimax(board, -1);
        
}
