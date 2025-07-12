

// From wikipedia: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
// The 'classic' rules are:
/*
Each cell has two states: alive or dead.
Each cell's next state is determined by the number of its living neighbors.
Rules:

    Any live cell with fewer than two live neighbours dies (underpopulation).
    Any live cell with two or three live neighbours lives on.
    Any live cell with more than three live neighbours dies (overpopulation).
    Any dead cell with exactly three live neighbours becomes a live cell (reproduction)
 */

#define PROB_HARDY 0.08 // likelihood of a cell being hardy and remaining alive

void apply_rules(double randN, int *grid, int *newGrid, int id, int w) {
    // Implementing the Game of Life Rules
    // NOTE: randN is used to determine if a cell is hardy, which affects its survival
    //       if randN < PROB_HARDY, the cell is hardy and will remain alive if it is alive
    //       When randN is -1.0, it means we are not using the stochastic rules, 
    //       so we do not check for hardiness.
  
        int neighbors[8] = {
            grid[id + (w + 2)], //south
            grid[id - (w + 2)], //north
            grid[id + 1],       // east
            grid[id - 1],       // west
            grid[id + (w + 3)], // southeast
            grid[id - (w + 3)], // northwest
            grid[id - (w + 1)], // northeast
            grid[id + (w + 1)]};// southwest

        int liveNeighbors = 0;  
        for (int k = 0; k < 8; k++) {  
            liveNeighbors += neighbors[k];
        }

    // Any live cell with fewer than two live neighbours dies (underpopulation).
    if (grid[id] == 1 && liveNeighbors < 2) {
        newGrid[id] = 0;  // cell dies, unless it is hardy
        if ((randN < PROB_HARDY) && (randN >= 0.0)) {
            newGrid[id] = 1;  // cell remains alive if it is hardy
        }
    }
    // Any live cell with two or three live neighbours lives on.
    else if (grid[id] == 1 && (liveNeighbors == 2 || liveNeighbors == 3)) {
        newGrid[id] = 1;  // cell lives on
    }
    // Any live cell with more than three live neighbours dies (overpopulation).
    else if (grid[id] == 1 && liveNeighbors > 3) {
        newGrid[id] = 0;  // cell dies unless it is hardy
        if ((randN < PROB_HARDY) && (randN >= 0.0)) {
            newGrid[id] = 1;  // cell remains alive if it is hardy
        }
    }
    // Any dead cell with exactly three live neighbours becomes a live cell (reproduction)
    else if (grid[id] == 0 && liveNeighbors == 3) {
        newGrid[id] = 1;  // cell becomes alive
    }
 
}
