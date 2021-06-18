import numpy as np

UNBURNT = 1
BURNT = 0
SMOLDERING = 2
BURNING = 3



def initialize_forest(size):
    """ Create the forest of unburnt trees as a 2D array

    Parameters:
        size (int): number of trees in each row and column
    Returns:
        size x size numpy array
    """

    forest = np.empty( (size, size), dtype='u4')
    forest.fill(UNBURNT)
    return forest


def light_tree(row_size, forest, x, y):
    """ Tree at position x, y in forest set to smoldering

    Parameters:
        row_size (int): number of trees in each row and column
        forest (array): array representing the 2D forest
        x (int), y (int): x,y location of tree to set smoldering

    post[forest]:
        If x, y is properly within the array, the tree at x, y is set.
        Otherwise, one tree in the center of the forest is set.
    """

    # note that indexes into numpy arrays must be integers
    if x >= row_size or y >= row_size :
        print("Warning: starting position out of bounds; using center")
        i = int(row_size/2)
        j = int(row_size/2)
    else:
        i = int(x)
        j = int(y)

    forest[i,j] = SMOLDERING

def fire_spreads(prob_spread):
    """ Generates a random number between 0 an 1 and checks if < prob_spread

    Parameters:
        prob_spread (float):
            probability threshold for determining whether burning tree will
            spread to neighboring tree

    Returns:
        True if new random value is < prob_spread, False otherwise
    """

    prob = np.random.random_sample()

    if prob < prob_spread:
        return True
    else:
        return False

def forest_burns(forest, row_size, prob_spread):
    """One round of burning the forest

    Sets every burning tree to burnt, and every smoldering tree to burning.
    Then sets fire to unburt trees next to burning trees randomly, based on
    the probability of spreading.

    Parameters:
        forest (array): array representing the 2D forest
        row_size (int): number of trees in each row and column
        prob_spread (float):
            probability threshold for determining whether burning tree will
            spread to neighboring tree

    """

    # burning trees burn down, smoldering trees ignite
    for index, value in np.ndenumerate(forest):
        # print(forest[index], value) #debug
        if forest[index] == BURNING:
            forest[index] = BURNT
        if forest[index] == SMOLDERING:
            # print(index, " SMOLDERING")  #debug
            # print(index[0], index[1])
            forest[index] = BURNING

    for index, value in np.ndenumerate(forest):
        i, j = index[0], index[1]    # row, col, location in grid

        # unburnt trees surrounding burning trees catch fire if
        # random probability is above threshold of probability of spreading
        if forest[index] == BURNING:
            if i != 0 :  #check tree to north if not on top row
                if forest[(i-1, j)] == UNBURNT  and fire_spreads(prob_spread):
                    forest[(i-1, j)] = SMOLDERING
            if i != row_size-1 :  #check tree to south if not on last row
                if forest[(i+1, j)] == UNBURNT and fire_spreads(prob_spread):
                    forest[(i+1, j)] = SMOLDERING
            if j != 0 :  #check tree to west if not on left side
                if forest[(i, j-1)] == UNBURNT and fire_spreads(prob_spread):
                    forest[(i, j-1)] = SMOLDERING
            if j != row_size-1 : #check tree to east if not on right side
                if forest[(i, j+1)] == UNBURNT and fire_spreads(prob_spread):
                    forest[(i, j+1)] = SMOLDERING


def forest_is_burning(forest):
    """ Checks for any remaining smoldering or burning trees

    Parameters:
        forest (array): array representing the 2D forest

    Returns:
        True if at least one tree is still in burning or smoldering state.
        False if all trees are burnt.

    """
    for row in forest:
        for tree in row:
            if tree == SMOLDERING or tree == BURNING:
                return True
    return False

def get_percent_burned(forest, row_size):
    """ Determine how many trees burned during fire

    Parameters:
        forest (array): array representing the 2D forest
        row_size (int): number of trees in each row and column

    Returns:
        percent : float
            Percentage of total number of trees that were burned,
            as a float between 0 and 1.
    """

    sum = 0
    for row in forest:
        for tree in row:
            if tree == BURNT:
                sum +=1

    return float(sum)/float(row_size*row_size)

def print_forest(forest):
    """ Ascii display of forest

    Prints values for state of each tree in the forest.

    Parameters:
        forest (array): array representing the 2D forest
    """

    for row in forest:
        rowStr = ''
        for tree in row:
            if tree == BURNT:
                rowStr = rowStr + '.'
            elif tree == UNBURNT:
                rowStr = rowStr + 'Y'
            elif tree == SMOLDERING:
                rowStr = rowStr + 'S'
            else:
                rowStr = rowStr + 'B'

        print(rowStr)
    print('\n')
