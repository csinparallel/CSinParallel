#
# Functions used by both the sequential and MPI version of
# the larger simulation with many trials.
#
 # Libby Shoop     Macalester College
 #
import argparse      # for command-line arguments
from fire_functions import *

def parseArguments():
    """Handle command line arguments

    Run with -h to get details of each argument.

    Returns:
        A list containg each argument provided
    """
    # process command line arguments
    # see https://docs.python.org/3.3/howto/argparse.html#id1
    parser = argparse.ArgumentParser()
    parser.add_argument("numTreesPerRow", help="number of trees in row of square grid")
    parser.add_argument("probabilityIncrement", help="amount to increment the probability threshold of fire spreading for each set of probability trials")
    parser.add_argument("numberOfTrials", help="number of times to run the fire simulation with a new forest for each proability in set of probabilities")

    args = parser.parse_args()

    row_size = int(args.numTreesPerRow)
    prob_spread_increment = float(args.probabilityIncrement)
    num_trials = int(args.numberOfTrials)

    return [row_size, prob_spread_increment, num_trials]


def burn_until_out(row_size, forest, prob_spread):
    """ one simulation of the buring forest
    Parameters:
        row_size (int): number of trees in each row and column
        forest (array): array representing the 2D forest
        prob_spread (float):
            probability threshold for determining whether burning tree will
            spread to neighboring tree
    """

    percent_burned = 0.0
    # for now start burning at midlle tree
    middle_tree_index = int(row_size/2)
    light_tree(row_size, forest, middle_tree_index, middle_tree_index)

    iter = 0 # how many iterations before the fire burns out
    while forest_is_burning(forest):
        # print("burning") # debug
        forest_burns(forest, row_size, prob_spread)
        iter += 1

    percent_burned = get_percent_burned(forest, row_size)

    # print_forest(forest)  #debug

    return int(iter), float(percent_burned)
