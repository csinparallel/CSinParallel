#
# Run one simulation of a fire burning at one probability threshold.
#
# Ported to python from the original Shodor foundation example:
#  https://www.shodor.org/refdesk/Resources/Tutorials/BasicMPI/
#
# Libby Shoop     Macalester College
#
import argparse      # for command-line arguments
import matplotlib.pyplot as plt

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
    parser.add_argument("probabilityOfSpread", help="probability threshold of fire spreading from one burning tree to a non-burning tree next to it (percent between 0 and 1)")
    # could add: optional arguments for i, j position of starting tree
    args = parser.parse_args()

    row_size = int(args.numTreesPerRow)
    prob_spread = float(args.probabilityOfSpread)

    return row_size, prob_spread

############################# main() ##########################
def main():

    row_size, prob_spread = parseArguments()

    forest = initialize_forest(row_size)

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
    print("Iterations until fire burns out: {}".format(iter))
    print("Percent burned: {0:4.3f}".format(percent_burned))
    # print_forest(forest)

    plt.figure("Single Forest Fire Simulation",figsize=(7,7))
    plt.pcolor(forest, cmap=plt.cm.get_cmap('Greens', 2))
    plt.title("{0}x{0} grid of trees, Probability {3:3.2f}\nIterations until fire burns out: {1} Percent burned: {2:4.3f}\nGreen squares are live trees after one simulation".format(row_size, iter, percent_burned, prob_spread))

    plt.show()


########## Run the main function
main()
