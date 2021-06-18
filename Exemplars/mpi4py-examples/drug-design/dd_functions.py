import random
import string
import argparse
from mpi4py import MPI

DFLT_maxLigand = 5
DFLT_nLigands = 120
DFLT_protein = "the cat in the hat wore the hat to the cat hat party"
#shorter protein takes less time
#DFLT_protein = "Cwm fjord bank glyphs vext quiz"
# DFLT_protein = "How razorback-jumping frogs can level six piqued gymnasts"

# function getCOmmandLineArgs

def getCommandLineArgs():
    parser = argparse.ArgumentParser(
        description="CSinParallel Drug Design simulation")

    parser.add_argument('nLigands', metavar='count', type=int, nargs='?',
        default=DFLT_nLigands, help='number of ligands to generate')
    parser.add_argument('--maxLigand', metavar='max-length', type=int, nargs='?',
        default=DFLT_maxLigand, help='maximum length of a ligand')
    parser.add_argument('--protein', metavar='protein', type=str, nargs='?',
        default=DFLT_protein, help='protein string to compare ligands against')
    parser.add_argument('--verbose', action='store_const', const=True,
                        default=False, help='print verbose output')
    args = parser.parse_args()
    return args

def genRandomLigands(args):
    random.seed(1000)
    ligands = []
    for l in range(args.nLigands):
        ligands.append(makeLigand(args.maxLigand))
    return ligands

# If args.nLigands <=18, create a pre-determined set of example ligands.
# Otherwise, create a set of ligands whose length randomly varies from 2
# to args.maxLigand
def genLigandList(args):
    if (args.nLigands <= 18):
        ligands = ["razvex", "qudgy", "afrs", "sst", "pgfht", "rt", "id", \
        "how", "aaddh",  "df", "os", "hid", \
        "sad", "fl", "rd", "edp", "dfgt", "spa"]
        return ligands[0:args.nLigands]
    else:
        return genRandomLigands(args)

# function makeLigand
#   1 argument:  maximum length of a ligand
#   return:  a random ligand string of random length between 2 and arg1

def makeLigand(maxLength):
#    len = random.randint(2, maxLength)
    # So the times do not get too large, create
    # more ligands of length 2, 3 by using a gamma distribution
    len = int(random.gammavariate(4.2, 0.8))
    if (len < 2):
        len = 2
    elif (len > maxLength):
        len = maxLength

    ligand = ""
    for c in range(len):
        ligand = ligand + string.ascii_lowercase[random.randint(0,25)]
    return ligand

# function score
#   2 arguments:  a ligand and a protein sequence
#   return:  int, simulated binding score for ligand arg1 against protein arg2

def score(lig, pro):
    if len(lig) == 0 or len(pro) == 0:
        return 0
    if lig[0] == pro[0]:
        return 1 + score(lig[1:], pro[1:])
    else:
        return max(score(lig[1:], pro), score(lig, pro[1:]))

# functions used by Master
#
# receive next finished result
def masterReceiveResults(comm, args):
    stat = MPI.Status()
    results = comm.recv(source=MPI.ANY_SOURCE, status=stat)
    workerId = stat.Get_source()
    printIf(args.verbose, "master received {} with score {} from {}"\
    .format(results[1], results[0], workerId), flush=True)
    return results, workerId

# Keep track of the maximum score and ligand that achieved it
def updateMaximum(score, lig, maxScore, maxScoreLigands):
    if score > maxScore:
        # new best scoring one
        maxScore = score
        maxScoreLigands = lig
    elif score == maxScore:
        maxScoreLigands = maxScoreLigands + "," + lig

    return maxScore, maxScoreLigands

# function printIf - used for verbose output
#   variable number of arguments:  a boolean, then valid arguments for print
#   state change:  if arg1 is True, call print with the remaining arguments

def printIf(cond, *positionals, **keywords):
    if cond:
        print(*positionals, **keywords)
