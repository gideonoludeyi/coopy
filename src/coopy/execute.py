import argparse
import logging
import os
import random
import sys
import typing
from dataclasses import dataclass

from deap import gp

from coopy.core import (
    create_predator_pset,
    create_prey_pset,
)
from coopy.simulator import PursuitSimulator
from coopy.util import parse_file

logger = logging.getLogger(__name__)


@dataclass
class ExecuteArgs:
    """parameters for the `experiment` command"""

    mapfile: str | os.PathLike = "data/map.txt"
    max_moves: int = 600
    outputfile: typing.TextIO = sys.stdout
    inputfile: typing.TextIO = sys.stdin
    random_seed: int | float | str | bytes | bytearray | None = None


def dir_icon(d: complex) -> str:
    """convert a direction into the appropriate predator icon"""
    return {
        0 - 1j: "^",  # up
        1 + 0j: ">",  # right
        0 + 1j: "v",  # down
        -1 + 0j: "<",  # left
    }[d]


def genmap(sim: PursuitSimulator) -> str:
    """generates an ASCII map of predators and preys from the current state of the simulator"""
    nrows, ncols = sim.nrows, sim.ncols
    preds = sim.predators
    preys = sim.preys
    eaten = sim.eaten
    grid = [["." for _ in range(ncols)] for _ in range(nrows)]
    for prey in eaten:
        if prey is not None:
            row, col, d = prey[0]
            grid[int(row)][int(col)] = "x"  # eaten prey
    for i, (row, col, d) in enumerate(preys):
        if eaten[i] is None:
            grid[int(row)][int(col)] = "#"  # alive prey
    for row, col, d in preds:
        grid[int(row)][int(col)] = dir_icon(d)  # predator
    return "\n".join("".join(cell for cell in row) for row in grid)


def execute(args: ExecuteArgs):
    """
    generates a trace ASCII map visualizing the state of the simulation at each time step
    """
    random.seed(args.random_seed)  # set random seed for reproducibility
    config = parse_file(args.mapfile)
    sim = PursuitSimulator(
        predators=config["preds"],
        preys=config["preys"],
        nrows=config["nrows"],
        ncols=config["ncols"],
        max_moves=args.max_moves,
    )
    predator_pset = create_predator_pset()
    prey_pset = create_prey_pset()
    with args.inputfile as f:
        # the second line is the string representation of the predator program
        # the fifth line is the string representation of the prey program
        # example: `tmp/singlepred-multiprey.solution.txt`
        lines = f.readlines()
        bestpred_routine = gp.compile(lines[1], pset=predator_pset)
        bestprey_routine = gp.compile(lines[4], pset=prey_pset)
    # output the ASCII map for each step in the simulation
    with args.outputfile as f:
        print(genmap(sim), file=f, end="\n\n")
        for _ in sim.run_iter(bestpred_routine, bestprey_routine):
            print(genmap(sim), file=f, end="\n\n")
    return 0


def setup_parser(parser: argparse.ArgumentParser):
    """
    defines command-line arguments for the execute subcommand
    """
    parser.add_argument(
        "-o",
        "--output",
        dest="outputfile",
        type=argparse.FileType("w"),
        default=sys.stdout,
        required=False,
        help="the file which to write the executed trace on the map [default: stdout]",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="inputfile",
        type=argparse.FileType("r"),
        default=sys.stdin,
        required=False,
        help="the file which to read the best solution tree [default: stdin]",
    )
    parser.add_argument(
        "--mapfile",
        dest="mapfile",
        type=str,
        default="data/map.txt",
        required=False,
        help="the file which to read the initial map configuration [default: data/map.txt]",
    )
    parser.add_argument(
        "--max-moves", dest="max_moves", type=int, required=False, default=600
    )
    parser.add_argument(
        "-s", "--seed", dest="random_seed", type=int, required=False, default=123
    )

    def handler(args: argparse.Namespace):
        exec_args = ExecuteArgs(
            mapfile=args.mapfile,
            max_moves=args.max_moves,
            outputfile=args.outputfile,
            inputfile=args.inputfile,
            random_seed=args.random_seed,
        )
        return execute(exec_args)

    parser.set_defaults(__exec__=handler)
