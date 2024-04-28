import argparse
import logging
import os
import random
import sys
import typing
from dataclasses import dataclass

import numpy as np
from deap import algorithms, creator, tools

from coopy.core import (
    create_predator_pset,
    create_prey_pset,
    predator_setup,
    prey_setup,
)
from coopy.simulator import PursuitSimulator, Specie
from coopy.util import parse_file

logger = logging.getLogger(__name__)


@dataclass
class RunArgs:
    mapfile: str | os.PathLike = "data/map.txt"
    max_moves: int = 600
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    tournsize: int = 3
    popsize: int = 100
    n_generations: int = 30
    outputfile: typing.TextIO = sys.stdout
    logfile: typing.TextIO = sys.stderr
    random_seed: int | float | str | bytes | bytearray | None = None


def run(args: RunArgs):
    random.seed(args.random_seed)

    config = parse_file(args.mapfile)
    sim = PursuitSimulator(
        predators=config["preds"],
        preys=config["preys"],
        nrows=config["nrows"],
        ncols=config["ncols"],
        max_moves=args.max_moves,
    )

    predator_pset = create_predator_pset()
    predator_toolbox = predator_setup(sim, predator_pset, tournsize=args.tournsize)

    prey_pset = create_prey_pset()
    prey_toolbox = prey_setup(sim, prey_pset, tournsize=args.tournsize)

    logbook = tools.Logbook()
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("max", np.max)

    species = [
        predator_toolbox.population(n=args.popsize),
        prey_toolbox.population(n=args.popsize),
    ]
    representatives = [random.choice(s) for s in species]
    for g in range(args.n_generations):
        next_representatives = [None] * len(species)
        for i, s in enumerate(species):
            is_predator = any(isinstance(ind, creator.Predator) for ind in s)
            toolbox = predator_toolbox if is_predator else prey_toolbox
            r = representatives[:i] + representatives[i + 1 :]
            s = algorithms.varAnd(
                toolbox.select(s, len(s)),
                toolbox,
                args.crossover_rate,
                args.mutation_rate,
            )
            for ind in s:
                ind.fitness.values = toolbox.evaluate(
                    ind,
                    r,
                    opp_psets=[prey_pset if is_predator else predator_pset],
                )
            species[i] = s
            logbook.record(
                gen=g,
                species=Specie.PREDATOR if is_predator else Specie.PREY,
                evals=len(s),
                **stats.compile(s),
            )
            logger.debug(logbook.stream)
            next_representatives[i] = toolbox.get_best(s)[0]
        representatives = next_representatives
    print(logbook, file=args.logfile)

    bestpred = max(species[0], key=lambda ind: ind.fitness.values)
    bestprey = max(species[1], key=lambda ind: ind.fitness.values)
    with args.outputfile as f:
        print(
            f"Best Predator: fitness={bestpred.fitness.values}, height={bestpred.height}",
            file=f,
        )
        print(str(bestpred), file=f)
        f.write("\n")
        print(
            f"Best Prey: fitness={bestprey.fitness.values}, height={bestprey.height}",
            file=f,
        )
        print(str(bestprey), file=f)

    return 0


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-o",
        "--output",
        dest="outputfile",
        type=argparse.FileType("w"),
        default=sys.stdout,
        required=False,
        help="the file which to write the best solution tree [default: stdout]",
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
        "--logfile",
        dest="logfile",
        type=argparse.FileType("w"),
        default=sys.stderr,
        required=False,
        help="the file which to write the training logs [default: stderr]",
    )
    parser.add_argument(
        "-c",
        "--crossover-rate",
        dest="crossover_rate",
        type=float,
        required=False,
        default=0.9,
    )
    parser.add_argument(
        "-m",
        "--mutation-rate",
        dest="mutation_rate",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "-g",
        "--generations",
        dest="n_generations",
        type=int,
        required=False,
        default=30,
    )
    parser.add_argument(
        "-p", "--popsize", dest="popsize", type=int, required=False, default=100
    )
    parser.add_argument(
        "--max-moves", dest="max_moves", type=int, required=False, default=600
    )
    parser.add_argument(
        "--tournsize",
        dest="tournsize",
        type=int,
        required=False,
        default=3,
        help="the number of individuals to compete in tournament [default: 3]",
    )
    parser.add_argument(
        "-s", "--seed", dest="random_seed", type=int, required=False, default=123
    )

    def handler(args: argparse.Namespace):
        run_args = RunArgs(
            mapfile=args.mapfile,
            max_moves=args.max_moves,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            tournsize=args.tournsize,
            popsize=args.popsize,
            n_generations=args.n_generations,
            outputfile=args.outputfile,
            logfile=args.logfile,
            random_seed=args.random_seed,
        )
        return run(run_args)

    parser.set_defaults(__exec__=handler)
