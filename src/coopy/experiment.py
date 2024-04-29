import argparse
import logging
import os
import pathlib
import random
from dataclasses import dataclass

from coopy.run import RunArgs, run

logger = logging.getLogger(__name__)


@dataclass
class ExperimentArgs:
    mapfile: str | os.PathLike = "data/map.txt"
    max_moves: int = 600
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    tournsize: int = 3
    popsize: int = 100
    n_generations: int = 30
    n_elites: int = 0
    outdir: str | os.PathLike = "output.d/"
    n_runs: int = 10
    random_seed: int | float | str | bytes | bytearray | None = None


def experiment(args: ExperimentArgs):
    random.seed(args.random_seed)
    seeds = []
    while len(seeds) < args.n_runs:
        if (seed := random.randint(1_000, 100_000_000)) not in seeds:
            seeds.append(seed)
    output_dir = pathlib.Path(args.outdir)
    output_dir.mkdir(exist_ok=True)
    logger.info("%s", seeds)
    for seed in seeds:
        with (
            open(output_dir.joinpath(f"output_{seed}.txt"), "w") as outfile,
            open(output_dir.joinpath(f"logs_{seed}.txt"), "w") as logfile,
        ):
            run(
                RunArgs(
                    mapfile=args.mapfile,
                    max_moves=args.max_moves,
                    crossover_rate=args.crossover_rate,
                    mutation_rate=args.mutation_rate,
                    tournsize=args.tournsize,
                    popsize=args.popsize,
                    n_generations=args.n_generations,
                    n_elites=args.n_elites,
                    outputfile=outfile,
                    logfile=logfile,
                    random_seed=seed,
                )
            )
        logger.info("Completed Run with seed=%s", seed)


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--mapfile",
        dest="mapfile",
        type=str,
        default="data/map.txt",
        required=False,
        help="the file which to read the initial map configuration [default: data/map.txt]",
    )
    parser.add_argument(
        "--output-dir",
        dest="outdir",
        type=str,
        default="output.d/",
        required=False,
        help="the file which to save the experiment results [default: output.d/]",
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
        "-e",
        "--elites",
        dest="n_elites",
        type=int,
        required=False,
        default=0,
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
    parser.add_argument("--runs", dest="n_runs", type=int, required=False, default=10)
    parser.add_argument(
        "-s", "--seed", dest="random_seed", type=int, required=False, default=123
    )

    def handler(args: argparse.Namespace):
        exp_args = ExperimentArgs(
            mapfile=args.mapfile,
            outdir=args.outdir,
            max_moves=args.max_moves,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            tournsize=args.tournsize,
            popsize=args.popsize,
            n_generations=args.n_generations,
            n_elites=args.n_elites,
            n_runs=args.n_runs,
            random_seed=args.random_seed,
        )
        return experiment(exp_args)

    parser.set_defaults(__exec__=handler)
