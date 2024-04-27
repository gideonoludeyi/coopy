import argparse
import random
import sys
from functools import partial

import numpy as np
from deap import algorithms, base, creator, gp, tools

from coopy.util import parse_file

from .simulator import (
    PursuitSimulator,
    Specie,
    backward,
    forward,
    if_enemy_around,
    left,
    progn,
    right,
)


def evaluate(
    individual,
    opponents,
    *,
    pset: gp.PrimitiveSet,
    opp_psets: list[gp.PrimitiveSet],
    sim: PursuitSimulator,
):
    if isinstance(individual, creator.Predator):
        pred_routine = gp.compile(individual, pset=pset)
        prey_routine = gp.compile(opponents[0], pset=opp_psets[0])
        sim.run(pred_routine, prey_routine)
        avg_survival_time = sum(x[1] for x in sim.eaten if x is not None) / len(
            sim.eaten
        )
        return (1 - avg_survival_time / sim.max_moves,)
    else:
        pred_routine = gp.compile(opponents[0], pset=opp_psets[0])
        prey_routine = gp.compile(individual, pset=pset)
        sim.run(pred_routine, prey_routine)
        avg_survival_time = sum(x[1] for x in sim.eaten if x is not None) / len(
            sim.eaten
        )
        return (avg_survival_time / sim.max_moves,)


def create_predator_pset() -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("PREDATOR", arity=0)
    pset.addPrimitive(lambda *args: partial(progn, *args), 2, name="prog2")
    pset.addPrimitive(lambda *args: partial(progn, *args), 3, name="prog3")
    pset.addPrimitive(
        lambda *args: partial(
            if_enemy_around, *args, type_=Specie.PREDATOR, radius=1.0
        ),
        2,
        name="ifPreyAround",
    )
    pset.addTerminal(partial(forward, type_=Specie.PREDATOR), name="forward")
    pset.addTerminal(partial(backward, type_=Specie.PREDATOR), name="backward")
    pset.addTerminal(partial(left, type_=Specie.PREDATOR), name="left")
    pset.addTerminal(partial(right, type_=Specie.PREDATOR), name="right")
    return pset


def create_prey_pset() -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("PREY", arity=0)
    pset.addPrimitive(lambda *args: partial(progn, *args), 2, name="prog2")
    pset.addPrimitive(lambda *args: partial(progn, *args), 3, name="prog3")
    pset.addPrimitive(
        lambda *args: partial(if_enemy_around, *args, type_=Specie.PREY, radius=1.0),
        2,
        name="ifPredatorAround",
    )
    pset.addTerminal(partial(forward, type_=Specie.PREY), name="forward")
    pset.addTerminal(partial(backward, type_=Specie.PREY), name="backward")
    pset.addTerminal(partial(left, type_=Specie.PREY), name="left")
    pset.addTerminal(partial(right, type_=Specie.PREY), name="right")
    return pset


def predator_setup(
    sim: PursuitSimulator,
    pset: gp.PrimitiveSet,
    tournsize: int | None = None,
):
    creator.create("PredatorFitness", base.Fitness, weights=(1.0,))
    creator.create(
        "Predator",
        gp.PrimitiveTree,
        fitness=creator.PredatorFitness,
        steps=list,
    )
    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genFull, pset=pset, min_=3, max_=7)
    toolbox.register(
        "individual", tools.initIterate, creator.Predator, toolbox.expr_init
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, pset=pset, sim=sim)
    toolbox.register("select", tools.selTournament, tournsize=tournsize or 1)
    toolbox.register("get_best", tools.selBest, k=1)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=3, max_=7)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(lambda ind: ind.height, 17))
    toolbox.decorate("mutate", gp.staticLimit(lambda ind: ind.height, 17))
    return toolbox


def prey_setup(
    sim: PursuitSimulator,
    pset: gp.PrimitiveSet,
    tournsize: int | None = None,
):
    creator.create("PreyFitness", base.Fitness, weights=(1.0,))
    creator.create(
        "Prey",
        gp.PrimitiveTree,
        fitness=creator.PreyFitness,
        steps=list,
    )
    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genFull, pset=pset, min_=3, max_=7)
    toolbox.register("individual", tools.initIterate, creator.Prey, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, pset=pset, sim=sim)
    toolbox.register("select", tools.selTournament, tournsize=tournsize or 1)
    toolbox.register("get_best", tools.selBest, k=1)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=3, max_=7)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(lambda ind: ind.height, 17))
    toolbox.decorate("mutate", gp.staticLimit(lambda ind: ind.height, 17))
    return toolbox


parser = argparse.ArgumentParser(prog="Pursuit")
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
    "-e", "--elites", dest="n_elites", type=int, required=False, default=0
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
    default=None,
    help="the number of individuals to compete in tournament [default: none - uses roulette wheel instead]",
)
parser.add_argument(
    "-s", "--seed", dest="random_seed", type=int, required=False, default=123
)


def main() -> int:
    args = parser.parse_args()
    random.seed(args.random_seed)

    config = parse_file(args.mapfile)
    sim = PursuitSimulator(
        predators=config["preds"],
        preys=config["preys"],
        nrows=config["nrows"],
        ncols=config["ncols"],
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
        predator_toolbox.population(n=args.popsize // 2),
        prey_toolbox.population(n=args.popsize // 2),
    ]
    representatives = [random.choice(s) for s in species]
    for g in range(args.n_generations):
        next_representatives = [None] * len(species)
        for i, s in enumerate(species):
            is_predator = any(isinstance(ind, creator.Predator) for ind in s)
            toolbox = predator_toolbox if is_predator else prey_toolbox
            s = toolbox.select(s, len(s))
            s = algorithms.varAnd(s, toolbox, args.crossover_rate, args.mutation_rate)
            r = representatives[:i] + representatives[i + 1 :]
            for ind in s:
                ind.fitness.values = toolbox.evaluate(
                    ind,
                    r,
                    opp_psets=[prey_pset if is_predator else predator_pset],
                )
            species[i] = s
            record = stats.compile(s)
            logbook.record(
                gen=g,
                species=Specie.PREDATOR if is_predator else Specie.PREY,
                evals=len(s),
                **record,
            )
            print(logbook.stream, file=args.logfile)
            next_representatives[i] = toolbox.get_best(s)[0]
        representatives = next_representatives

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


if __name__ == "__main__":
    sys.exit(main())
