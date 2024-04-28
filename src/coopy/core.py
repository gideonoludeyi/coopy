import os
from functools import partial
from deap import algorithms, base, creator, gp, tools

from .simulator import (
    PursuitSimulator,
    Specie,
    backward,
    forward,
    if_enemy_around,
    if_enemy_behind,
    if_enemy_forward,
    if_enemy_left,
    if_enemy_right,
    left,
    progn,
    right,
)

FITFN = os.getenv(
    "FITFN", "0"
)  # 0 -> survival_time_fitness, 1 -> eaten_fitness, default -> survival_time_fitness


def survival_time_fitness(sim: PursuitSimulator):
    avg_survival_time = sum(
        x[1] if x is not None else sim.max_moves for x in sim.eaten
    ) / len(sim.eaten)
    return avg_survival_time / sim.max_moves


def eaten_fitness(sim: PursuitSimulator):
    return sim.n_eaten() / len(sim.o_preys)


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
        if FITFN == "1":
            return (eaten_fitness(sim),)
        else:
            return (1 - survival_time_fitness(sim),)
    else:
        pred_routine = gp.compile(opponents[0], pset=opp_psets[0])
        prey_routine = gp.compile(individual, pset=pset)
        sim.run(pred_routine, prey_routine)
        if FITFN == "1":
            return (1 - eaten_fitness(sim),)
        else:
            return (survival_time_fitness(sim),)


def create_predator_pset() -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("PREDATOR", arity=0)
    pset.addPrimitive(lambda *args: partial(progn, *args), 2, name="prog2")
    pset.addPrimitive(lambda *args: partial(progn, *args), 3, name="prog3")
    pset.addPrimitive(
        lambda *args: partial(
            if_enemy_around, *args, type_=Specie.PREDATOR, radius=5.0
        ),
        2,
        name="ifPreyAround",
    )
    pset.addPrimitive(
        lambda *args: partial(if_enemy_left, *args, type_=Specie.PREDATOR, radius=5.0),
        2,
        name="ifPreyLeft",
    )
    pset.addPrimitive(
        lambda *args: partial(if_enemy_right, *args, type_=Specie.PREDATOR, radius=5.0),
        2,
        name="ifPreyRight",
    )
    pset.addPrimitive(
        lambda *args: partial(
            if_enemy_forward, *args, type_=Specie.PREDATOR, radius=5.0
        ),
        2,
        name="ifPreyForward",
    )
    pset.addPrimitive(
        lambda *args: partial(
            if_enemy_behind, *args, type_=Specie.PREDATOR, radius=5.0
        ),
        2,
        name="ifPreyBehind",
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
    pset.addPrimitive(
        lambda *args: partial(if_enemy_left, *args, type_=Specie.PREY, radius=1.0),
        2,
        name="ifPredatorLeft",
    )
    pset.addPrimitive(
        lambda *args: partial(if_enemy_right, *args, type_=Specie.PREY, radius=1.0),
        2,
        name="ifPredatorRight",
    )
    pset.addPrimitive(
        lambda *args: partial(if_enemy_forward, *args, type_=Specie.PREY, radius=1.0),
        2,
        name="ifPredatorForward",
    )
    pset.addPrimitive(
        lambda *args: partial(if_enemy_behind, *args, type_=Specie.PREY, radius=1.0),
        2,
        name="ifPredatorBehind",
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
    if not hasattr(creator, "PredatorFitness"):
        creator.create("PredatorFitness", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Predator"):
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
    if not hasattr(creator, "PreyFitness"):
        creator.create("PreyFitness", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Prey"):
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
