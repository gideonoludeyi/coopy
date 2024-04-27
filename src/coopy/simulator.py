from enum import Enum, auto
from itertools import cycle
from typing import Callable

from .util import dist


class Specie(Enum):
    PREDATOR = auto()
    PREY = auto()


Updater = Callable[[tuple[float, float, complex]], tuple[float, float, complex]]


class PursuitSimulator:
    def __init__(
        self,
        predators: list[tuple[float, float, complex]],
        preys: list[tuple[float, float, complex]],
        nrows: int,
        ncols: int,
        max_moves: int = 600,
    ) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.o_predators = predators
        self.o_preys = preys
        self.predators = list(self.o_predators)
        self.preys = list(self.o_preys)
        self.eaten: list[tuple[tuple[float, float, complex], int] | None] = [
            None
        ] * len(self.preys)
        self.n_moves = 0
        self.max_moves = max_moves

    def is_around(
        self, reftype: Specie, refidx: int, target_specie: Specie, radius: float
    ):
        if reftype == Specie.PREDATOR:
            ref = self.predators[refidx]
        else:
            ref = self.preys[refidx]
        if target_specie == Specie.PREDATOR:
            entities = self.predators
        else:
            entities = self.preys
        return any(e for e in entities if dist(ref[:2], e[:2]) <= radius)

    def update(self, reftype: Specie, refidx: int, update: Updater):
        if reftype == Specie.PREDATOR:
            self.predators[refidx] = update(self.predators[refidx])
            for i, prey in enumerate(self.preys):
                if dist(prey[:2], self.predators[refidx][:2]) <= 1.0:
                    self.eaten[i] = (prey, self.n_moves)
        else:
            self.preys[refidx] = update(self.preys[refidx])

    def run(self, pred_routine, prey_routine):
        self.predators = list(self.o_predators)
        self.preys = list(self.o_preys)
        self.eaten = [None] * len(self.preys)
        self.n_moves = 0
        predators = [
            cycle(pred_routine(sim=self, index=i)) for i in range(len(self.predators))
        ]
        preys = [cycle(prey_routine(sim=self, index=i)) for i in range(len(self.preys))]
        while len(self.preys) > 0 and self.n_moves < self.max_moves:
            for routine_iter in predators:
                next(routine_iter)
            for i, routine_iter in enumerate(preys):
                if self.eaten[i] is None:
                    next(routine_iter)
            self.n_moves += 1


def progn(*outs, sim: PursuitSimulator, index: int):
    for out in outs:
        yield from out(sim=sim, index=index)


def if_enemy_around(
    true_action,
    false_action,
    *,
    sim: PursuitSimulator,
    type_: Specie,
    index: int,
    radius: float = 1.0,
):
    enemy_type = Specie.PREDATOR if type_ == Specie.PREY else Specie.PREY
    is_ahead = sim.is_around(type_, index, enemy_type, radius=radius)
    if is_ahead:
        yield from true_action(sim=sim, index=index)
    else:
        yield from false_action(sim=sim, index=index)


def forward(*, sim: PursuitSimulator, type_: Specie, index: int):
    def move_forward(state: tuple[float, float, complex]):
        (x, y, d) = state
        newx = (x + d.real) % sim.ncols
        newy = (y + d.imag) % sim.nrows
        return newx, newy, d

    # print(f"{type_} :: FORWARD")
    sim.update(type_, index, move_forward)
    yield


def backward(*, sim: PursuitSimulator, type_: Specie, index: int):
    def move_backward(state: tuple[float, float, complex]):
        (x, y, d) = state
        newx = (x - d.real) % sim.ncols
        newy = (y - d.imag) % sim.nrows
        return newx, newy, d

    # print(f"{type_} :: BACKWARD")
    sim.update(type_, index, move_backward)
    yield


def left(*, sim: PursuitSimulator, type_: Specie, index: int):
    def turn_left(state):
        return state[0], state[1], state[2] * -1j

    # print(f"{type_} :: LEFT")
    sim.update(type_, index, turn_left)
    yield


def right(*, sim: PursuitSimulator, type_: Specie, index: int):
    def turn_right(state):
        return state[0], state[1], state[2] * 1j

    # print(f"{type_} :: RIGHT")
    sim.update(type_, index, turn_right)
    yield
