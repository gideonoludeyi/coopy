from enum import Enum, auto
from typing import Callable

from .util import dist, normalize


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
        max_moves: int,
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
            entities = [
                prey for i, prey in enumerate(self.preys) if self.eaten[i] is None
            ]
        return any(
            e
            for e in entities
            if dist(ref[:2], e[:2], nrows=self.nrows, ncols=self.ncols) <= radius
        )

    def is_forward(
        self, reftype: Specie, refidx: int, target_specie: Specie, radius: float
    ):
        if reftype == Specie.PREDATOR:
            ref = self.predators[refidx]
        else:
            ref = self.preys[refidx]
        if target_specie == Specie.PREDATOR:
            entities = self.predators
        else:
            entities = [
                prey for i, prey in enumerate(self.preys) if self.eaten[i] is None
            ]

        def predicate(e):
            distance = dist(ref[:2], e[:2], nrows=self.nrows, ncols=self.ncols)
            direction = complex(
                normalize(e[0] - ref[0]),
                normalize(e[1] - ref[1]),
            )
            return distance <= radius and direction == ref[2]

        return any(e for e in entities if predicate(e))

    def is_left(
        self, reftype: Specie, refidx: int, target_specie: Specie, radius: float
    ):
        if reftype == Specie.PREDATOR:
            ref = self.predators[refidx]
        else:
            ref = self.preys[refidx]
        if target_specie == Specie.PREDATOR:
            entities = self.predators
        else:
            entities = [
                prey for i, prey in enumerate(self.preys) if self.eaten[i] is None
            ]

        def predicate(e):
            distance = dist(ref[:2], e[:2], nrows=self.nrows, ncols=self.ncols)
            direction = complex(
                normalize(e[0] - ref[0]),
                normalize(e[1] - ref[1]),
            )
            return distance <= radius and direction == (ref[2] * -1j)

        return any(e for e in entities if predicate(e))

    def is_right(
        self, reftype: Specie, refidx: int, target_specie: Specie, radius: float
    ):
        if reftype == Specie.PREDATOR:
            ref = self.predators[refidx]
        else:
            ref = self.preys[refidx]
        if target_specie == Specie.PREDATOR:
            entities = self.predators
        else:
            entities = [
                prey for i, prey in enumerate(self.preys) if self.eaten[i] is None
            ]

        def predicate(e):
            distance = dist(ref[:2], e[:2], nrows=self.nrows, ncols=self.ncols)
            direction = complex(
                normalize(e[0] - ref[0]),
                normalize(e[1] - ref[1]),
            )
            return distance <= radius and direction == (ref[2] * 1j)

        return any(e for e in entities if predicate(e))

    def is_behind(
        self, reftype: Specie, refidx: int, target_specie: Specie, radius: float
    ):
        if reftype == Specie.PREDATOR:
            ref = self.predators[refidx]
        else:
            ref = self.preys[refidx]
        if target_specie == Specie.PREDATOR:
            entities = self.predators
        else:
            entities = [
                prey for i, prey in enumerate(self.preys) if self.eaten[i] is None
            ]

        def predicate(e):
            distance = dist(ref[:2], e[:2], nrows=self.nrows, ncols=self.ncols)
            direction = complex(
                normalize(e[0] - ref[0]),
                normalize(e[1] - ref[1]),
            )
            return distance <= radius and direction == (ref[2] * -1)

        return any(e for e in entities if predicate(e))

    def update(self, reftype: Specie, refidx: int, update: Updater):
        if reftype == Specie.PREDATOR:
            self.predators[refidx] = update(self.predators[refidx])
            for i, prey in enumerate(self.preys):
                if (
                    dist(
                        prey[:2],
                        self.predators[refidx][:2],
                        nrows=self.nrows,
                        ncols=self.ncols,
                    )
                    <= 1.0
                ):
                    self.eaten[i] = (prey, self.n_moves)
        else:
            self.preys[refidx] = update(self.preys[refidx])

    def n_eaten(self):
        return len([prey for prey in self.eaten if prey is not None])

    def run(self, pred_routine, prey_routine):
        list(self.run_iter(pred_routine, prey_routine))

    def cycle(self, iter_fn, *args, **kwargs):
        while True:
            yield from iter_fn(*args, **kwargs)

    def run_iter(self, pred_routine, prey_routine):
        self.predators = list(self.o_predators)
        self.preys = list(self.o_preys)
        self.eaten = [None] * len(self.preys)
        self.n_moves = 0
        predators = [
            self.cycle(pred_routine, sim=self, index=i)
            for i in range(len(self.predators))
        ]
        preys = [
            self.cycle(prey_routine, sim=self, index=i) for i in range(len(self.preys))
        ]
        while self.n_eaten() < len(self.preys) and self.n_moves < self.max_moves:
            for routine_iter in predators:
                next(routine_iter)
            for i, routine_iter in enumerate(preys):
                if self.eaten[i] is None:
                    next(routine_iter)
            self.n_moves += 1
            yield


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


def if_enemy_forward(
    true_action,
    false_action,
    *,
    sim: PursuitSimulator,
    type_: Specie,
    index: int,
    radius: float = 1.0,
):
    enemy_type = Specie.PREDATOR if type_ == Specie.PREY else Specie.PREY
    is_ahead = sim.is_forward(type_, index, enemy_type, radius=radius)
    if is_ahead:
        yield from true_action(sim=sim, index=index)
    else:
        yield from false_action(sim=sim, index=index)


def if_enemy_left(
    true_action,
    false_action,
    *,
    sim: PursuitSimulator,
    type_: Specie,
    index: int,
    radius: float = 1.0,
):
    enemy_type = Specie.PREDATOR if type_ == Specie.PREY else Specie.PREY
    is_ahead = sim.is_left(type_, index, enemy_type, radius=radius)
    if is_ahead:
        yield from true_action(sim=sim, index=index)
    else:
        yield from false_action(sim=sim, index=index)


def if_enemy_right(
    true_action,
    false_action,
    *,
    sim: PursuitSimulator,
    type_: Specie,
    index: int,
    radius: float = 1.0,
):
    enemy_type = Specie.PREDATOR if type_ == Specie.PREY else Specie.PREY
    is_ahead = sim.is_right(type_, index, enemy_type, radius=radius)
    if is_ahead:
        yield from true_action(sim=sim, index=index)
    else:
        yield from false_action(sim=sim, index=index)


def if_enemy_behind(
    true_action,
    false_action,
    *,
    sim: PursuitSimulator,
    type_: Specie,
    index: int,
    radius: float = 1.0,
):
    enemy_type = Specie.PREDATOR if type_ == Specie.PREY else Specie.PREY
    is_ahead = sim.is_behind(type_, index, enemy_type, radius=radius)
    if is_ahead:
        yield from true_action(sim=sim, index=index)
    else:
        yield from false_action(sim=sim, index=index)


def forward(*, sim: PursuitSimulator, type_: Specie, index: int):
    def move_forward(state: tuple[float, float, complex]):
        (row, col, d) = state
        newx = (row + d.imag) % sim.nrows
        newy = (col + d.real) % sim.ncols
        return newx, newy, d

    # print(f"{type_} :: FORWARD \t | {sim.n_moves}")
    sim.update(type_, index, move_forward)
    yield


def backward(*, sim: PursuitSimulator, type_: Specie, index: int):
    def move_backward(state: tuple[float, float, complex]):
        (row, col, d) = state
        newx = (row - d.imag) % sim.nrows
        newy = (col - d.real) % sim.ncols
        return newx, newy, d

    # print(f"{type_} :: BACKWARD \t | {sim.n_moves}")
    sim.update(type_, index, move_backward)
    yield


def left(*, sim: PursuitSimulator, type_: Specie, index: int):
    def turn_left(state):
        return state[0], state[1], state[2] * -1j

    # print(f"{type_} :: LEFT \t | {sim.n_moves}")
    sim.update(type_, index, turn_left)
    yield


def right(*, sim: PursuitSimulator, type_: Specie, index: int):
    def turn_right(state):
        return state[0], state[1], state[2] * 1j

    # print(f"{type_} :: RIGHT \t | {sim.n_moves}")
    sim.update(type_, index, turn_right)
    yield
