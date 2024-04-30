from enum import Enum, auto
from typing import Callable

from .util import dist, normalize


class Specie(Enum):
    """specifies the species represented in the problem"""

    PREDATOR = auto()
    PREY = auto()


# type alias for functions that transform a position/orientation
Updater = Callable[[tuple[float, float, complex]], tuple[float, float, complex]]


class PursuitSimulator:
    """simulator that runs the predator and prey programs in a simulation"""

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
        self.o_predators = predators  # stores the original predators positions
        self.o_preys = preys  # stores the original prey positions
        self.predators = list(self.o_predators)
        self.preys = list(self.o_preys)
        self.eaten: list[tuple[tuple[float, float, complex], int] | None] = [
            None
        ] * len(self.preys)
        self.n_moves = 0
        self.max_moves = max_moves

    def is_around(
        self, reftype: Specie, refidx: int, target_specie: Specie, radius: float
    ) -> bool:
        """determines whether a member of the `target_specie` is within the given radius of the reference individual"""
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
        """determines whether a member of the `target_specie` is within radius units ahead of the reference individual"""
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
        """determines whether a member of the `target_specie` is within radius units to the left of the reference individual"""
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
        """determines whether a member of the `target_specie` is within radius units to the right of the reference individual"""
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
        """determines whether a member of the `target_specie` is within radius units behind the reference individual"""
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
        """updates the reference individuals position/direction"""
        if reftype == Specie.PREDATOR:
            self.predators[refidx] = update(self.predators[refidx])
            # marks a prey as eaten if it overlaps with the predator's new position
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
        """computes the number of eaten prey"""
        return len([prey for prey in self.eaten if prey is not None])

    def run(self, pred_routine, prey_routine):
        """run the simulation"""
        list(self.run_iter(pred_routine, prey_routine))

    def cycle(self, iter_fn, *args, **kwargs):
        """helper function to repeatedly run a gp program tree over and over"""
        while True:
            yield from iter_fn(*args, **kwargs)

    def run_iter(self, pred_routine, prey_routine):
        """run the simulation, yielding control back to the caller after each time step"""
        # reset simulation state
        self.predators = list(self.o_predators)
        self.preys = list(self.o_preys)
        self.eaten = [None] * len(self.preys)
        self.n_moves = 0
        # generate predator brains from the specified routine
        predators = [
            self.cycle(pred_routine, sim=self, index=i)
            for i in range(len(self.predators))
        ]
        # generate prey brains from the specified routine
        preys = [
            self.cycle(prey_routine, sim=self, index=i) for i in range(len(self.preys))
        ]
        # run the simulation until all preys are eaten or the maximum number of moves is reached
        while self.n_eaten() < len(self.preys) and self.n_moves < self.max_moves:
            # perform predators' next action
            for routine_iter in predators:
                next(routine_iter)
            # perform preys' next action if they're still alive
            for i, routine_iter in enumerate(preys):
                if self.eaten[i] is None:
                    next(routine_iter)
            self.n_moves += 1
            yield  # return control back to the caller


def progn(*outs, sim: PursuitSimulator, index: int):
    """runs the `outs` arguments sequentially"""
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
    """runs the `true_action` if a member of the opposing specie is within `radius` units around the current entity
    otherwise run `false_action`
    """
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
    """runs the `true_action` if a member of the opposing specie is within `radius` units ahead of the current entity
    otherwise run `false_action`
    """
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
    """runs the `true_action` if a member of the opposing specie is within `radius` units to the left of the current entity
    otherwise run `false_action`
    """
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
    """runs the `true_action` if a member of the opposing specie is within `radius` units to the right of the current entity
    otherwise run `false_action`
    """
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
    """runs the `true_action` if a member of the opposing specie is within `radius` units behind of the current entity
    otherwise run `false_action`
    """
    enemy_type = Specie.PREDATOR if type_ == Specie.PREY else Specie.PREY
    is_ahead = sim.is_behind(type_, index, enemy_type, radius=radius)
    if is_ahead:
        yield from true_action(sim=sim, index=index)
    else:
        yield from false_action(sim=sim, index=index)


def forward(*, sim: PursuitSimulator, type_: Specie, index: int):
    """advances the current entity forward, wrapping around the map if it hits the edge"""

    def move_forward(state: tuple[float, float, complex]):
        (row, col, d) = state
        newx = (row + d.imag) % sim.nrows
        newy = (col + d.real) % sim.ncols
        return newx, newy, d

    sim.update(type_, index, move_forward)
    yield


def backward(*, sim: PursuitSimulator, type_: Specie, index: int):
    """moves the current entity backward, wrapping around the map if it hits the edge"""

    def move_backward(state: tuple[float, float, complex]):
        (row, col, d) = state
        newx = (row - d.imag) % sim.nrows
        newy = (col - d.real) % sim.ncols
        return newx, newy, d

    sim.update(type_, index, move_backward)
    yield


def left(*, sim: PursuitSimulator, type_: Specie, index: int):
    """turns the current entity to the left"""

    def turn_left(state):
        return (
            state[0],
            state[1],
            state[2] * -1j,  # complex number arithmetic (-1j rotates to the left)
        )

    sim.update(type_, index, turn_left)
    yield


def right(*, sim: PursuitSimulator, type_: Specie, index: int):
    """turns the current entity to the right"""

    def turn_right(state):
        return (
            state[0],
            state[1],
            state[2] * 1j,  # complex number arithmetic (1j rotates to the right)
        )

    sim.update(type_, index, turn_right)
    yield
