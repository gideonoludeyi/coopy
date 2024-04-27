import itertools
import math
import os
import random


def parse_str(text: str):
    pred_icons = {
        "^": 0 - 1j,  # up
        ">": 1 + 0j,  # right
        "v": 0 + 1j,  # down
        "<": -1 + 0j,  # left
    }
    prey_icon = "#"

    assert (
        sum(map(text.count, pred_icons.keys())) >= 1
    ), f"at least one of {list(pred_icons.keys())} must be present in the environment"
    grid = [list(line) for line in map(str.strip, text.split("\n")) if line != ""]
    nrows = len(grid)
    ncols = len(grid[0]) if nrows > 0 else 0
    preds: list[tuple[float, float, complex]] = []
    for row, col in itertools.product(range(nrows), range(ncols)):
        if grid[row][col] in pred_icons.keys():
            preds.append((row, col, pred_icons[grid[row][col]]))
    preys = [
        (row, col, random.choice(list(pred_icons.values())))
        for row in range(nrows)
        for col in range(ncols)
        if grid[row][col] == prey_icon
    ]
    return dict(
        nrows=nrows,
        ncols=ncols,
        preds=preds,
        preys=preys,
    )


def parse_file(filepath: str | os.PathLike):
    with open(filepath, "r") as file:
        return parse_str(file.read())


def dist(pos1: tuple[float, float], pos2: tuple[float, float]):
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
