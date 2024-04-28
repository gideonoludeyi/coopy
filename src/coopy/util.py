import itertools
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


def dist(pos1: tuple[float, float], pos2: tuple[float, float], nrows: int, ncols: int):
    x1, y1 = pos1
    x2, y2 = pos2
    # manhattan distance that accounts for a looping arena
    xdist = min(abs(x2 - x1), abs((ncols - x2) + x1), abs((ncols - x1) + x2))
    ydist = min(abs(y2 - y1), abs((nrows - y2) + y1), abs((nrows - y1) + y2))
    return xdist + ydist


def normalize(value):
    if value:
        return value / abs(value)
    else:
        return value
