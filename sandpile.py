"""
Arbelian sandpile model

https://en.wikipedia.org/wiki/Abelian_sandpile_model


Usage:

    * Run 100 steps on the default 10 x 10 board, with verbose logging:

          python sandpile.py --steps 100 --log-level debug

    * Run 1,000 steps on a larger board:

          python sandpile.py -s 1000 -r 20 -c 20

    * Output to PNG

          python sandpile.py -s 1000 -r 50 -c 50 --format png

    * Output to GIF

          python sandpile.py -s 1000 -r 18 -c 18 --format gif

    python sandpile.py --help

Dependencies:
* numpy
* Pillow for PNG and GIF output format
"""

import math
import logging
import random
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import chain
from multiprocessing import Pool
from time import perf_counter
from typing import Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImagePalette

GRID_ROWS = 10
GRID_COLUMNS = 10


@dataclass
class Coords:
    row: int
    column: int


class Board:
    cells: np.ndarray

    def __init__(self, cells: np.array):
        self.cells = cells

    @classmethod
    def new(cls, rows: int = GRID_ROWS, columns: int = GRID_COLUMNS):
        """
        Create a new board of the specified size, with all
        cells initialized to zero.
        """
        cells = np.zeros((rows, columns), dtype=int)
        return cls(cells)

    def __iter__(self):
        for row in self.cells:
            for cell in row:
                yield cell

    @property
    def limits(self) -> Tuple[int, int]:
        return self.cells.shape

    def is_on_board(self, coords: Coords) -> bool:
        row_count, column_count = self.limits
        return (
            coords.row >= 0
            and coords.column >= 0
            and coords.row < row_count
            and coords.column < column_count
        )

    def get_value(self, coords: Coords) -> np.int64:
        if not self.is_on_board(coords):
            raise IndexError(f"{coords} not on board")
        return self.cells[coords.row][coords.column]

    def set_value(self, coords: Coords, value: int):
        if not self.is_on_board(coords):
            raise IndexError(f"{coords} not on board")
        logging.debug("Setting %s to %s", coords, value)
        self.cells[coords.row][coords.column] = value

    def incr(self, coords: Coords, by=1) -> int:
        """
        Increment the cell at the given coords, by the specified amount.

        Returns the new value.
        """
        current = self.get_value(coords)
        new = current + by
        self.set_value(coords, new)
        return new

    def get_critical_coords(self, threshold: int) -> List[Coords]:
        """
        Returns a list of cell coordinates where the cell value
        is greater than or equal to the given threshold.
        """
        coords = []
        for row, column in np.argwhere(self.cells >= threshold):
            coords.append(Coords(row=row, column=column))
        return coords

    def copy(self):
        return self.__class__(cells=self.cells.copy())

    def __repr__(self):
        row_count, column_count = self.limits
        return f"{self.__class__.__name__}<{row_count}, {column_count}>"

    def __str__(self):
        return repr(self.cells)


class Algorithm:
    critical_point = 4
    board: Board
    steps = 0
    topples = 0

    def __init__(self, board: Optional[Board] = None):
        self.board = board if board else Board.new()

    def step(self):
        """
        Add sand to random coordinates, and resolve the board
        by toppling any critical cells.
        """
        coords = self.get_random_coords()
        logging.debug("Chose random coords %s", coords)
        self.board.incr(coords)
        self.resolve()
        self.steps += 1

    def get_random_coords(self) -> Coords:
        row_count, column_count = self.board.limits
        row = math.floor(random.uniform(0, row_count))
        column = math.floor(random.uniform(0, column_count))
        return Coords(row=row, column=column)

    def resolve(self):
        """
        Topple all critical cells, until the board is stable.
        """
        critical_cells = self.board.get_critical_coords(threshold=self.critical_point)
        while len(critical_cells) > 0:
            logging.debug("critical cells: %s", critical_cells)
            for coords in critical_cells:
                self.topple_cell(coords)
            critical_cells = self.board.get_critical_coords(
                threshold=self.critical_point
            )

    def topple_cell(self, coords: Coords):
        """
        Topple the cell at the given location.

        1. Reduce its slope by 4
        2. Add 1 to all the neighbors
        """
        logging.debug("toppling cell at %s", coords)
        slope = self.board.get_value(coords)
        if slope < self.critical_point:
            raise ValueError("Cell at {coords} is not critical")
        self.board.set_value(coords, slope - 4)
        for neigbor in self.get_cell_neighbors(coords):
            self.board.incr(neigbor)
        self.topples += 1

    def get_cell_neighbors(self, coords: Coords) -> Iterator[Coords]:
        translations = (
            (-1, 0),
            (0, 1),
            (1, 0),
            (0, -1),
        )

        for d_row, d_column in translations:
            neighbor = Coords(row=coords.row + d_row, column=coords.column + d_column)

            if self.board.is_on_board(neighbor):
                yield neighbor


@dataclass
class Frame:
    number: int
    board: Board
    topples: int


def output_text(_, frames: List[Frame]):
    logging.debug("rendering text")
    print(frames[-1].board)


# https://www.colourlovers.com/palette/292482/Terra
COLOR_MAP = {
    3: (232, 221, 203),
    2: (205, 179, 128),
    1: (3, 101, 100),
    0: (3, 22, 52),
}


def render_png(board: Board) -> Image:
    SCALE_FACTOR = 10
    logging.debug("rendering PNG")

    rows, cols = board.limits
    img_data = np.zeros((rows * SCALE_FACTOR, cols * SCALE_FACTOR, 3), dtype=np.uint8)
    for (row, col), value in np.ndenumerate(board.cells):
        out_row = row * SCALE_FACTOR
        out_col = col * SCALE_FACTOR
        color = COLOR_MAP.get(value) or COLOR_MAP[0]
        for i in range(SCALE_FACTOR):
            for j in range(SCALE_FACTOR):
                img_data[out_row + i][out_col + j] = color
    return Image.fromarray(img_data, "RGB")


def output_png(filename, frames: List[Frame]):
    board = frames[-1].board
    image = render_png(board)
    image.save(f"{filename}.png")
    logging.info("Image saved to board.png")
    print(board)


def output_gif(filename, frames: List[Frame]):
    TOTAL_SECONDS = 20
    duration = TOTAL_SECONDS * 1000 / len(frames)

    with Pool() as pool:
        images = pool.map(render_png, [frame.board for frame in frames])

    first, rest = images[0], images[1:]
    palette = ImagePalette.ImagePalette(
        size=len(COLOR_MAP), palette=list(chain(COLOR_MAP.values()))
    )
    first.save(
        f"{filename}.gif",
        save_all=True,
        append_images=rest,
        duration=duration,
        palette=palette,
        interlace=True,
        optimize=False,
        loop=1,
    )


OUTPUT_FUNCTIONS = {
    "text": output_text,
    "png": output_png,
    "gif": output_gif,
}


def main(steps=10, columns=GRID_COLUMNS, rows=GRID_ROWS, format="text"):
    board = Board.new(rows=rows, columns=columns)
    algo = Algorithm(board=board)
    render_function = OUTPUT_FUNCTIONS[format]
    frames: List[Frame] = []

    start = perf_counter()
    for i in range(steps):
        algo.step()
        frames.append(Frame(number=i, board=algo.board.copy(), topples=algo.topples))
    end = perf_counter()

    render_start = perf_counter()
    render_function(f"board_steps_{steps}_rows_{rows}_cols_{columns}", frames)
    render_end = perf_counter()

    print(f"Steps={algo.steps}")
    print(f"Topples={algo.topples}")
    print(f"Seconds={end - start}")
    render_seconds = render_end - render_start
    if render_seconds > 1:
        print(f"Render seconds: {render_seconds}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--steps", default=100, type=int)
    parser.add_argument("-r", "--rows", default=GRID_ROWS, type=int)
    parser.add_argument("-c", "--columns", default=GRID_COLUMNS, type=int)
    parser.add_argument("-l", "--log-level", default="info", type=str)
    parser.add_argument(
        "-f", "--format", default="text", choices=OUTPUT_FUNCTIONS.keys()
    )

    args = parser.parse_args()

    loglevel = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=loglevel)

    main(steps=args.steps, rows=args.rows, columns=args.columns, format=args.format)
