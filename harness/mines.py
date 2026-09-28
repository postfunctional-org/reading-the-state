"""Minesweeper as a memory-free decision benchmark, with an exact solver.

Why this game: the revealed board *is* the complete state. There is nothing to
remember, so a stateless decision model is not being asked to do something its
architecture forbids - which was the whole problem with the roguelike.

And unlike almost any other environment, the ground truth here is an exact
probability. Enumerate every mine configuration consistent with the visible
numbers and the total mine count, weight them, and you get the true P(mine) for
every cell in closed form. That gives:

  * a Choice target with no proxy grading - the provably safe cells, when any
    exist, and otherwise the minimum-risk cell;
  * a Noul target that is a real number rather than an outcome frequency, which
    is a calibration probe almost nothing else offers;
  * a free split between positions that are *deducible* and positions that
    require a *guess*, which are different capabilities and worth scoring apart.

Pure stdlib.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations
from math import comb

HIDDEN, OPEN = 0, 1


def neighbours(x: int, y: int, w: int, h: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                yield nx, ny


class Board:
    """A Minesweeper position. Mines are placed after the first click, as usual."""

    def __init__(self, seed: int, w: int = 9, h: int = 9, n_mines: int = 10):
        self.seed = seed
        self.w, self.h, self.n_mines = w, h, n_mines
        self.rng = random.Random(seed)
        self.mines: set[tuple[int, int]] = set()
        self.state = [[HIDDEN] * w for _ in range(h)]
        self.exploded: tuple[int, int] | None = None
        self.moves = 0
        self._placed = False

    # -- setup -----------------------------------------------------------

    def _place(self, safe: tuple[int, int]) -> None:
        """Place mines avoiding the first click and its neighbourhood (standard rule)."""
        forbidden = {safe} | set(neighbours(*safe, self.w, self.h))
        cells = [(x, y) for y in range(self.h) for x in range(self.w) if (x, y) not in forbidden]
        self.mines = set(self.rng.sample(cells, self.n_mines))
        self._placed = True

    def opening_move(self) -> tuple[int, int]:
        """A deterministic first click, so every episode starts from a real position."""
        return (self.rng.randrange(self.w), self.rng.randrange(self.h))

    # -- queries ---------------------------------------------------------

    def count(self, x: int, y: int) -> int:
        return sum(1 for n in neighbours(x, y, self.w, self.h) if n in self.mines)

    def is_open(self, x: int, y: int) -> bool:
        return self.state[y][x] == OPEN

    def hidden_cells(self) -> list[tuple[int, int]]:
        return [(x, y) for y in range(self.h) for x in range(self.w) if self.state[y][x] == HIDDEN]

    @property
    def lost(self) -> bool:
        return self.exploded is not None

    @property
    def won(self) -> bool:
        return not self.lost and len(self.hidden_cells()) == self.n_mines

    @property
    def done(self) -> bool:
        return self.lost or self.won

    def frontier(self) -> list[tuple[int, int]]:
        """Hidden cells touching at least one open number - where the information is."""
        out = []
        for (x, y) in self.hidden_cells():
            if any(self.is_open(nx, ny) for nx, ny in neighbours(x, y, self.w, self.h)):
                out.append((x, y))
        return out

    def constraints(self) -> list[tuple[list[tuple[int, int]], int]]:
        """Each open number becomes (its hidden neighbours, how many of them are mines)."""
        cons = []
        for y in range(self.h):
            for x in range(self.w):
                if not self.is_open(x, y):
                    continue
                hidden = [n for n in neighbours(x, y, self.w, self.h) if self.state[n[1]][n[0]] == HIDDEN]
                if hidden:
                    cons.append((hidden, self.count(x, y)))
        return cons

    # -- play ------------------------------------------------------------

    def open_cell(self, x: int, y: int) -> list[str]:
        """Open one cell; zeros flood outward. Returns a short event log."""
        # Python's negative indexing silently wraps, so an out-of-range cell would
        # quietly open the opposite edge, miss the mine check and void first-click
        # safety. Refuse rather than corrupt the episode.
        if not (0 <= x < self.w and 0 <= y < self.h):
            raise ValueError(f"cell ({x},{y}) is outside the {self.w}x{self.h} board")
        if not self._placed:
            self._place((x, y))
        if self.state[y][x] == OPEN or self.done:
            return []
        self.moves += 1
        if (x, y) in self.mines:
            self.state[y][x] = OPEN
            self.exploded = (x, y)
            return [f"Cell ({x},{y}) was a mine."]

        opened = 0
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if self.state[cy][cx] == OPEN:
                continue
            self.state[cy][cx] = OPEN
            opened += 1
            if self.count(cx, cy) == 0:
                for nx, ny in neighbours(cx, cy, self.w, self.h):
                    if self.state[ny][nx] == HIDDEN and (nx, ny) not in self.mines:
                        stack.append((nx, ny))
        if opened > 1:
            return [f"Cell ({x},{y}) was empty; {opened} cells opened."]
        return [f"Cell ({x},{y}) shows {self.count(x, y)}."]

    # -- rendering -------------------------------------------------------

    def render(self, reveal_all: bool = False) -> str:
        head = "    " + " ".join(f"{x}" for x in range(self.w))
        rows = [head]
        for y in range(self.h):
            cells = []
            for x in range(self.w):
                if reveal_all and (x, y) in self.mines:
                    cells.append("*")
                elif self.state[y][x] == HIDDEN:
                    cells.append(".")
                else:
                    c = self.count(x, y)
                    cells.append(str(c) if c else "_")
            rows.append(f"{y:2d}  " + " ".join(cells))
        return "\n".join(rows)

    def status(self) -> str:
        hidden = len(self.hidden_cells())
        mines_left = sum(1 for m in self.mines if self.state[m[1]][m[0]] == HIDDEN)
        return (
            f"MINEFIELD {self.w}x{self.h}, {self.n_mines} mines. "
            f"{hidden} cells still hidden, so {hidden - mines_left} safe cells remain. "
            f"Move {self.moves}."
        )


# ---------------------------------------------------------------------------
# exact solver
# ---------------------------------------------------------------------------


@dataclass
class Solution:
    prob: dict[tuple[int, int], float]      # P(mine) per hidden cell
    outside_prob: float                      # P(mine) for cells touching no number
    safe: list[tuple[int, int]]              # cells with P == 0
    mines_certain: list[tuple[int, int]]     # cells with P == 1
    total_configs: float                     # weighted count of consistent worlds
    components: int
    frontier: list[tuple[int, int]] = field(default_factory=list)

    @property
    def deducible(self) -> bool:
        """Is there a provably safe cell? If not, the position requires a guess."""
        return bool(self.safe)


def _enumerate_component(cells, cons):
    """All consistent mine assignments for one independent block of constraints.

    Returns {mine_count: (n_assignments, {cell: n_assignments_where_mine})}.
    """
    idx = {c: i for i, c in enumerate(cells)}
    packed = [([idx[c] for c in cs], k) for cs, k in cons]
    n = len(cells)
    by_count: dict[int, list[int]] = {}
    tally: dict[int, dict[int, int]] = {}

    assign = [-1] * n

    def feasible() -> bool:
        for ids, k in packed:
            lo = sum(1 for i in ids if assign[i] == 1)
            unknown = sum(1 for i in ids if assign[i] == -1)
            if lo > k or lo + unknown < k:
                return False
        return True

    def rec(i: int, used: int):
        if not feasible():
            return
        if i == n:
            by_count[used] = by_count.get(used, 0) + 1
            row = tally.setdefault(used, {})
            for j in range(n):
                if assign[j] == 1:
                    row[j] = row.get(j, 0) + 1
            return
        for v in (0, 1):
            assign[i] = v
            rec(i + 1, used + v)
        assign[i] = -1

    rec(0, 0)
    return cells, by_count, tally


def solve(board: Board) -> Solution:
    """Exact per-cell mine probabilities for the current position."""
    if board.lost:
        # An opened mine makes the revealed numbers unsatisfiable, so every
        # consistent world is eliminated and the weights collapse to zero. The
        # answer would be fabricated; say so rather than return it.
        raise ValueError("cannot solve a lost board: the revealed numbers are unsatisfiable")
    hidden = board.hidden_cells()
    cons = board.constraints()
    front = board.frontier()
    outside = [c for c in hidden if c not in set(front)]
    remaining = board.n_mines

    if not cons or not front:
        p = remaining / len(hidden) if hidden else 0.0
        probs = {c: p for c in hidden}
        eps = 1e-12
        return Solution(
            probs, p,
            sorted([c for c in hidden if p < eps]),
            sorted([c for c in hidden if p > 1 - eps]),
            1.0, 0, front,
        )

    # split constraints into independent components (they share no cells)
    parent = {c: c for c in front}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for cells, _k in cons:
        for c in cells[1:]:
            union(cells[0], c)

    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for c in front:
        groups.setdefault(find(c), []).append(c)
    comp_cons: dict[tuple[int, int], list] = {g: [] for g in groups}
    for cells, k in cons:
        comp_cons[find(cells[0])].append((cells, k))

    comps = [_enumerate_component(cells, comp_cons[g]) for g, cells in groups.items()]
    n_out = len(outside)

    # convolve the components' mine-count distributions
    def convolve(dists):
        acc = {0: 1}
        for d in dists:
            nxt: dict[int, int] = {}
            for a, wa in acc.items():
                for b, wb in d.items():
                    nxt[a + b] = nxt.get(a + b, 0) + wa * wb
            acc = nxt
        return acc

    all_dists = [c[1] for c in comps]
    total_dist = convolve(all_dists)

    def outside_ways(t: int) -> int:
        left = remaining - t
        if left < 0 or left > n_out:
            return 0
        return comb(n_out, left)

    total_w = sum(w * outside_ways(t) for t, w in total_dist.items())
    if total_w == 0:
        # No mine layout satisfies the revealed numbers and the mine count at
        # once, so there is no distribution to report. Returning a uniform value
        # here would be a fabricated answer presented as an exact one.
        raise ValueError(
            "position is infeasible: no mine layout satisfies the revealed numbers "
            f"together with {remaining} remaining mines"
        )

    prob: dict[tuple[int, int], float] = {}
    for j, (cells, by_count, tally) in enumerate(comps):
        others = convolve([all_dists[i] for i in range(len(comps)) if i != j])
        for local_idx, cell in enumerate(cells):
            wmine = 0
            for kj, row in tally.items():
                cnt = row.get(local_idx, 0)
                if not cnt:
                    continue
                for ko, wo in others.items():
                    wmine += cnt * wo * outside_ways(kj + ko)
            prob[cell] = wmine / total_w

    exp_outside = sum(w * outside_ways(t) * (remaining - t) for t, w in total_dist.items()) / total_w
    p_out = (exp_outside / n_out) if n_out else 0.0
    for c in outside:
        prob[c] = p_out

    eps = 1e-12
    safe = sorted([c for c in hidden if prob[c] < eps])
    certain = sorted([c for c in hidden if prob[c] > 1 - eps])
    return Solution(prob, p_out, safe, certain, float(total_w), len(comps), front)


# ---------------------------------------------------------------------------
# verification: exact solver vs exhaustive enumeration of whole boards
# ---------------------------------------------------------------------------


def brute_force(board: Board) -> dict[tuple[int, int], float]:
    """Ground truth by enumerating every legal mine placement. Small boards only."""
    hidden = board.hidden_cells()
    opened = [(x, y) for y in range(board.h) for x in range(board.w) if board.is_open(x, y)]
    tally = {c: 0 for c in hidden}
    total = 0
    for placement in combinations(hidden, board.n_mines):
        ms = set(placement)
        ok = True
        for (x, y) in opened:
            if sum(1 for n in neighbours(x, y, board.w, board.h) if n in ms) != board.count(x, y):
                ok = False
                break
        if not ok:
            continue
        total += 1
        for c in placement:
            tally[c] += 1
    if not total:
        return {c: 0.0 for c in hidden}
    return {c: tally[c] / total for c in hidden}


if __name__ == "__main__":
    import time

    print("cross-checking the exact solver against exhaustive enumeration")
    worst, checked, boards = 0.0, 0, 0
    t0 = time.time()
    for seed in range(60):
        b = Board(seed, w=5, h=5, n_mines=4)
        b.open_cell(*b.opening_move())
        for _ in range(3):
            if b.done:
                break
            sol = solve(b)
            bf = brute_force(b)
            for c in b.hidden_cells():
                worst = max(worst, abs(sol.prob[c] - bf[c]))
                checked += 1
            boards += 1
            # advance by opening the safest cell
            pick = min(b.hidden_cells(), key=lambda c: sol.prob[c])
            b.open_cell(*pick)
    print(f"  {boards} positions, {checked} cell probabilities compared")
    print(f"  max |solver - brute force| = {worst:.3e}   ({time.time()-t0:.1f}s)")
    assert worst < 1e-9, "solver disagrees with brute force"
    print("  OK")
