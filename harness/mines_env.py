"""The Minesweeper episode, its action set, and its exact ground truth.

One turn = one System One request carrying eight questions about the same board,
which is exactly the native physical batch size:

  open   (Choice)  - which hidden cell to open. Candidates are the frontier cells,
                     rebuilt every turn. This drives play.
  safe_i (Noul x6) - "the cell at column X row Y does not contain a mine", asked
                     about up to six probe cells spanning the true-probability
                     range. Graded against 1 - P(mine), which the solver knows
                     exactly. This is the calibration probe, and it doubles as a
                     test of the documented "one context, many questions" claim.
  risk   (Score)   - a five-level rubric for how dangerous the board is.

Only `open` steers the game. The rest are recorded.
"""

from __future__ import annotations

import random

from mines import Board, neighbours, solve

RISK_LEVELS = [
    "Safe. Every useful move here is provably free of mines.",
    "Mostly solved. There is a safe move, and it is easy to see.",
    "Tight. A safe move exists but the deduction is awkward.",
    "Forced gamble. No move is provably safe; the best is still a risk.",
    "Bad gamble. Every remaining move is more likely than not to be a mine.",
]

INSTRUCTIONS = (
    "Minesweeper. A digit counts the mines touching that square, '_' means none, "
    "'.' is hidden. Choose the hidden square least likely to be a mine; if one can be "
    "proved safe from the numbers, choose that."
)

N_PROBES = 6


def cell_name(c) -> str:
    return f"c{c[0]}r{c[1]}"


def describe(board: Board, c) -> str:
    """What a player reads off the board about this cell - no solving done for them.

    Deliberately terse. Every candidate description is rendered into every question,
    so on a 20-candidate board a wordy format costs over 2000 tokens and locks the
    1024-token encoder models out of the experiment entirely.
    """
    x, y = c
    nums = sorted(
        board.count(nx, ny)
        for nx, ny in neighbours(x, y, board.w, board.h)
        if board.is_open(nx, ny)
    )
    hidden_touch = sum(
        1 for nx, ny in neighbours(x, y, board.w, board.h) if not board.is_open(nx, ny)
    )
    touching = ",".join(str(n) for n in nums) if nums else "none"
    return f"col {x} row {y}, touching {touching}, {hidden_touch} hidden"


class MinesEpisode:
    """One game. Deterministic given the seed and the sequence of opened cells."""

    def __init__(self, seed: int, w: int = 9, h: int = 9, n_mines: int = 20, max_moves: int = 80):
        self.seed = seed
        self.board = Board(seed, w, h, n_mines)
        self.board.open_cell(*self.board.opening_move())
        self.rng = random.Random(seed * 7919 + 13)
        self.max_moves = max_moves
        self.opened_safely = 0
        self._sol = None
        self._sol_key = None

    # -- solver cache ----------------------------------------------------

    @property
    def solution(self):
        key = (self.board.moves, len(self.board.hidden_cells()))
        if self._sol_key != key:
            self._sol = solve(self.board)
            self._sol_key = key
        return self._sol

    @property
    def done(self) -> bool:
        return self.board.done or self.board.moves >= self.max_moves

    @property
    def result(self) -> str:
        if self.board.won:
            return "cleared"
        if self.board.lost:
            return "exploded"
        return "timeout"

    # -- what the model sees --------------------------------------------

    def candidates(self) -> list:
        front = self.solution.frontier
        if front:
            return sorted(front)
        return sorted(self.board.hidden_cells())

    def probe_cells(self) -> list:
        """Up to six frontier cells spanning the true-probability range."""
        sol = self.solution
        cands = self.candidates()
        if not cands:
            return []
        ranked = sorted(cands, key=lambda c: sol.prob[c])
        if len(ranked) <= N_PROBES:
            return ranked
        # even spread across the sorted-by-risk list, so probabilities are not all 0
        step = (len(ranked) - 1) / (N_PROBES - 1)
        picked, seen = [], set()
        for i in range(N_PROBES):
            c = ranked[int(round(i * step))]
            if c not in seen:
                seen.add(c)
                picked.append(c)
        return picked

    def state_text(self) -> str:
        b = self.board
        sol = self.solution
        parts = [b.status(), "", b.render()]
        return "\n".join(parts)

    # -- ground truth ----------------------------------------------------

    def truth(self) -> dict:
        """Exact grading information for this position."""
        sol = self.solution
        cands = self.candidates()
        if not cands:
            return {}
        probs = {c: sol.prob[c] for c in cands}
        best = min(probs.values())
        return {
            # Defined over the candidates actually offered, not over the whole
            # board: "is one of the squares this agent may choose provably safe?"
            "deducible": best < 1e-9,
            "best_p": best,
            "worst_p": max(probs.values()),
            "n_safe": len(sol.safe),
            "n_candidates": len(cands),
            "probs": probs,
            "components": sol.components,
        }

    def step(self, cell) -> list[str]:
        sol = self.solution
        p = sol.prob.get(cell, 1.0)
        events = self.board.open_cell(*cell)
        if not self.board.lost:
            self.opened_safely += 1
        return events

    def score(self) -> dict:
        b = self.board
        return {
            "seed": self.seed,
            "result": self.result,
            "moves": b.moves,
            "cleared": b.won,
            "hidden_left": len(b.hidden_cells()),
            "safe_opens": self.opened_safely,
            "width": b.w,
            "height": b.h,
            "n_mines": b.n_mines,
        }
