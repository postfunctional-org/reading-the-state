"""Control agents for Coalmouth.

These are the yardsticks the Decision models get measured against:

* ``RandomAgent``  - the floor. Picks uniformly among legal actions.
* ``HeuristicAgent`` - a hand-written greedy roguelike player. Not optimal, but it
  understands the game's actual incentives (heal when hurt, eat when hungry, hit
  what is next to you, walk toward loot, then take the stairs). It doubles as the
  *reference policy* for regret: on any given turn we can ask what it would have
  done and how much worse the model's pick was.

Pure stdlib, same as the engine.
"""

from __future__ import annotations

import random
from collections import deque

from rogue import DIRS, MAP_H, MAP_W, Game


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, game: Game):
        return self.rng.choice(game.legal_actions()).key


class HeuristicAgent:
    """Greedy but sensible. Roughly how a human plays a roguelike on autopilot."""

    name = "heuristic"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    # -- pathfinding over the tiles the player has actually explored ------

    @staticmethod
    def _bfs_step(game: Game, goals: set[tuple[int, int]]) -> str | None:
        """First move on the shortest known path to any goal tile."""
        if not goals:
            return None
        lvl = game.level
        start = (game.player.x, game.player.y)
        if start in goals:
            return None
        seen = {start}
        # queue holds (pos, first_direction_name)
        q: deque[tuple[tuple[int, int], str]] = deque()
        for name, (dx, dy) in DIRS.items():
            nx, ny = start[0] + dx, start[1] + dy
            if not lvl.walkable(nx, ny):
                continue
            if lvl.monster_at(nx, ny) is not None:
                continue
            seen.add((nx, ny))
            q.append(((nx, ny), name))
        while q:
            (pos, first) = q.popleft()
            if pos in goals:
                return f"move_{first}"
            for name, (dx, dy) in DIRS.items():
                nx, ny = pos[0] + dx, pos[1] + dy
                if (nx, ny) in seen:
                    continue
                if not (0 <= nx < MAP_W and 0 <= ny < MAP_H):
                    continue
                if not lvl.explored[ny][nx] or not lvl.walkable(nx, ny):
                    continue
                if lvl.monster_at(nx, ny) is not None:
                    continue
                seen.add((nx, ny))
                q.append(((nx, ny), first))
        return None

    @staticmethod
    def _frontier(game: Game) -> set[tuple[int, int]]:
        """Explored floor tiles that touch something unexplored - i.e. where to go next."""
        lvl = game.level
        out = set()
        for y in range(MAP_H):
            for x in range(MAP_W):
                if not lvl.explored[y][x] or not lvl.walkable(x, y):
                    continue
                for dx, dy in DIRS.values():
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < MAP_W and 0 <= ny < MAP_H and not lvl.explored[ny][nx]:
                        out.add((x, y))
                        break
        return out

    # -- policy -----------------------------------------------------------

    def act(self, game: Game) -> str:
        p = game.player
        keys = {a.key: a for a in game.legal_actions()}
        adjacent = [a for a in keys.values() if a.kind == "attack"]

        # survival first
        if p.hp <= p.maxhp * 0.35 and "quaff" in keys:
            return "quaff"
        if p.food < 40 and "eat" in keys:
            return "eat"

        # fight what is already on top of you, weakest first
        if adjacent:
            adjacent.sort(key=lambda a: a.target.hp)
            # badly hurt with no potion: back away if there is anywhere to go
            if p.hp <= p.maxhp * 0.25 and "quaff" not in keys:
                retreat = [a for a in keys.values() if a.kind == "move"]
                if retreat:
                    return self.rng.choice(retreat).key
            return adjacent[0].key

        if "pickup" in keys:
            return "pickup"

        # top up out of combat while food allows
        if p.hp < p.maxhp * 0.6 and p.food > 120 and not game.visible_monsters():
            return "rest"

        lvl = game.level
        loot = {(it.x, it.y) for it in lvl.items if lvl.explored[it.y][it.x]}
        if loot:
            mv = self._bfs_step(game, loot)
            if mv and mv in keys:
                return mv

        if "descend" in keys:
            return "descend"
        if lvl.stairs is not None and lvl.explored[lvl.stairs[1]][lvl.stairs[0]]:
            mv = self._bfs_step(game, {lvl.stairs})
            if mv and mv in keys:
                return mv

        mv = self._bfs_step(game, self._frontier(game))
        if mv and mv in keys:
            return mv

        moves = [a for a in keys.values() if a.kind == "move"]
        if moves:
            return self.rng.choice(moves).key
        return "rest"


def play(agent, seed: int) -> dict:
    g = Game(seed)
    while not g.done:
        g.step(agent.act(g))
    return g.score()


if __name__ == "__main__":
    import statistics
    from collections import Counter

    for cls in (RandomAgent, HeuristicAgent):
        rows = [play(cls(seed=s), s) for s in range(200)]
        res = Counter(r["result"] for r in rows)
        print(
            f"{cls.name:10s} win {sum(r['won'] for r in rows)/len(rows):5.1%}  "
            f"mean depth {statistics.mean(r['depth'] for r in rows):4.2f}  "
            f"mean kills {statistics.mean(r['kills'] for r in rows):4.2f}  "
            f"mean turns {statistics.mean(r['turns'] for r in rows):5.1f}  {dict(res)}"
        )


class FrozenPriorAgent:
    """The control that decides whether any of this is real.

    Plays using a model's *marginal* preference over action keys, measured across a
    whole corpus of its own turns, with the current state thrown away. If a model
    cannot beat its own state-blind marginal, then its apparent competence was a
    fixed ranking over action types and nothing it read on the map mattered.

    `weights` maps action key -> mean probability the model gave that key when it
    was on offer. Among the legal actions it takes the highest-weighted one, which
    keeps it deterministic and gives the live model the friendliest possible
    comparison (no sampling noise to hide behind).
    """

    def __init__(self, weights: dict, seed: int = 0, name: str = "frozen-prior"):
        self.weights = weights
        self.name = name
        self.rng = random.Random(seed)

    def act(self, game) -> str:
        actions = game.legal_actions()
        best, best_w = None, -1.0
        for a in actions:
            w = self.weights.get(a.key, self.weights.get(a.kind, 0.0))
            if w > best_w:
                best, best_w = a.key, w
        return best or self.rng.choice(actions).key
