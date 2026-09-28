"""Coalmouth - a compact, seeded roguelike built as a decision-model benchmark.

Design constraints that shaped this file:

* The entire game state must render into roughly 1024 tokens, because
  Decision-1.0-Kai-0.6B rejects anything larger. That caps the map at 28x14 and
  forces a terse status line.
* Every turn must expose a small, explicitly enumerated set of legal actions with
  short natural-language descriptions, because that is exactly the shape of a
  Decision "Choice" question: criteria are supplied at runtime.
* Everything is deterministic given a seed, so two different models can be run
  through byte-identical situations and compared as a paired sample.

No third-party imports: the engine has to run whatever state the venv is in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

MAP_W, MAP_H = 28, 14
SIGHT = 7
MAX_TURNS = 250
WIN_DEPTH = 5

WALL, FLOOR, STAIRS = "#", ".", ">"

# ---------------------------------------------------------------------------
# content tables
# ---------------------------------------------------------------------------

BESTIARY = {
    #  ch   hp  atk  def  min_depth  weight  sight
    "rat": dict(ch="r", hp=5, atk=1, dfn=0, min_depth=1, weight=10, sight=4),
    "kobold": dict(ch="k", hp=7, atk=2, dfn=0, min_depth=1, weight=8, sight=5),
    "goblin": dict(ch="g", hp=10, atk=3, dfn=1, min_depth=2, weight=8, sight=6),
    "snake": dict(ch="s", hp=8, atk=4, dfn=1, min_depth=2, weight=6, sight=4),
    "orc": dict(ch="o", hp=14, atk=5, dfn=2, min_depth=3, weight=7, sight=6),
    "wraith": dict(ch="w", hp=13, atk=6, dfn=2, min_depth=4, weight=5, sight=7),
    "ogre": dict(ch="O", hp=20, atk=7, dfn=3, min_depth=4, weight=4, sight=5),
}

ITEM_CHARS = {
    "potion": "!",
    "ration": "%",
    "weapon": ")",
    "armor": "[",
    "gold": "$",
    "amulet": "*",
}

DIRS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


@dataclass
class Monster:
    kind: str
    x: int
    y: int
    hp: int
    maxhp: int
    atk: int
    dfn: int
    ch: str
    sight: int
    awake: bool = False

    @property
    def alive(self) -> bool:
        return self.hp > 0


@dataclass
class Item:
    kind: str
    x: int
    y: int
    power: int = 0  # weapon/armor bonus, gold amount


@dataclass
class Action:
    """One legal move. `key` is the Choice candidate id, `desc` its description."""

    key: str
    desc: str
    kind: str          # move | attack | pickup | quaff | eat | descend | rest
    target: object = None


@dataclass
class Player:
    x: int = 0
    y: int = 0
    hp: int = 24
    maxhp: int = 24
    atk: int = 4
    dfn: int = 1
    food: int = 260
    gold: int = 0
    potions: int = 1
    rations: int = 1
    depth: int = 1
    kills: int = 0
    level: int = 1
    has_amulet: bool = False

    @property
    def alive(self) -> bool:
        return self.hp > 0


# ---------------------------------------------------------------------------
# level generation
# ---------------------------------------------------------------------------


class Level:
    """Rooms joined by L-shaped corridors. Classic, legible, and always connected."""

    def __init__(self, rng: random.Random, depth: int):
        self.rng = rng
        self.depth = depth
        self.grid = [[WALL] * MAP_W for _ in range(MAP_H)]
        self.rooms: list[tuple[int, int, int, int]] = []
        self.monsters: list[Monster] = []
        self.items: list[Item] = []
        self.explored = [[False] * MAP_W for _ in range(MAP_H)]
        self.stairs: tuple[int, int] | None = None
        self._carve()

    # -- carving ---------------------------------------------------------

    def _carve(self) -> None:
        attempts = 60
        want = self.rng.randint(5, 7)
        for _ in range(attempts):
            if len(self.rooms) >= want:
                break
            w = self.rng.randint(4, 7)
            h = self.rng.randint(3, 5)
            x = self.rng.randint(1, MAP_W - w - 2)
            y = self.rng.randint(1, MAP_H - h - 2)
            if any(self._overlaps((x, y, w, h), r) for r in self.rooms):
                continue
            self.rooms.append((x, y, w, h))
            for yy in range(y, y + h):
                for xx in range(x, x + w):
                    self.grid[yy][xx] = FLOOR

        for a, b in zip(self.rooms, self.rooms[1:]):
            self._corridor(self._centre(a), self._centre(b))

    @staticmethod
    def _overlaps(a, b) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax + aw + 1 <= bx or bx + bw + 1 <= ax or ay + ah + 1 <= by or by + bh + 1 <= ay)

    @staticmethod
    def _centre(room) -> tuple[int, int]:
        x, y, w, h = room
        return x + w // 2, y + h // 2

    def _corridor(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        (x1, y1), (x2, y2) = a, b
        if self.rng.random() < 0.5:
            for x in range(min(x1, x2), max(x1, x2) + 1):
                self.grid[y1][x] = FLOOR
            for y in range(min(y1, y2), max(y1, y2) + 1):
                self.grid[y][x2] = FLOOR
        else:
            for y in range(min(y1, y2), max(y1, y2) + 1):
                self.grid[y][x1] = FLOOR
            for x in range(min(x1, x2), max(x1, x2) + 1):
                self.grid[y2][x] = FLOOR

    # -- queries ---------------------------------------------------------

    def walkable(self, x: int, y: int) -> bool:
        return 0 <= x < MAP_W and 0 <= y < MAP_H and self.grid[y][x] != WALL

    def free_floor(self, taken: Iterable[tuple[int, int]]) -> tuple[int, int]:
        taken = set(taken)
        for _ in range(500):
            x = self.rng.randint(1, MAP_W - 2)
            y = self.rng.randint(1, MAP_H - 2)
            if self.grid[y][x] == FLOOR and (x, y) not in taken:
                return x, y
        for y in range(MAP_H):
            for x in range(MAP_W):
                if self.grid[y][x] == FLOOR and (x, y) not in taken:
                    return x, y
        raise RuntimeError("no free floor")

    def monster_at(self, x: int, y: int) -> Monster | None:
        for m in self.monsters:
            if m.alive and m.x == x and m.y == y:
                return m
        return None

    def item_at(self, x: int, y: int) -> Item | None:
        for it in self.items:
            if it.x == x and it.y == y:
                return it
        return None


# ---------------------------------------------------------------------------
# the game
# ---------------------------------------------------------------------------


class Game:
    """One episode of Coalmouth. Seeded, deterministic, fully observable to the harness."""

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.player = Player()
        self.turn = 0
        self.done = False
        self.result: str | None = None
        self.log: list[str] = []
        self.kills = 0
        self.damage_taken = 0
        self.damage_dealt = 0
        # A decision model is stateless: it sees one position and answers. Without a
        # trail it cannot tell "north" from "north again, having just come from there",
        # and two-tile oscillation is the inevitable result. Any human player has this
        # information for free, so withholding it would make the task unfairly
        # memoryless rather than merely hard.
        self.recent: list[str] = []
        self.visits: dict[tuple[int, int, int], int] = {}
        self.level: Level = None  # type: ignore[assignment]
        self._new_level(1)

    # -- level plumbing --------------------------------------------------

    def _new_level(self, depth: int) -> None:
        self.player.depth = depth
        self.level = Level(self.rng, depth)
        lvl = self.level
        px, py = lvl._centre(lvl.rooms[0])
        self.player.x, self.player.y = px, py
        taken = {(px, py)}

        if depth < WIN_DEPTH:
            sx, sy = lvl.free_floor(taken)
            lvl.stairs = (sx, sy)
            lvl.grid[sy][sx] = STAIRS
            taken.add((sx, sy))
        else:
            ax, ay = lvl.free_floor(taken)
            lvl.items.append(Item("amulet", ax, ay))
            taken.add((ax, ay))

        pool = [(k, v) for k, v in BESTIARY.items() if v["min_depth"] <= depth]
        weights = [v["weight"] for _, v in pool]
        n_mon = 2 + depth
        for _ in range(n_mon):
            kind, spec = self.rng.choices(pool, weights=weights, k=1)[0]
            mx, my = lvl.free_floor(taken)
            taken.add((mx, my))
            hp = spec["hp"] + self.rng.randint(0, depth)
            lvl.monsters.append(
                Monster(kind, mx, my, hp, hp, spec["atk"], spec["dfn"], spec["ch"], spec["sight"])
            )

        for _ in range(self.rng.randint(2, 4)):
            roll = self.rng.random()
            ix, iy = lvl.free_floor(taken)
            taken.add((ix, iy))
            if roll < 0.34:
                lvl.items.append(Item("potion", ix, iy))
            elif roll < 0.58:
                lvl.items.append(Item("ration", ix, iy))
            elif roll < 0.74:
                lvl.items.append(Item("weapon", ix, iy, power=self.rng.randint(1, 2)))
            elif roll < 0.88:
                lvl.items.append(Item("armor", ix, iy, power=1))
            else:
                lvl.items.append(Item("gold", ix, iy, power=self.rng.randint(5, 25)))

        self._update_fov()

    # -- field of view ---------------------------------------------------

    def _update_fov(self) -> None:
        px, py = self.player.x, self.player.y
        lvl = self.level
        self.visible = set()
        for dy in range(-SIGHT, SIGHT + 1):
            for dx in range(-SIGHT, SIGHT + 1):
                if dx * dx + dy * dy > SIGHT * SIGHT:
                    continue
                tx, ty = px + dx, py + dy
                if not (0 <= tx < MAP_W and 0 <= ty < MAP_H):
                    continue
                if self._ray_clear(px, py, tx, ty):
                    self.visible.add((tx, ty))
                    lvl.explored[ty][tx] = True

    def _ray_clear(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        """Bresenham; walls block, but the wall tile itself is seen."""
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        cx, cy = x0, y0
        while (cx, cy) != (x1, y1):
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy
            if (cx, cy) == (x1, y1):
                return True
            if self.level.grid[cy][cx] == WALL:
                return False
        return True

    # -- rendering -------------------------------------------------------

    def render_map(self) -> str:
        lvl = self.level
        rows = []
        for y in range(MAP_H):
            row = []
            for x in range(MAP_W):
                if not lvl.explored[y][x]:
                    row.append(" ")
                    continue
                if (x, y) == (self.player.x, self.player.y):
                    row.append("@")
                    continue
                vis = (x, y) in self.visible
                m = lvl.monster_at(x, y) if vis else None
                if m is not None:
                    row.append(m.ch)
                    continue
                it = lvl.item_at(x, y)
                if it is not None:
                    row.append(ITEM_CHARS[it.kind])
                    continue
                row.append(lvl.grid[y][x])
            rows.append("".join(row).rstrip())
        return "\n".join(rows)

    def visible_monsters(self) -> list[Monster]:
        return [m for m in self.level.monsters if m.alive and (m.x, m.y) in self.visible]

    def state_text(self) -> str:
        p = self.player
        seen = self.visible_monsters()
        parts = [
            f"DUNGEON OF COALMOUTH - depth {p.depth} of {WIN_DEPTH}, turn {self.turn}",
            f"HP {p.hp}/{p.maxhp}  attack {p.atk}  defence {p.dfn}  experience level {p.level}  food {p.food}  gold {p.gold}",
            f"Carrying: {p.potions} healing potion(s), {p.rations} ration(s)"
            + ("  YOU HOLD THE AMULET" if p.has_amulet else ""),
            "",
            self.render_map(),
            "",
            "Map key: @ you   > stairs down   ! potion   % ration   ) weapon   [ armor   $ gold   * amulet   # wall   . floor   (blank = unexplored)",
        ]
        if seen:
            desc = []
            for m in seen:
                d = self._dist(m.x, m.y)
                bearing = self._bearing(m.x, m.y)
                desc.append(f"{m.ch}={m.kind} {m.hp}/{m.maxhp}hp atk{m.atk} {d} step(s) {bearing}")
            parts.append("Monsters in sight: " + "; ".join(desc))
        else:
            parts.append("Monsters in sight: none")
        parts.append(self.trail_line())
        if self.log:
            parts.append("Just happened: " + " ".join(self.log[-3:]))
        return "\n".join(parts)

    def state_prose(self) -> str:
        """The same turn, described instead of drawn.

        Every fact the ASCII view carries is here except exact grid topology:
        status, what is in each of the four adjacent squares, what is in sight and
        how far, whether the stairs are known, and where unexplored ground lies.
        The point of having both is to separate "cannot decide" from "cannot read
        a map" - these models were trained on text decisions, not cartography.
        """
        p = self.player
        lvl = self.level
        parts = [
            f"DUNGEON OF COALMOUTH - depth {p.depth} of {WIN_DEPTH}, turn {self.turn}.",
            f"HP {p.hp} of {p.maxhp}. Attack {p.atk}. Defence {p.dfn}. Experience level {p.level}. "
            f"Food {p.food}. Gold {p.gold}.",
            f"Carrying {p.potions} healing potion(s) and {p.rations} ration(s)."
            + (" YOU HOLD THE AMULET." if p.has_amulet else ""),
        ]

        around = []
        for name, (dx, dy) in DIRS.items():
            nx, ny = p.x + dx, p.y + dy
            if not lvl.walkable(nx, ny):
                what = "solid rock"
            else:
                m = lvl.monster_at(nx, ny)
                it = lvl.item_at(nx, ny)
                if m is not None:
                    what = f"a {m.kind} ({m.hp} of {m.maxhp} hp)"
                elif it is not None:
                    what = f"a {it.kind} lying on the floor"
                elif (nx, ny) == lvl.stairs:
                    what = "the staircase down"
                else:
                    what = "open floor"
            around.append(f"to the {name} is {what}")
        parts.append("Immediately around you: " + ", ".join(around) + ".")

        here = lvl.item_at(p.x, p.y)
        if here is not None:
            parts.append(f"You are standing on a {here.kind}.")
        if lvl.stairs is not None and (p.x, p.y) == lvl.stairs:
            parts.append("You are standing on the staircase down.")

        seen = self.visible_monsters()
        if seen:
            bits = [
                f"a {m.kind} ({m.hp} of {m.maxhp} hp, attack {m.atk}) "
                f"{self._dist(m.x, m.y)} step(s) to the {self._bearing(m.x, m.y)}"
                for m in seen
            ]
            parts.append("You can see " + "; ".join(bits) + ".")
        else:
            parts.append("No monster is in sight.")

        loot = [it for it in lvl.items if (it.x, it.y) in self.visible and (it.x, it.y) != (p.x, p.y)]
        if loot:
            bits = [
                f"a {it.kind} {self._dist(it.x, it.y)} step(s) to the {self._bearing(it.x, it.y)}"
                for it in loot
            ]
            parts.append("You can also see " + "; ".join(bits) + ".")

        if lvl.stairs is None:
            parts.append("The Amulet is somewhere on this level; you have not found it yet.")
        elif lvl.explored[lvl.stairs[1]][lvl.stairs[0]]:
            sx, sy = lvl.stairs
            parts.append(
                f"You know where the staircase down is: {self._dist(sx, sy)} step(s) to the {self._bearing(sx, sy)}."
            )
        else:
            parts.append("You have not found the staircase down on this level yet.")

        unexplored = []
        for name, (dx, dy) in DIRS.items():
            for step in range(1, SIGHT + 2):
                tx, ty = p.x + dx * step, p.y + dy * step
                if not (0 <= tx < MAP_W and 0 <= ty < MAP_H):
                    break
                if not lvl.explored[ty][tx]:
                    unexplored.append(name)
                    break
        parts.append(
            ("Unexplored ground lies to the " + ", ".join(unexplored) + ".")
            if unexplored
            else "Everything within reach has been explored."
        )

        parts.append(self.trail_line())
        if self.log:
            parts.append("Just happened: " + " ".join(self.log[-3:]))
        return "\n".join(parts)

    def _dist(self, x: int, y: int) -> int:
        return max(abs(x - self.player.x), abs(y - self.player.y))

    def _bearing(self, x: int, y: int) -> str:
        dx, dy = x - self.player.x, y - self.player.y
        v = "north" if dy < 0 else "south" if dy > 0 else ""
        h = "west" if dx < 0 else "east" if dx > 0 else ""
        return (v + h) or "here"

    # -- actions ---------------------------------------------------------

    def legal_actions(self) -> list[Action]:
        p = self.player
        lvl = self.level
        acts: list[Action] = []

        for name, (dx, dy) in DIRS.items():
            nx, ny = p.x + dx, p.y + dy
            if not lvl.walkable(nx, ny):
                continue
            m = lvl.monster_at(nx, ny)
            if m is not None:
                acts.append(
                    Action(
                        f"attack_{name}",
                        f"Attack the {m.kind} standing {name} of you (it has {m.hp} of {m.maxhp} hit points left).",
                        "attack",
                        m,
                    )
                )
            else:
                it = lvl.item_at(nx, ny)
                if it is not None:
                    what = f"onto the {it.kind}"
                elif (nx, ny) == lvl.stairs:
                    what = "onto the staircase down"
                else:
                    what = "onto open floor"
                acts.append(Action(f"move_{name}", f"Walk one step {name}, {what}.", "move", (dx, dy)))

        here = lvl.item_at(p.x, p.y)
        if here is not None:
            acts.append(
                Action("pickup", f"Pick up the {here.kind} you are standing on.", "pickup", here)
            )
        if p.potions > 0:
            acts.append(
                Action(
                    "quaff",
                    f"Drink a healing potion to restore hit points (you are at {p.hp} of {p.maxhp}).",
                    "quaff",
                )
            )
        if p.rations > 0 and p.food < 200:
            acts.append(
                Action("eat", f"Eat a ration to refill your food (food is {p.food}).", "eat")
            )
        if lvl.stairs is not None and (p.x, p.y) == lvl.stairs:
            acts.append(
                Action(
                    "descend",
                    f"Take the staircase down to depth {p.depth + 1}, leaving this level behind.",
                    "descend",
                )
            )
        # Resting only does anything below full health - the engine's regen check is
        # `hp < maxhp`. Offering it at full HP offers a strictly dominated no-op that
        # merely burns food, which is a flaw in the option set rather than a real
        # choice, so it is withheld. A fallback keeps the action list non-empty in the
        # (unreachable in practice) case where nothing else is legal.
        if p.hp < p.maxhp:
            acts.append(
                Action("rest", "Stand still and rest for one turn, recovering a little health.", "rest")
            )
        if not acts:
            acts.append(Action("rest", "Stand still for one turn; there is nothing else to do.", "rest"))
        return acts

    # -- the turn --------------------------------------------------------

    def step(self, key: str) -> list[str]:
        if self.done:
            return []
        acts = {a.key: a for a in self.legal_actions()}
        if key not in acts:
            key = "rest"
        act = acts[key]
        self.log = []
        p = self.player

        if act.kind == "move":
            dx, dy = act.target
            p.x, p.y = p.x + dx, p.y + dy
        elif act.kind == "attack":
            self._attack(act.target)
        elif act.kind == "pickup":
            self._pickup(act.target)
        elif act.kind == "quaff":
            p.potions -= 1
            heal = self.rng.randint(10, 16)
            p.hp = min(p.maxhp, p.hp + heal)
            self.log.append(f"You drink a potion and recover {heal} hit points.")
        elif act.kind == "eat":
            p.rations -= 1
            p.food = min(300, p.food + 180)
            self.log.append("You eat a ration.")
        elif act.kind == "descend":
            self._new_level(p.depth + 1)
            self.log.append(f"You descend to depth {p.depth}.")
            self.turn += 1
            self._upkeep()
            self.recent.clear()
            self._mark_trail(key)
            return self.log
        elif act.kind == "rest":
            if p.hp < p.maxhp and self.rng.random() < 0.45:
                p.hp += 1

        self._monsters_act()
        self.turn += 1
        self._upkeep()
        self._update_fov()
        self._mark_trail(key)
        return self.log

    def _mark_trail(self, key: str) -> None:
        self.recent.append(key)
        del self.recent[:-6]
        p = self.player
        spot = (p.depth, p.x, p.y)
        self.visits[spot] = self.visits.get(spot, 0) + 1

    def trail_line(self) -> str:
        p = self.player
        here = self.visits.get((p.depth, p.x, p.y), 0)
        moves = ", ".join(self.recent) if self.recent else "nothing yet"
        line = f"Your last moves, oldest first: {moves}."
        if here >= 3:
            line += f" You have already stood on this exact tile {here} times on this level."
        return line

    def _attack(self, m: Monster) -> None:
        roll = self.rng.randint(1, 6) + self.player.atk - m.dfn
        dmg = max(0, roll)
        m.hp -= dmg
        self.damage_dealt += dmg
        m.awake = True
        if m.hp <= 0:
            self.kills += 1
            p = self.player
            p.kills += 1
            self.log.append(f"You kill the {m.kind}.")
            if p.kills % 3 == 0:  # a kill every three earns a level: +hp, +attack
                p.level += 1
                p.maxhp += 5
                p.hp = min(p.maxhp, p.hp + 5)
                p.atk += 1
                self.log.append(f"You reach experience level {p.level}.")
        else:
            self.log.append(f"You hit the {m.kind} for {dmg}.")

    def _pickup(self, it: Item) -> None:
        p = self.player
        self.level.items.remove(it)
        if it.kind == "potion":
            p.potions += 1
            self.log.append("You pick up a healing potion.")
        elif it.kind == "ration":
            p.rations += 1
            self.log.append("You pick up a ration.")
        elif it.kind == "weapon":
            p.atk += it.power
            self.log.append(f"You wield a better weapon (+{it.power} attack).")
        elif it.kind == "armor":
            p.dfn += it.power
            self.log.append(f"You put on better armour (+{it.power} defence).")
        elif it.kind == "gold":
            p.gold += it.power
            self.log.append(f"You pocket {it.power} gold.")
        elif it.kind == "amulet":
            p.has_amulet = True
            self.done = True
            self.result = "win"
            self.log.append("You lift the Amulet of Coalmouth. You win.")

    def _monsters_act(self) -> None:
        p = self.player
        for m in self.level.monsters:
            if not m.alive:
                continue
            d = max(abs(m.x - p.x), abs(m.y - p.y))
            if not m.awake:
                if d <= m.sight and (m.x, m.y) in self.visible:
                    m.awake = True
                else:
                    continue
            if d == 1:
                roll = self.rng.randint(1, 4) + m.atk - p.dfn
                dmg = max(0, roll)
                p.hp -= dmg
                self.damage_taken += dmg
                self.log.append(f"The {m.kind} hits you for {dmg}.")
                if p.hp <= 0:
                    self.done = True
                    self.result = "dead"
                    self.log.append(f"The {m.kind} kills you.")
                    return
            elif d <= m.sight:
                sx = (p.x > m.x) - (p.x < m.x)
                sy = (p.y > m.y) - (p.y < m.y)
                for nx, ny in ((m.x + sx, m.y + sy), (m.x + sx, m.y), (m.x, m.y + sy)):
                    if self.level.walkable(nx, ny) and self.level.monster_at(nx, ny) is None and (nx, ny) != (p.x, p.y):
                        m.x, m.y = nx, ny
                        break

    def _upkeep(self) -> None:
        p = self.player
        if self.done:
            return
        p.food -= 1
        if p.food <= 0:
            p.hp -= 1
            if p.hp <= 0:
                self.done = True
                self.result = "starved"
                self.log.append("You starve to death.")
                return
        if self.turn >= MAX_TURNS:
            self.done = True
            self.result = "timeout"
            self.log.append("The dungeon outlasts you.")

    # -- scoring ---------------------------------------------------------

    def score(self) -> dict:
        p = self.player
        return dict(
            seed=self.seed,
            result=self.result,
            depth=p.depth,
            turns=self.turn,
            hp=max(0, p.hp),
            gold=p.gold,
            kills=self.kills,
            damage_dealt=self.damage_dealt,
            damage_taken=self.damage_taken,
            won=self.result == "win",
        )


if __name__ == "__main__":  # tiny smoke test: a random walker
    import statistics

    results, depths, turns = [], [], []
    for s in range(200):
        g = Game(s)
        r = random.Random(s * 7919 + 1)
        while not g.done:
            acts = g.legal_actions()
            g.step(r.choice(acts).key)
        sc = g.score()
        results.append(sc["result"])
        depths.append(sc["depth"])
        turns.append(sc["turns"])
    from collections import Counter

    print("random agent over 200 seeds:", Counter(results))
    print("mean depth", round(statistics.mean(depths), 2), " mean turns", round(statistics.mean(turns), 1))
