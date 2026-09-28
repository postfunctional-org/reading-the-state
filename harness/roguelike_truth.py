"""Roguelike ground truth: which positions have exactly one non-blunder move.

bench/export_roguelike.py plays seeded games and keeps only the turns this labels,
45 of each rule, so every roguelike position has one right answer.
"""

from __future__ import annotations

from rogue import Game

INSTRUCTIONS = (
    "You are the adventurer exploring the dungeon described in the state. "
    "Study the map, your hit points, what is next to you and what you are carrying, "
    "then choose the single best action to take on this turn. "
    "You want to survive, get stronger, and reach the Amulet on depth 5. "
    "Making progress matters: explore ground you have not seen yet, and take the "
    "staircase down when you find it."
)


def tactical_ground_truth(game: Game) -> tuple[str | None, str | None]:
    """Label the turns where a correct answer is not a matter of taste.

    Returns (required_action_key, rule_name) or (None, None) when the position is
    genuinely a judgement call. Deliberately conservative: it only fires when a
    competent human player would call any other move a blunder.
    """
    p = game.player
    keys = {a.key: a for a in game.legal_actions()}
    adjacent = [a for a in keys.values() if a.kind == "attack"]

    # About to die, holding the cure.
    worst = max((a.target.atk for a in adjacent), default=0)
    if adjacent and "quaff" in keys and p.hp <= worst + 4:
        return "quaff", "heal_or_die"

    # Starving with food in the pack.
    if p.food <= 10 and "eat" in keys:
        return "eat", "eat_or_starve"

    # Standing on the amulet: the game ends the moment you pick it up.
    here = game.level.item_at(p.x, p.y)
    if here is not None and here.kind == "amulet" and "pickup" in keys:
        return "pickup", "take_the_win"

    # One hit finishes it, and it is already hitting you.
    if adjacent:
        killable = [a for a in adjacent if a.target.hp <= p.atk - a.target.dfn + 1]
        if killable and p.hp > worst * 2:
            return killable[0].key, "finish_the_kill"

    return None, None
