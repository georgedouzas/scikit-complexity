"""Cooperation: evolutionary games between neighbours and in tournaments."""

from ._spatial_games import Contributor, Prisoner, public_goods_game, spatial_prisoners_dilemma
from ._tournaments import Strategist, Tournament, play, prisoners_dilemma_tournament

__all__: list[str] = [
    'Contributor',
    'Prisoner',
    'Strategist',
    'Tournament',
    'play',
    'prisoners_dilemma_tournament',
    'public_goods_game',
    'spatial_prisoners_dilemma',
]
