"""Cooperation: evolutionary games between neighbours and in tournaments."""

from ._spatial_games import ContributorComponent, PrisonerComponent, public_goods_game, spatial_prisoners_dilemma
from ._tournaments import StrategistComponent, TournamentInteraction, play, prisoners_dilemma_tournament

__all__: list[str] = [
    'ContributorComponent',
    'PrisonerComponent',
    'StrategistComponent',
    'TournamentInteraction',
    'play',
    'prisoners_dilemma_tournament',
    'public_goods_game',
    'spatial_prisoners_dilemma',
]
