"""Cooperation: evolutionary games between neighbours and in tournaments."""

from ._spatial_games import ContributorElement, PrisonerElement, public_goods_game, spatial_prisoners_dilemma
from ._tournaments import StrategistElement, TournamentInteraction, play, prisoners_dilemma_tournament

__all__: list[str] = [
    'ContributorElement',
    'PrisonerElement',
    'StrategistElement',
    'TournamentInteraction',
    'play',
    'prisoners_dilemma_tournament',
    'public_goods_game',
    'spatial_prisoners_dilemma',
]
