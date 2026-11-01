"""Cooperation: evolutionary games between neighbours, in hyperedges, and in tournaments, with imitation."""

from ._hypergraph_games import (
    DilemmaElement,
    HyperedgeGoodsInteraction,
    SimplicialDilemmaInteraction,
    hypergraph_public_goods,
    simplicial_social_dilemmas,
)
from ._spatial_games import (
    ContributorElement,
    PrisonerElement,
    PrisonersDilemmaInteraction,
    PublicGoodsInteraction,
    public_goods_game,
    spatial_prisoners_dilemma,
)
from ._strategist_view import StrategistView
from ._tournaments import StrategistElement, TournamentInteraction, play, prisoners_dilemma_tournament

__all__: list[str] = [
    'ContributorElement',
    'DilemmaElement',
    'HyperedgeGoodsInteraction',
    'PrisonerElement',
    'PrisonersDilemmaInteraction',
    'PublicGoodsInteraction',
    'SimplicialDilemmaInteraction',
    'StrategistElement',
    'StrategistView',
    'TournamentInteraction',
    'hypergraph_public_goods',
    'play',
    'prisoners_dilemma_tournament',
    'public_goods_game',
    'simplicial_social_dilemmas',
    'spatial_prisoners_dilemma',
]
