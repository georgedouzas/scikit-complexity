"""Cooperation: evolutionary games between neighbours, in hyperedges, in tournaments, and between groups."""

from ._contributor import ContributorElement
from ._dilemma import DilemmaElement
from ._hyperedge_goods import HyperedgeGoodsInteraction
from ._hypergraph_games import hypergraph_public_goods, simplicial_social_dilemmas
from ._multilevel_selection import nonlinear_pairwise_group_conflict, trimorphic_multilevel_selection
from ._prisoner import PrisonerElement
from ._prisoners_dilemma import PrisonersDilemmaInteraction
from ._public_goods import PublicGoodsInteraction
from ._simplicial_dilemma import SimplicialDilemmaInteraction
from ._spatial_games import public_goods_game, spatial_prisoners_dilemma
from ._strategist import StrategistElement
from ._strategist_view import StrategistView
from ._tournament import TournamentInteraction, play
from ._tournaments import prisoners_dilemma_tournament

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
    'nonlinear_pairwise_group_conflict',
    'play',
    'prisoners_dilemma_tournament',
    'public_goods_game',
    'simplicial_social_dilemmas',
    'spatial_prisoners_dilemma',
    'trimorphic_multilevel_selection',
]
