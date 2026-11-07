"""Cooperation: evolutionary games between neighbours, in hyperedges, in tournaments, and between groups."""

from ._contributor import ContributorElement
from ._dilemma import DilemmaElement
from ._donor_view import DonorView
from ._game_transitions import DonorElement, GameTransitionsInteraction, game_transitions
from ._hyperedge_goods import HyperedgeGoodsInteraction
from ._hypergraph_games import hypergraph_public_goods, simplicial_social_dilemmas
from ._iterated_dilemma import IteratedDilemmaInteraction
from ._iterated_prisoner import IteratedPrisonerElement
from ._multilevel_selection import (
    nonlinear_pairwise_group_conflict,
    trimorphic_multilevel_selection,
    two_level_replicator,
)
from ._prisoner import PrisonerElement
from ._prisoners_dilemma import PrisonersDilemmaInteraction
from ._public_goods import PublicGoodsInteraction
from ._punisher_group_view import PunisherGroupView
from ._simplicial_dilemma import SimplicialDilemmaInteraction
from ._spatial_games import iterated_lattice_dilemma, public_goods_game, spatial_prisoners_dilemma
from ._strategist import StrategistElement
from ._strategist_view import StrategistView
from ._tournament import TournamentInteraction, play
from ._tournaments import prisoners_dilemma_tournament
from ._two_level_birth_death import PunisherGroupElement, TwoLevelBirthDeathInteraction, two_level_birth_death

__all__: list[str] = [
    'ContributorElement',
    'DilemmaElement',
    'DonorElement',
    'DonorView',
    'GameTransitionsInteraction',
    'HyperedgeGoodsInteraction',
    'IteratedDilemmaInteraction',
    'IteratedPrisonerElement',
    'PrisonerElement',
    'PrisonersDilemmaInteraction',
    'PublicGoodsInteraction',
    'PunisherGroupElement',
    'PunisherGroupView',
    'SimplicialDilemmaInteraction',
    'StrategistElement',
    'StrategistView',
    'TournamentInteraction',
    'TwoLevelBirthDeathInteraction',
    'game_transitions',
    'hypergraph_public_goods',
    'iterated_lattice_dilemma',
    'nonlinear_pairwise_group_conflict',
    'play',
    'prisoners_dilemma_tournament',
    'public_goods_game',
    'simplicial_social_dilemmas',
    'spatial_prisoners_dilemma',
    'trimorphic_multilevel_selection',
    'two_level_birth_death',
    'two_level_replicator',
]
