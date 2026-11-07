"""Cooperation: evolutionary games between neighbours, in hyperedges, in tournaments, and between groups."""

from ._contributor import ContributorElement
from ._game_transitions import DonorElement, DonorView, GameTransitionsInteraction, game_transitions
from ._hypergraph_games import (
    DilemmaElement,
    HyperedgeGoodsInteraction,
    SimplicialDilemmaInteraction,
    hypergraph_public_goods,
    simplicial_social_dilemmas,
)
from ._multilevel_selection import (
    nonlinear_pairwise_group_conflict,
    trimorphic_multilevel_selection,
    two_level_replicator,
)
from ._spatial_games import (
    IteratedDilemmaInteraction,
    IteratedPrisonerElement,
    PrisonerElement,
    PrisonersDilemmaInteraction,
    PublicGoodsInteraction,
    iterated_lattice_dilemma,
    public_goods_game,
    spatial_prisoners_dilemma,
)
from ._tournaments import StrategistElement, StrategistView, TournamentInteraction, play, prisoners_dilemma_tournament
from ._two_level_birth_death import (
    PunisherGroupElement,
    PunisherGroupView,
    TwoLevelBirthDeathInteraction,
    two_level_birth_death,
)

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
