"""Cooperation: evolutionary games between neighbours, in hyperedges, in tournaments, and between groups."""

from ._contributor import ContributorElement
from ._game_transitions import DonorElement, DonorView, GameTransitionsInteraction, GameTransitionsSystem
from ._hypergraph_games import (
    DilemmaElement,
    HyperedgeGoodsInteraction,
    HypergraphPublicGoodsSystem,
    SimplicialDilemmaInteraction,
    SimplicialSocialDilemmasSystem,
)
from ._multilevel_selection import (
    NonlinearPairwiseGroupConflictSystem,
    TrimorphicMultilevelSelectionSystem,
    TwoLevelReplicatorSystem,
)
from ._spatial_games import (
    IteratedDilemmaInteraction,
    IteratedLatticeDilemmaSystem,
    IteratedPrisonerElement,
    PrisonerElement,
    PrisonersDilemmaInteraction,
    PublicGoodsGameSystem,
    PublicGoodsInteraction,
    SpatialPrisonersDilemmaSystem,
)
from ._tournaments import (
    PrisonersDilemmaTournamentSystem,
    StrategistElement,
    StrategistView,
    TournamentInteraction,
    play,
)
from ._two_level_birth_death import (
    PunisherGroupElement,
    PunisherGroupView,
    TwoLevelBirthDeathInteraction,
    TwoLevelBirthDeathSystem,
)

__all__: list[str] = [
    'ContributorElement',
    'DilemmaElement',
    'DonorElement',
    'DonorView',
    'GameTransitionsInteraction',
    'GameTransitionsSystem',
    'HyperedgeGoodsInteraction',
    'HypergraphPublicGoodsSystem',
    'IteratedDilemmaInteraction',
    'IteratedLatticeDilemmaSystem',
    'IteratedPrisonerElement',
    'NonlinearPairwiseGroupConflictSystem',
    'PrisonerElement',
    'PrisonersDilemmaInteraction',
    'PrisonersDilemmaTournamentSystem',
    'PublicGoodsGameSystem',
    'PublicGoodsInteraction',
    'PunisherGroupElement',
    'PunisherGroupView',
    'SimplicialDilemmaInteraction',
    'SimplicialSocialDilemmasSystem',
    'SpatialPrisonersDilemmaSystem',
    'StrategistElement',
    'StrategistView',
    'TournamentInteraction',
    'TrimorphicMultilevelSelectionSystem',
    'TwoLevelBirthDeathInteraction',
    'TwoLevelBirthDeathSystem',
    'TwoLevelReplicatorSystem',
    'play',
]
