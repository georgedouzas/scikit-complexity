"""Evolution: evolutionary games, quasispecies, hypercycles, finite populations, and fitness landscapes."""

from ._finite_populations import (
    AdaptiveClimbing,
    Climber,
    Individual,
    MoranStep,
    TangledNatureStep,
    adaptive_climbing,
    moran_process,
    nk_landscape,
    tangled_nature,
)
from ._games import CyclicSpecies, HawksAndDoves, Replicators, replicator_dynamics, rock_paper_scissors
from ._prebiotic import (
    CatalyticNetwork,
    GenotypeClass,
    MasterSequence,
    Mutation,
    eigen_quasispecies,
    hypercycles,
)

__all__: list[str] = [
    'AdaptiveClimbing',
    'CatalyticNetwork',
    'Climber',
    'CyclicSpecies',
    'GenotypeClass',
    'HawksAndDoves',
    'Individual',
    'MasterSequence',
    'MoranStep',
    'Mutation',
    'Replicators',
    'TangledNatureStep',
    'adaptive_climbing',
    'eigen_quasispecies',
    'hypercycles',
    'moran_process',
    'nk_landscape',
    'replicator_dynamics',
    'rock_paper_scissors',
    'tangled_nature',
]
