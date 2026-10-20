"""Evolution: evolutionary games, quasispecies, hypercycles, finite populations, landscapes, and coevolution."""

from ._coevolution import Species, bak_sneppen
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
from ._games import CyclicSpecies, Replication, Strategy, replicator_dynamics, rock_paper_scissors
from ._prebiotic import (
    Catalysis,
    GenotypeClass,
    MasterSequence,
    Molecule,
    Mutation,
    eigen_quasispecies,
    hypercycles,
)
from ._substitutions import Locus, Substitution, origin_fixation_walk

__all__: list[str] = [
    'AdaptiveClimbing',
    'Catalysis',
    'Climber',
    'CyclicSpecies',
    'GenotypeClass',
    'Individual',
    'Locus',
    'MasterSequence',
    'Molecule',
    'MoranStep',
    'Mutation',
    'Replication',
    'Species',
    'Strategy',
    'Substitution',
    'TangledNatureStep',
    'adaptive_climbing',
    'bak_sneppen',
    'eigen_quasispecies',
    'hypercycles',
    'moran_process',
    'nk_landscape',
    'origin_fixation_walk',
    'replicator_dynamics',
    'rock_paper_scissors',
    'tangled_nature',
]
