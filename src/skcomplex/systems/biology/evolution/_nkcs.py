"""Evolution systems: species that coevolve on coupled NK fitness landscapes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import DiscreteTime
from ....core.components.system import System
from ..._catalog import register_system
from ._coevolutionary_climbing import CoevolutionaryClimbingInteraction
from ._coevolving_species import CoevolvingSpeciesElement


@register_system(
    field='biology/evolution',
    tags=('coevolution', 'fitness landscape', 'avalanches', 'criticality', 'agents'),
    references=('Kauffman and Johnsen 1991', 'Thurner et al. 2018'),
    defaults={'until': 200},
)
def nkcs(
    n_species: int = 10,
    *,
    n: int = 12,
    k: int = 2,
    c: int = 1,
    s: int = 2,
    seed: int = 0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the NKCS model of Kauffman and Johnsen, species climbing NK landscapes coupled to each other's genes.

    Every gene of a species depends on `k` genes of its own species and on `c` genes in each of `s` partner species.
    A species keeps a point mutation when it raises its fitness, which changes the landscapes of the species that
    depend on it, so adaptations trigger coevolutionary avalanches. Rugged landscapes, `k` above about `c s`, freeze
    the ecosystem once every species sits on a peak, and smoother ones keep it in a chaotic regime of endless
    adaptation.

    Args:
        n_species:
            The number of species.

        n:
            The number of genes of every species.

        k:
            The number of genes of its own species each gene depends on.

        c:
            The number of genes of every partner species each gene depends on.

        s:
            The number of partner species of every species.

        seed:
            The seed of the landscapes and the couplings.

        random_state:
            The seed or generator of the initial genomes.

    Returns:
        system:
            The system, with the genomes `species.genome`, fitnesses `species.fitness`, and accepted mutations
            `species.moves` of the group `species`.

    Examples:
        >>> from skcomplex.systems.biology.evolution import nkcs
        >>> nkcs(5).elements[0].prepare().size_
        5
    """
    genomes = np.random.default_rng(random_state).integers(0, 2**n, size=n_species).astype(float)
    species = GroupElement('species', element=CoevolvingSpeciesElement, size=n_species, values={'genome': genomes})
    climbing = CoevolutionaryClimbingInteraction('climbing', species='species', n=n, k=k, c=c, s=s, seed=seed)
    return System(elements=[species], interactions=[climbing], time=DiscreteTime())
