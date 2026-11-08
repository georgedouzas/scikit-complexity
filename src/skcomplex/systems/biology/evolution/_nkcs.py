"""Evolution systems: species that coevolve on coupled NK fitness landscapes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import lru_cache
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import DiscreteTime
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import RoleSpec, StateSpec, VisualSpec
from ....core.types import Array


class CoevolvingSpeciesView(AgentView):
    """View of a coevolving species, with its states as attributes.

    `CoevolvingSpeciesElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    genome: float
    fitness: float
    moves: float


class CoevolvingSpeciesElement(Element):
    """Species of an NKCS ecosystem, a genome whose fitness depends on its own genes and on genes of other species.

    A `CoevolutionaryClimbingInteraction` mutates the genome and sets the fitness and the number of accepted moves of
    the last step.

    Args:
        label:
            The label of the species or of the group of species.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        genome:
            The bits of the genome, as an integer.

        fitness:
            The fitness of the genome against the current genomes of the other species.

        moves:
            Whether the species accepted a mutation in the last step.

    Examples:
        >>> from skcomplex.systems.biology.evolution import CoevolvingSpeciesElement
        >>> CoevolvingSpeciesElement('species', genome=5.0).genome
        5.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        genome: Annotated[float, StateSpec(domain=NN)] = 0.0,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        moves: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.genome = genome
        self.fitness = fitness
        self.moves = moves

    @classmethod
    def _get_view(cls) -> type[CoevolvingSpeciesView]:
        """Return the view of a species, `CoevolvingSpeciesView`."""
        return CoevolvingSpeciesView


class CoevolutionaryClimbingInteraction(Interaction):
    """Hill climbing of species on coupled NK landscapes, one point mutation per species and step, the NKCS model.

    Every gene of a species contributes to its fitness through a random table of its own state, the states of the next
    `k` genes of the species, and the states of `c` random genes in each of `s` other species. The fitness of a species
    is the mean contribution of its `n` genes. At every step the species are picked in turn, and each flips one random
    bit and keeps the flip when it raises its fitness against the current genomes of the others. The tables, the
    partner species, and the coupled genes are fixed by `seed`.

    Args:
        label:
            The label of the interaction.

        species:
            The path of the group of species.

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

    Examples:
        >>> from skcomplex.systems.biology.evolution import CoevolutionaryClimbingInteraction
        >>> CoevolutionaryClimbingInteraction('climbing', species='species', k=4).k
        4
    """

    draws = True

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        species: Annotated[str, RoleSpec()],
        n: int = 12,
        k: int = 2,
        c: int = 1,
        s: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__(label)
        self.species = species
        self.n = n
        self.k = k
        self.c = c
        self.s = s
        self.seed = seed

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the climbing rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.climb)]

    def climb(self, system: SystemView) -> None:
        """Let every species in turn try one point mutation, and update the fitness of all.

        Args:
            system:
                The system.
        """
        species = system.agents(self.species, CoevolvingSpeciesView)
        genomes = [int(item.genome) for item in species]
        for index, item in enumerate(species):
            mutant = genomes[index] ^ (1 << int(system.rng.integers(self.n)))
            trial = [*genomes[:index], mutant, *genomes[index + 1 :]]
            accepted = self._fitness(index, trial) > self._fitness(index, genomes)
            if accepted:
                genomes[index] = mutant
            item.moves = float(accepted)
        for index, item in enumerate(species):
            item.genome = float(genomes[index])
            item.fitness = self._fitness(index, genomes)

    def _fitness(self, index: int, genomes: list[int]) -> float:
        """Return the fitness of a species, the mean contribution of its genes at the current genomes."""
        partners, genes, tables = self._landscape(len(genomes), (self.n, self.k, self.c, self.s, self.seed))
        own = genomes[index]
        total = 0.0
        for locus in range(self.n):
            bits = [(own >> ((locus + offset) % self.n)) & 1 for offset in range(self.k + 1)]
            for slot, partner in enumerate(partners[index]):
                bits.extend((genomes[int(partner)] >> int(gene)) & 1 for gene in genes[index, locus, slot])
            key = sum(bit << position for position, bit in enumerate(bits))
            total += tables[index, locus, key]
        return total / self.n

    @staticmethod
    @lru_cache(maxsize=8)
    def _landscape(size: int, settings: tuple[int, int, int, int, int]) -> tuple[Array, Array, Array]:
        """Return the partner species, the coupled genes of every gene, and the tables, from `(n, k, c, s, seed)`."""
        n, k, c, s, seed = settings
        rng = np.random.default_rng(seed)
        others = [np.delete(np.arange(size), index) for index in range(size)]
        partners = np.array([rng.choice(choices, size=s, replace=False) for choices in others]).reshape(size, s)
        genes = rng.integers(0, n, size=(size, n, s, c))
        tables = rng.random((size, n, 2 ** (1 + k + c * s)))
        return partners.astype(float), genes.astype(float), tables


class NKCSSystem(BaseSystem):
    """The NKCS model of Kauffman and Johnsen, species climbing NK landscapes coupled to each other's genes.

    Every gene of a species depends on `k` genes of its own species and on `c` genes in each of `s` partner species. A
    species keeps a point mutation when it raises its fitness, which changes the landscapes of the species that depend
    on it, so adaptations trigger coevolutionary avalanches. Rugged landscapes, `k` above about `c s`, freeze the
    ecosystem once every species sits on a peak, and smoother ones keep it in a chaotic regime of endless adaptation.
    The system has the genomes `species.genome`, fitnesses `species.fitness`, and accepted mutations `species.moves` of
    the group `species`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.evolution import NKCSSystem
        >>> NKCSSystem(5).prepare().system_.elements[0].prepare().size_
        5
    """

    field = 'biology/evolution'
    topics = ('coevolution', 'fitness landscape', 'avalanches', 'criticality', 'agents')
    references = ('Kauffman and Johnsen 1991', 'Thurner et al. 2018')
    until = 200

    def __init__(
        self,
        n_species: int = 10,
        *,
        n: int = 12,
        k: int = 2,
        c: int = 1,
        s: int = 2,
        seed: int = 0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_species = n_species
        self.n = n
        self.k = k
        self.c = c
        self.s = s
        self.seed = seed
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the NKCS model of Kauffman and Johnsen."""
        genomes = np.random.default_rng(self.random_state).integers(0, 2**self.n, size=self.n_species).astype(float)
        species = GroupElement(
            'species',
            element=CoevolvingSpeciesElement,
            size=self.n_species,
            values={'genome': genomes},
        )
        climbing = CoevolutionaryClimbingInteraction(
            'climbing',
            species='species',
            n=self.n,
            k=self.k,
            c=self.c,
            s=self.s,
            seed=self.seed,
        )
        return System(self.label, elements=[species], interactions=[climbing], time=DiscreteTime())
