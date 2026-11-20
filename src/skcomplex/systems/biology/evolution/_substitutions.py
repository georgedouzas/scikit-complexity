"""Evolution systems: monomorphic populations that walk a fitness landscape.

They walk by the origin and fixation of mutants.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from functools import partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable, Tags
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.components.laws.utils import total, where
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role


class LocusElement(Element):
    """Locus of the genome of a monomorphic population, with an allele of `-1` or `1`.

    Args:
        label:
            The label of the locus or the group of loci.

        s:
            The allele, `-1` or `1`.

        h:
            The random field of the locus, the fitness gained by the allele `1` over the allele `-1`, halved.

        space:
            The lattice whose sites hold one locus each, or `None` for one locus.

    Examples:
        >>> from skcomplex.core.components.spaces import LatticeSpace
        >>> from skcomplex.systems.biology.evolution import LocusElement
        >>> LocusElement('genome', h=0.05, space=LatticeSpace(10)).h
        0.05
    """

    visual = VisualSpec('image', {'low': -1.0, 'high': 1.0, 'history': 64})

    def __init__(
        self,
        label: str,
        *,
        s: Annotated[float | Array, ZZ] = -1.0,
        h: float | Array = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.s = s
        self.h = h


class SubstitutionInteraction(Interaction):
    """Origin and fixation of mutants that flip one locus of a genome.

    The genome is the one of a monomorphic population.

    Every locus of each of the `N` individuals mutates at the rate `mu`, so mutants of a given locus arise at the rate
    `N mu`. A mutant from the fitness `f` to `f'` fixes with the Moran probability
    `(1 - f / f') / (1 - (f / f')^N)`, so the allele `s` of the locus moves between `-1` and `1` by transitions at the
    product of the two. The fitness is `F0 + sum_i h_i s_i + sum_ij J_ij s_i s_j / 2`, and a flip of locus `i` changes
    it by `-2 s_i (h_i + sum_j J_ij s_j)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of loci.

        N:
            The size of the population.

        mu:
            The rate at which one locus of an individual mutates.

        F0:
            The fitness of a genome without fields and couplings, above the sum of the `|h_i|` and `|J_ij|` so that
            every fitness is positive.

        couplings:
            The symmetric matrix `J` of couplings between loci, with a zero diagonal, or `None` for no couplings.

        state:
            The name of the state of the loci that holds the allele.

    Examples:
        >>> from skcomplex.systems.biology.evolution import SubstitutionInteraction
        >>> SubstitutionInteraction('substitution', target='genome', N=50.0).N
        50.0
    """

    law_classes = (JumpLaw,)

    def _get_tags(self) -> Tags:
        """Return the tags of the jumps at a rate, the only laws of the interaction."""
        return self._get_variant_tags(self.law_classes)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        N: Annotated[float, RealSet.open(0, oo)] = 100.0,
        mu: Annotated[float, RealSet.closed_open(0, oo)] = 0.01,
        F0: Annotated[float, RealSet.open(0, oo)] = 1.0,
        couplings: npt.ArrayLike | None = None,
        state: str = 's',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.N = N
        self.mu = mu
        self.F0 = F0
        self.couplings = couplings
        self.state = state

    def _get_fitness(self, namespace: SimpleNamespace) -> tuple[Expression, Expression]:
        """Return the local field `h_i + sum_j J_ij s_j` of a locus and the genome fitness."""
        genome = namespace.target
        alleles = getattr(genome, self.state)
        if self.couplings is None:
            return genome.h, namespace.F0 + total(genome.h * alleles)
        couplings = np.asarray(self.couplings, dtype=float)
        return genome.h + total(alleles, weights=couplings), SubstitutionInteraction._get_fitness_of(
            genome,
            namespace.F0,
            couplings,
        )

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the substitutions of the allele of every locus, from `-1` to `1` and back.

        Args:
            context:
                The namespace of the genome, with `context.target`.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        s = context
        alleles = getattr(s.target, self.state)
        local, fitness = self._get_fitness(s)
        ratio = fitness / (fitness - 2 * alleles * local)
        rate = s.N * s.mu * abs((1 - ratio) / (1 - ratio**s.N))
        self.laws_: list[BaseLaw] = [
            JumpLaw(rate * where(alleles == -1.0, 1, 0), {alleles: 2.0}),
            JumpLaw(rate * where(alleles == 1.0, 1, 0), {alleles: -2.0}),
        ]

    @staticmethod
    def _get_fitness_of(genome: SimpleNamespace, F0: Expression, couplings: Array) -> Expression:
        """Return the fitness of the genome of a group of loci.

        The fitness is `F0 + sum_i h_i s_i + sum_ij J_ij s_i s_j / 2`.
        """
        return F0 + total(genome.h * genome.s) + total(genome.s * total(genome.s, weights=couplings)) / 2


class OriginFixationWalkSystem(BaseSystem):
    """A monomorphic population that climbs a fitness landscape.

    Mutants arise and fix one at a time.

    Under strong selection and weak mutation the population carries one genome of `loci` alleles `-1` or `1`, and every
    substitution of a mutant moves it to a neighbouring genome. The walk climbs the landscape and settles on the
    stationary distribution `Q ~ f^(N - 1)`, a Boltzmann distribution of the energy `-ln f` at the temperature `1 / (N -
    1)`. Random fields alone, `coupling = 0`, give a landscape with one peak. Couplings `J_ij` of variance `coupling^2 /
    loci` give the rugged landscape of a spin glass, where a small population escapes local peaks. The system has the
    alleles `genome.s` on a one-dimensional lattice of loci, starting from the genome of all `-1`, and the fitness
    `fitness` of the genome.

    Args:
        loci:
            The number of loci of the genome.

        N:
            The size of the population.

        mu:
            The rate at which one locus of an individual mutates.

        field_scale:
            The standard deviation of the random fields `h_i`.

        coupling:
            The scale of the random couplings `J_ij`, or 0 for none.

        F0:
            The fitness of a genome without fields and couplings, above the sum of the `|h_i|` and `|J_ij|` so that
            every fitness is positive.

        random_state:
            The seed or generator of the fields and couplings.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.evolution import OriginFixationWalkSystem
        >>> OriginFixationWalkSystem(8).prepare().system_.elements[0].h.shape
        (8,)
    """

    field = 'biology/evolution'
    topics = ('population genetics', 'fitness landscape', 'jump process', 'fixation', 'strong selection weak mutation')
    references = ('Bertin 2021', 'Sella and Hirsh 2005', 'Drossel 2001')
    until = 640.0
    dt = 10.0

    @staticmethod
    def _get_field_share(member: SimpleNamespace) -> Expression:
        """Return the share `h_i s_i` of one locus in the fitness of the genome."""
        return member.h * member.s

    @staticmethod
    def _get_coupling_share(alleles: Array, couplings: Array) -> float:
        """Return the share `sum_ij J_ij s_i s_j / 2` of the couplings in the fitness.

        The couplings are between loci, in the fitness of the genome.
        """
        return float(alleles @ couplings @ alleles / 2)

    @staticmethod
    def _get_genome_fitness(s: SimpleNamespace) -> Expression:
        """Return the fitness of the genome.

        It is the base fitness of the interaction `substitution` and the shares of loci.
        """
        return s.substitution.F0 + s.field_share + s.coupling_share

    @staticmethod
    def _walk_observables(loci: int, couplings: Array | None) -> list[Observable]:
        """Return the shares of the fields and the couplings, and the genome fitness.

        The shares are those in the fitness of the genome.
        """
        matrix = np.zeros((loci, loci)) if couplings is None else np.asarray(couplings)
        return [
            Observable('field_share', OriginFixationWalkSystem._get_field_share, group='genome', reduce='sum'),
            Observable(
                'coupling_share',
                group='genome',
                variable='s',
                reduce=partial(OriginFixationWalkSystem._get_coupling_share, couplings=matrix),
            ),
            Observable('fitness', OriginFixationWalkSystem._get_genome_fitness),
        ]

    def __init__(
        self,
        loci: int = 96,
        *,
        N: float = 100.0,
        mu: float = 0.01,
        field_scale: float = 0.01,
        coupling: float = 0.0,
        F0: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.loci = loci
        self.N = N
        self.mu = mu
        self.field_scale = field_scale
        self.coupling = coupling
        self.F0 = F0
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a monomorphic population.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        fields = rng.normal(0.0, self.field_scale, self.loci)
        upper = np.triu(rng.normal(0.0, self.coupling / np.sqrt(self.loci), (self.loci, self.loci)), 1)
        matrix = upper + upper.T
        couplings = matrix if self.coupling else None
        genome = LocusElement('genome', s=-np.ones(self.loci), h=fields, space=LatticeSpace(self.loci))
        self.system_ = System(
            self.label,
            elements=[genome],
            interactions=[
                SubstitutionInteraction(
                    'substitution',
                    target='genome',
                    N=self.N,
                    mu=self.mu,
                    F0=self.F0,
                    couplings=couplings,
                ),
            ],
        )
        self.observables_ = tuple(OriginFixationWalkSystem._walk_observables(self.loci, couplings))
