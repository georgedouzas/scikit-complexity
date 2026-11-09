"""Evolution systems: monomorphic populations that walk a fitness landscape.

They walk by the origin and fixation of mutants.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import ZZ, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import TransitionInteraction
from ....core.components.laws import BaseLaw, JumpLaw
from ....core.components.laws.utils import total
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, RunSettings, System
from ....core.specs import ObservableSpec, ParameterSpec, RoleSpec, StateSpec, Tags, VisualSpec
from ....core.types import Array


class LocusElement(Element):
    """Locus of the genome of a monomorphic population, with an allele of `-1` or `1`.

    Args:
        label:
            The label of the locus or the group of loci.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        s:
            The allele, `-1` or `1`.

        h:
            The random field of the locus, the fitness gained by the allele `1` over the allele `-1`, halved.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.evolution import LocusElement
        >>> GroupElement('genome', element=LocusElement, size=10, values={'h': 0.05}).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('image', {'low': -1.0, 'high': 1.0, 'history': 64}),
        *,
        s: Annotated[float, StateSpec(domain=ZZ)] = -1.0,
        h: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s = s
        self.h = h


class SubstitutionInteraction(TransitionInteraction):
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
        target: Annotated[str, RoleSpec()],
        N: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        F0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        couplings: npt.ArrayLike | None = None,
        state: str = 's',
    ) -> None:
        super().__init__(label, target=target, state=state, source=-1.0, destination=1.0)
        self.N = N
        self.mu = mu
        self.F0 = F0
        self.couplings = couplings

    def _check_transition(self) -> None:
        """Leave the transition unchecked.

        Its rates derive from the parameters of the population.
        """

    def _get_fitness(self, namespace: SimpleNamespace) -> tuple[Expression, Expression]:
        """Return the local field `h_i + sum_j J_ij s_j` of a locus and the genome fitness."""
        genome = namespace.target
        alleles = getattr(genome, self.state)
        if self.couplings is None:
            return genome.h, namespace.F0 + total(genome.h * alleles)
        couplings = np.asarray(self.couplings, dtype=float)
        return genome.h + total(alleles, weights=couplings), SubstitutionInteraction._fitness_of(
            genome,
            namespace.F0,
            couplings,
        )

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the substitutions of the allele of every locus.

        They go from `-1` to `1` and back.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`.

        Returns:
            laws:
                The jumps of the allele.
        """
        s = namespace
        alleles = getattr(s.target, self.state)
        local, fitness = self._get_fitness(s)
        ratio = fitness / (fitness - 2 * alleles * local)
        rate = s.N * s.mu * abs((1 - ratio) / (1 - ratio**s.N))
        return [self._get_jump(alleles, -1.0, 1.0, rate), self._get_jump(alleles, 1.0, -1.0, rate)]

    @staticmethod
    def _fitness_of(genome: SimpleNamespace, F0: Expression, couplings: Array) -> Expression:
        """Return the fitness of the genome of a group of loci.

        The fitness is `F0 + sum_i h_i s_i + sum_ij J_ij s_i s_j / 2`.
        """
        return F0 + total(genome.h * genome.s) + total(genome.s * total(genome.s, weights=couplings)) / 2


def _field_share(member: SimpleNamespace) -> Expression:
    """Return the share `h_i s_i` of one locus in the fitness of the genome."""
    return member.h * member.s


def _coupling_share(alleles: Array, couplings: Array) -> float:
    """Return the share `sum_ij J_ij s_i s_j / 2` of the couplings in the fitness.

    The couplings are between loci, in the fitness of the genome.
    """
    return float(alleles @ couplings @ alleles / 2)


def _genome_fitness(s: SimpleNamespace) -> Expression:
    """Return the fitness of the genome.

    It is the base fitness of the interaction `substitution` and the shares of loci.
    """
    return s.substitution.F0 + s.field_share + s.coupling_share


def _walk_observables(system: System) -> list[ObservableSpec]:
    """Return the shares of the fields and the couplings, and the genome fitness.

    The shares are those in the fitness of the genome.
    """
    couplings = system.interactions[0].couplings
    loci = system.elements[0].prepare().size_
    matrix = np.zeros((loci, loci)) if couplings is None else np.asarray(couplings)
    return [
        ObservableSpec('field_share', _field_share, group='genome', reduce='sum'),
        ObservableSpec(
            'coupling_share',
            group='genome',
            variable='s',
            reduce=partial(_coupling_share, couplings=matrix),
        ),
        ObservableSpec('fitness', _genome_fitness),
    ]


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

    Examples:
        >>> from skcomplex.systems.biology.evolution import OriginFixationWalkSystem
        >>> OriginFixationWalkSystem(8).prepare().system_.elements[0].prepare().size_
        8
    """

    field = 'biology/evolution'
    topics = ('population genetics', 'fitness landscape', 'jump process', 'fixation', 'strong selection weak mutation')
    references = ('Bertin 2021', 'Sella and Hirsh 2005', 'Drossel 2001')
    until = 640.0
    dt = 10.0

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
    ) -> None:
        super().__init__(label)
        self.loci = loci
        self.N = N
        self.mu = mu
        self.field_scale = field_scale
        self.coupling = coupling
        self.F0 = F0
        self.random_state = random_state

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        They include the shares and the fitness of the genome of the built parts.

        Returns:
            settings:
                The tuned run settings by name.
        """
        settings = super().get_run_settings()
        settings['observables'] = _walk_observables(self._build_system())
        return settings

    def _build_system(self) -> System:
        """Return the parts of a monomorphic population."""
        rng = np.random.default_rng(self.random_state)
        fields = rng.normal(0.0, self.field_scale, self.loci)
        upper = np.triu(rng.normal(0.0, self.coupling / np.sqrt(self.loci), (self.loci, self.loci)), 1)
        matrix = upper + upper.T
        couplings = matrix if self.coupling else None
        genome = GroupElement(
            'genome',
            element=LocusElement,
            space=LatticeSpace(self.loci),
            values={'s': -np.ones(self.loci), 'h': fields},
            on='sites',
        )
        return System(
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
