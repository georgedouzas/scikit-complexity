"""Origin and fixation of mutants that flip one locus of the genome of a monomorphic population."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import TransitionInteraction
from ....core.components.laws import BaseLaw
from ....core.components.laws.utils import total
from ....core.specs import ParameterSpec, RoleSpec
from ....core.types import Array


class SubstitutionInteraction(TransitionInteraction):
    """Origin and fixation of mutants that flip one locus of the genome of a monomorphic population.

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
        """Leave the transition unchecked, since its rates derive from the parameters of the population."""

    def _get_fitness(self, namespace: SimpleNamespace) -> tuple[Expression, Expression]:
        """Return the local field `h_i + sum_j J_ij s_j` of a locus and the fitness of the genome."""
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
        """Return the substitutions of the allele of every locus, from `-1` to `1` and back.

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
        """Return the fitness `F0 + sum_i h_i s_i + sum_ij J_ij s_i s_j / 2` of the genome of a group of loci."""
        return F0 + total(genome.h * genome.s) + total(genome.s * total(genome.s, weights=couplings)) / 2
