"""Define the life cycle of the Tangled Nature system, whose births read the composition of the population."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import lru_cache
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet
from scipy.special import expit

from ....core.components.interactions import LifeCycleInteraction
from ....core.components.types import SystemViewProtocol
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, RoleSpec
from ....core.types import Array
from ._individual_view import IndividualView


class TangledNatureLifeCycleInteraction(LifeCycleInteraction):
    """Life cycle of the organisms of the Tangled Nature system, run by the visits of a phase drawn with replacement.

    A visited organism of type `S` reproduces with probability `1 / (1 + exp(-H(S)))`, where
    `H(S) = k / N sum_S' J(S, S') n(S') - mu N` reads the composition `n` of the population of `N` organisms, and is
    replaced by two offspring, whose genome bits flip with probability `p_mut` each. Then a random organism dies with
    probability `p_kill`. The couplings `J` are uniform in [-1, 1] between different types, 0 within a type, and fixed
    by `seed`.

    Args:
        label:
            The label of the life cycle.

        target:
            The path of the variable group of organisms.

        k:
            The strength of the couplings.

        mu:
            The limit of resources, which lowers reproduction in large populations.

        p_mut:
            The probability that a bit of an offspring's genome flips.

        p_kill:
            The probability that a visit kills a random organism.

        length:
            The number of bits of a genome.

        seed:
            The seed of the couplings.

        phase:
            The phase whose visits run the life cycle.

    Examples:
        >>> from skcomplex.systems.biology.evolution import TangledNatureLifeCycleInteraction
        >>> TangledNatureLifeCycleInteraction('evolution', target='organisms').offspring
        2
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        p_mut: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        p_kill: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        length: int = 8,
        seed: int = 0,
        phase: str | None = 'generation',
    ) -> None:
        super().__init__(
            label,
            target=target,
            birth=self.compute_birth,
            offspring=2,
            inherit=('trait',),
            mutate=self.mutate_genome,
            schedule='member',
            phase=phase,
            replace=True,
            cull=self.compute_kill,
            pick='random',
        )
        self.k = k
        self.mu = mu
        self.p_mut = p_mut
        self.p_kill = p_kill
        self.length = length
        self.seed = seed

    @staticmethod
    @lru_cache(maxsize=16)
    def _get_couplings(seed: int, length: int) -> Array:
        """Return the random couplings `J(S, S')`, uniform in [-1, 1] and 0 within a type."""
        size = 2**length
        couplings = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(size, size))
        np.fill_diagonal(couplings, 0.0)
        return couplings

    def compute_birth(self, agent: IndividualView, system: SystemView) -> float:
        """Return the probability `1 / (1 + exp(-H(S)))` that an organism of type `S` reproduces.

        Args:
            agent:
                The organism.

            system:
                The system view of the life cycle.

        Returns:
            probability:
                The probability of reproduction.
        """
        composition = system.composition('target', 'trait')
        size = sum(composition.values())
        row = self._get_couplings(self.seed, self.length)[int(agent.trait)]
        coupling = sum(row[int(np.asarray(genome))] * count for genome, count in composition.items())
        return float(expit(system.parameter('k') / size * coupling - system.parameter('mu') * size))

    def compute_kill(self, agent: IndividualView, system: SystemView) -> float:
        """Return the probability `p_kill` that a visit kills a random organism.

        Args:
            agent:
                The visited organism.

            system:
                The system view of the life cycle.

        Returns:
            probability:
                The probability of a death.
        """
        return system.parameter('p_kill')

    def mutate_genome(self, values: dict[str, object], system: SystemViewProtocol) -> dict[str, object]:
        """Return the values of an offspring whose every genome bit flips with probability `p_mut`.

        Args:
            values:
                The values of the offspring, with the genome of its parent as `trait`.

            system:
                The system view of the life cycle.

        Returns:
            values:
                The values with the mutated genome.
        """
        genome = int(np.asarray(values['trait'], dtype=float))
        flips = np.flatnonzero(system.rng.random(self.length) < system.parameter('p_mut'))
        return {**values, 'trait': float(genome ^ sum(1 << int(locus) for locus in flips))}
