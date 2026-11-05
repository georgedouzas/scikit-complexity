"""Agent that walks an NK fitness landscape, flipping one random bit of its genome when that raises its fitness."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import lru_cache
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ._climber_view import ClimberView


class ClimberElement(Element):
    """Agent that walks an NK fitness landscape, flipping one random bit of its genome when that raises its fitness.

    The fitness of a genome of `n` bits is the mean of the contributions of its bits, each a random function of the bit
    and the next `k` bits, fixed by `seed`.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        genome:
            The bits of the genome.

        fitness:
            The fitness of the genome.

        n:
            The number of bits.

        k:
            The number of other bits each bit interacts with.

        seed:
            The seed of the landscape.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.evolution import ClimberElement
        >>> GroupElement('climbers', element=ClimberElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'fitness', 'color_range': (0.0, 1.0)}),
        *,
        genome: Annotated[float, StateSpec(domain=NN)] = 0.0,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        k: Annotated[float, ParameterSpec(domain=NN)] = 2.0,
        seed: Annotated[float, ParameterSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.genome = genome
        self.fitness = fitness
        self.n = n
        self.k = k
        self.seed = seed

    @classmethod
    def _get_view(cls) -> type[ClimberView]:
        """Return the view of a climber, `ClimberView`."""
        return ClimberView

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

    def climb(self, agent: ClimberView, system: SystemView) -> None:
        """Flip a random bit, and keep the flip when it raises the fitness.

        Args:
            agent:
                The climber.

            system:
                The system.
        """
        n, k, seed = int(agent.n), int(agent.k), int(agent.seed)
        mutant = int(agent.genome) ^ (1 << int(system.rng.integers(n)))
        fitness = ClimberElement._nk_fitness(mutant, n, k, seed)
        if fitness > agent.fitness:
            agent.genome = float(mutant)
            agent.fitness = fitness

    @staticmethod
    @lru_cache(maxsize=16)
    def _nk_tables(seed: int, n: int, k: int) -> Array:
        """Return the fitness contributions of every locus for every configuration of itself and its `k` neighbours."""
        return np.random.default_rng(seed).random((n, 2 ** (k + 1)))

    @staticmethod
    def _nk_fitness(genome: int, n: int, k: int, seed: int) -> float:
        """Return the NK fitness of a genome, the mean contribution of its loci, each read with the next `k` loci."""
        tables = ClimberElement._nk_tables(seed, n, k)
        bits = [(genome >> locus) & 1 for locus in range(n)]
        total = 0.0
        for locus in range(n):
            index = sum(bits[(locus + offset) % n] << offset for offset in range(k + 1))
            total += tables[locus, index]
        return total / n
