"""The binary perceptron as a constraint satisfaction problem.

Its satisfiability collapses at a critical load.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import SatisfiabilityInteraction
from ....core.components.spaces import DiscreteTime
from ....core.components.system import BaseSystem, System
from ....core.errors import ValidationError
from ....core.specs import CardinalitySpec, StateSpec


class BinaryWeightElement(Element):
    """Weight of a binary perceptron.

    The state 1 stands for `+1` and the state 0 for `-1`.

    Args:
        label:
            The label of the weight or the group.

        on:
            Whether the weight is `+1`.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import BinaryWeightElement
        >>> BinaryWeightElement('weight', on=1.0).on
        1.0
    """

    def __init__(self, label: str, on: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0) -> None:
        super().__init__(label)
        self.on = on


class PerceptronSatisfiabilitySystem(BaseSystem):
    """A binary perceptron that must store random patterns.

    The patterns are revealed one per step, as cardinality constraints. A pattern `xi` of output `sigma` holds when
    `sigma sum_i w_i xi_i > 0`, so with an odd number `N` of weights `w_i = +-1`, when at least `(N + 1) / 2` of the
    literals `w_i = sigma xi_i` hold. The patterns are satisfiable up to the critical load `p / N` of about 0.83 in the
    limit of many weights (Bertin 2021 §6.3.3, Ex. 6.4). The patterns are drawn from `random_state`, so a disorder
    average sweeps it with `sweep`. The system has the group `weights` and the reports `decide.satisfiable` and
    `decide.revealed`.

    Args:
        n_weights:
            The number of weights, which is odd.

        load:
            The number of patterns over the number of weights.

        random_state:
            The seed or generator of the patterns.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The number of weights is even, so a pattern can tie.

    Examples:
        >>> from skcomplex.systems.computer_science.neural_networks import PerceptronSatisfiabilitySystem
        >>> PerceptronSatisfiabilitySystem(3, load=1.0).prepare().system_.interactions[0].constraints[0].bound
        2
    """

    field = 'computer_science/neural_networks'
    topics = ('perceptron', 'satisfiability', 'disorder', 'critical load')
    references = ('Bertin 2021', 'Krauth and Mezard 1989')
    until = 31

    def __init__(
        self,
        n_weights: int = 21,
        *,
        load: float = 1.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_weights = n_weights
        self.load = load
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system, built from its model parameters."""
        if self.n_weights % 2 == 0:
            error_msg = (
                f'perceptron_satisfiability has n_weights={self.n_weights!r}. Use an odd number, so no pattern ties.'
            )
            raise ValidationError(error_msg)
        rng = np.random.default_rng(self.random_state)
        n_patterns = round(self.load * self.n_weights)
        patterns = rng.choice([-1, 1], size=(n_patterns, self.n_weights)) * rng.choice([-1, 1], size=(n_patterns, 1))
        indices = np.arange(1, self.n_weights + 1)
        constraints = [
            CardinalitySpec(tuple((indices * pattern).tolist()), (self.n_weights + 1) // 2, 'at-least')
            for pattern in patterns
        ]
        weights = GroupElement('weights', element=BinaryWeightElement, size=self.n_weights)
        decide = SatisfiabilityInteraction('decide', target='weights', state='on', constraints=constraints)
        return System(self.label, elements=[weights], interactions=[decide], time=DiscreteTime())
