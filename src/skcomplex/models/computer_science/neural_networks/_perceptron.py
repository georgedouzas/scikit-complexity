"""The binary perceptron as a constraint satisfaction problem, whose satisfiability collapses at a critical load."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import SatisfiabilityInteraction
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....errors import ValidationError
from ....specs import CardinalitySpec
from ..._catalog import register_model
from ._binary_weight import BinaryWeightElement


@register_model(
    field='computer_science/neural_networks',
    tags=('perceptron', 'satisfiability', 'disorder', 'critical load'),
    references=('Bertin 2021', 'Krauth and Mezard 1989'),
    defaults={'until': 31},
)
def perceptron_satisfiability(
    n_weights: int = 21,
    *,
    load: float = 1.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a binary perceptron that must store random patterns, revealed one per step, as cardinality constraints.

    A pattern `xi` of output `sigma` holds when `sigma sum_i w_i xi_i > 0`, so with an odd number `N` of weights
    `w_i = +-1`, when at least `(N + 1) / 2` of the literals `w_i = sigma xi_i` hold. The patterns are satisfiable up to
    the critical load `p / N` of about 0.83 in the limit of many weights (Bertin 2021 §6.3.3, Ex. 6.4). The patterns are
    drawn from `random_state`, so a disorder average sweeps it through `GridDesign`.

    Args:
        n_weights:
            The number of weights, which is odd.

        load:
            The number of patterns over the number of weights.

        random_state:
            The seed or generator of the patterns.

    Returns:
        system:
            The model, with the group `weights` and the reports `decide.satisfiable` and `decide.revealed`.

    Raises:
        ValidationError:
            The number of weights is even, so a pattern can tie.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import perceptron_satisfiability
        >>> perceptron_satisfiability(3, load=1.0).interactions[0].constraints[0].bound
        2
    """
    if n_weights % 2 == 0:
        error_msg = f'perceptron_satisfiability has n_weights={n_weights!r}. Use an odd number, so no pattern ties.'
        raise ValidationError(error_msg)
    rng = np.random.default_rng(random_state)
    n_patterns = round(load * n_weights)
    patterns = rng.choice([-1, 1], size=(n_patterns, n_weights)) * rng.choice([-1, 1], size=(n_patterns, 1))
    indices = np.arange(1, n_weights + 1)
    constraints = [
        CardinalitySpec(tuple((indices * pattern).tolist()), (n_weights + 1) // 2, 'at-least') for pattern in patterns
    ]
    weights = GroupElement('weights', element=BinaryWeightElement, size=n_weights)
    decide = SatisfiabilityInteraction('decide', target='weights', state='on', constraints=constraints)
    return System(elements=[weights], interactions=[decide], time=DiscreteTime())
