"""Neural network models: the echo state network, a reservoir of leaky tanh units driven by an input."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import sin, tanh

from ....core.base import Domain, parameter, state
from ....core.components import Element
from ....core.laws import Law, Symbols, Update, total
from ....core.runtime import register_model
from ....core.system import System


def _recurrent_weights(
    n_units: int,
    connectivity: float,
    spectral_radius: float,
    rng: np.random.Generator,
) -> scipy.sparse.csr_array:
    """Return sparse Gaussian recurrent weights scaled to a spectral radius."""
    weights = scipy.sparse.random_array(
        (n_units, n_units),
        density=connectivity,
        format='csr',
        rng=rng,
        data_sampler=rng.standard_normal,
    )
    radius = np.abs(np.linalg.eigvals(weights.toarray())).max()
    return scipy.sparse.csr_array(weights * (spectral_radius / radius))


class ReservoirUnit(Element):
    """Leaky tanh unit of an echo state network, driven by the other units and by a sine input.

    The unit follows `r(t + 1) = (1 - a) r + a tanh(sum_j W_ij r_j + w u(t))` with the input `u(t) = A sin(f t)`.

    Args:
        label:
            The label of the unit or the group.

        r:
            The initial activity.

        w_in:
            The weight `w` of the input.

        leak:
            The leak rate `a`, 1 for units without memory of their own.

        amplitude:
            The amplitude `A` of the input.

        frequency:
            The angular frequency `f` of the input, per step.

        weights:
            The recurrent weights `W`, one row and one column per unit.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import ReservoirUnit
        >>> ReservoirUnit('unit').leak
        0.3
    """

    r: float = state(default=0.0)
    w_in: float = parameter(default=1.0)
    leak: float = parameter(default=0.3, domain=Domain.POSITIVE)
    amplitude: float = parameter(default=1.0)
    frequency: float = parameter(default=0.2)
    weights: Any = None

    def laws(self, s: Symbols) -> list[Law]:
        """Return the leaky update of the activity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        drive = total(s.r, weights=self.weights) + s.w_in * s.amplitude * sin(s.frequency * s.t)
        return [Update(s.r, (1 - s.leak) * s.r + s.leak * tanh(drive))]


@register_model(
    field='computer_science/neural_networks',
    tags=('neural network', 'reservoir computing', 'echo state', 'discrete map'),
    references=('Jaeger 2001', 'Cestnik and Martens 2025'),
    defaults={'until': 200, 'dt': 1},
)
def echo_state_network(
    n_units: int = 200,
    *,
    spectral_radius: float = 0.9,
    leak: float = 0.3,
    connectivity: float = 0.1,
    input_scaling: float = 1.0,
    amplitude: float = 1.0,
    frequency: float = 0.2,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return an echo state network, a reservoir of leaky tanh units with random recurrent weights driven by a sine.

    The recurrent weights are sparse and Gaussian, scaled to a spectral radius below 1, so the reservoir has the echo
    state property: its activities forget their initial values and become a function of the input history alone. A
    readout trained on these activities, as `skcomplex.core.analysis.fit_reservoir` does, turns the reservoir into a
    forecaster.

    Args:
        n_units:
            The number of units.

        spectral_radius:
            The largest modulus of the eigenvalues of the recurrent weights.

        leak:
            The leak rate of every unit, 1 for units without memory of their own.

        connectivity:
            The share of the recurrent weights that are not zero.

        input_scaling:
            The scale of the input weights, drawn uniformly from `[-input_scaling, input_scaling]`.

        amplitude:
            The amplitude of the sine input.

        frequency:
            The angular frequency of the sine input, per step.

        random_state:
            The seed or generator of the weights and of the initial activities.

    Returns:
        system:
            The model, with the group `units` of activities `r`, one per unit.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import echo_state_network
        >>> echo_state_network(50).elements[0].size
        50
    """
    rng = np.random.default_rng(random_state)
    weights = _recurrent_weights(n_units, connectivity, spectral_radius, rng)
    units = ReservoirUnit.group(
        'units',
        n_units,
        r=rng.uniform(-1.0, 1.0, n_units),
        w_in=rng.uniform(-input_scaling, input_scaling, n_units),
        leak=leak,
        amplitude=amplitude,
        frequency=frequency,
        weights=weights,
    )
    return System(components=[units])
