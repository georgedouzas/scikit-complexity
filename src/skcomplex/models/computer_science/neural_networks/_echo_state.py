"""Neural network models: the echo state network, a reservoir of leaky tanh units driven by an input."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import oo, sin, tanh
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, UpdateLaw
from ....core.laws.utils import total
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ..._catalog import register_model


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


class ReservoirUnitElement(Element):
    """Leaky tanh unit of an echo state network, driven by the other units and by a sine input.

    The unit follows `r(t + 1) = (1 - a) r + a tanh(sum_j W_ij r_j + w u(t))` with the input `u(t) = A sin(f t)`.

    Args:
        label:
            The label of the unit or the group.

        weights:
            The recurrent weights `W`, one row and one column per unit.

        **values:
            The values of the declared variables, by name:

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

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import ReservoirUnitElement
        >>> ReservoirUnitElement('unit').leak
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        r: Annotated[float, StateSpec()] = 0.0,
        w_in: Annotated[float, ParameterSpec()] = 1.0,
        leak: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.3,
        amplitude: Annotated[float, ParameterSpec()] = 1.0,
        frequency: Annotated[float, ParameterSpec()] = 0.2,
        weights: npt.ArrayLike | None = None,
    ) -> None:
        super().__init__(label, visual=visual)
        self.r = r
        self.w_in = w_in
        self.leak = leak
        self.amplitude = amplitude
        self.frequency = frequency
        self.weights = weights

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the leaky update of the activity.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        drive = total(s.r, weights=self.weights) + s.w_in * s.amplitude * sin(s.frequency * s.t)
        return [UpdateLaw(s.r, (1 - s.leak) * s.r + s.leak * tanh(drive))]


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
    readout trained on these activities, as `skcomplex.analysis.ReservoirAnalysis` does, turns the reservoir
    into a forecaster.

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
        >>> echo_state_network(50).elements[0].check().prepare().size_
        50
    """
    rng = np.random.default_rng(random_state)
    weights = _recurrent_weights(n_units, connectivity, spectral_radius, rng)
    units = GroupElement(
        'units',
        element=ReservoirUnitElement,
        size=n_units,
        values={
            'r': rng.uniform(-1.0, 1.0, n_units),
            'w_in': rng.uniform(-input_scaling, input_scaling, n_units),
            'leak': leak,
            'amplitude': amplitude,
            'frequency': frequency,
            'weights': weights,
        },
    )
    return System(elements=[units])
