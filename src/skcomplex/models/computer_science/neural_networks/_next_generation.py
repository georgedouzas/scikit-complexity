"""Next-generation reservoir computer: lagged variables and the ridge readout of their polynomial features."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ....analysis import ReservoirAnalysis
from ....core.elements import DeclaredElement, Element
from ....core.interactions import MapInteraction, RateInteraction
from ....core.laws.utils import lag
from ....core.spaces import ContinuousTime, DiscreteTime
from ....core.system import System
from ....engine.runtime import Simulation
from ....engine.solvers import ScipyODESolver
from ....errors import ValidationError
from ....specs import DeclaredVariableSpec, StateSpec
from ....types import Array
from ..._catalog import register_model

LORENZ_START = (-8.0, 8.0, 27.0)
LORENZ_TRANSIENT = 5.0
LORENZ_RTOL, LORENZ_ATOL = 1e-3, 1e-6
LORENZ_VARIABLES = ('x', 'y', 'z')
REQUIRED_KIND = 'next_generation'


class LorenzFlowElement(Element):
    """Point of the Lorenz flow that trains a reservoir, with the three variables of the flow.

    Args:
        label:
            The label of the point.

        x:
            The initial value of the first variable.

        y:
            The initial value of the second variable.

        z:
            The initial value of the third variable.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import LorenzFlowElement
        >>> LorenzFlowElement('flow', z=2.0).z
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        x: Annotated[float, StateSpec()] = LORENZ_START[0],
        y: Annotated[float, StateSpec()] = LORENZ_START[1],
        z: Annotated[float, StateSpec()] = LORENZ_START[2],
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z


def _lorenz_series(samples: int, step: float, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> Array:
    """Return samples of the Lorenz flow on its attractor, one row per sample and one column per variable."""
    rates: dict[str, Callable[[SimpleNamespace], Expression]] = {
        'x': lambda s: sigma * (s.target.y - s.target.x),
        'y': lambda s: s.target.x * (rho - s.target.z) - s.target.y,
        'z': lambda s: s.target.x * s.target.y - beta * s.target.z,
    }
    system = System(
        elements=[LorenzFlowElement('flow')],
        interactions=[RateInteraction(name, target='flow', state=name, rate=rate) for name, rate in rates.items()],
        time=ContinuousTime(),
    )
    solver = ScipyODESolver(method='RK23', rtol=LORENZ_RTOL, atol=LORENZ_ATOL)
    simulation = Simulation(system, solver=solver, dt=step).check().prepare()
    trajectory = simulation.run(until=LORENZ_TRANSIENT + step * (samples - 1)).trajectory_
    values: Array = np.column_stack(
        [np.asarray(trajectory.values(f'flow.{name}'), dtype=np.float64) for name in LORENZ_VARIABLES],
    )
    return values[-samples:]


def _reservoir_element(initial: Mapping[str, Sequence[float]]) -> DeclaredElement:
    """Return the element of scalar states with lags, from the values of every state and of its lags."""
    variables = {
        name: DeclaredVariableSpec(
            float(values[0]),
            StateSpec(lags=len(values) - 1, initial=tuple(float(value) for value in values[1:])),
        )
        for name, values in initial.items()
    }
    return DeclaredElement('reservoir', variables=variables)


def _increment(
    s: SimpleNamespace,
    *,
    names: Sequence[str],
    delay: int,
    powers: npt.NDArray[np.int_],
    weights: Array,
    bias: float,
    mean: Array,
    scale: Array,
    spread: float,
) -> Expression:
    """Return the increment of a variable, the ridge readout of the monomials of the rescaled lagged variables."""
    linear = [
        (lag(getattr(s.target, name), steps) - mean[index]) / scale[index]
        for steps in range(delay)
        for index, name in enumerate(names)
    ]
    monomials = [np.prod([linear[int(column)] for column in row]) for row in powers]
    features = [*linear, *monomials]
    readout = sum(float(weight) * feature for weight, feature in zip(weights, features, strict=True) if weight)
    return spread * (readout + bias)


def _fit_reservoir(
    series: Array | None,
    variables: Sequence[str],
    *,
    delay: int,
    order: int,
    ridge: float,
    standardize: bool,
    samples: int,
    step: float,
) -> ReservoirAnalysis:
    """Return the next-generation reservoir analysis fitted to a series, or to samples of the Lorenz flow."""
    values = _lorenz_series(samples, step) if series is None else series
    analysis = ReservoirAnalysis(
        list(variables),
        kind=REQUIRED_KIND,
        delay=delay,
        order=order,
        ridge=ridge,
        standardize=standardize,
        validation=0.0,
        dt=step,
    )
    return analysis.check().prepare(values).run()


@register_model(
    field='computer_science/neural_networks',
    tags=('neural network', 'reservoir computing', 'next generation', 'lags', 'discrete map'),
    references=('Gauthier et al. 2021', 'Cestnik and Martens 2025'),
    defaults={'until': 200, 'dt': 1},
)
def next_generation_reservoir(
    series: Array | None = None,
    *,
    analysis: ReservoirAnalysis | None = None,
    variables: Sequence[str] = LORENZ_VARIABLES,
    delay: int = 2,
    order: int = 2,
    ridge: float = 2.5e-6,
    standardize: bool = False,
    samples: int = 403,
    step: float = 0.025,
) -> System:
    """Return a next-generation reservoir computer whose variables follow the ridge readout of their monomials.

    Every variable keeps `delay - 1` lags, and one map in the increment mode adds to it the readout of the constant, the
    variables with their lags, and the monomials of degree `order` of those. The readout comes from a fitted
    `ReservoirAnalysis(kind='next_generation')`, and the variables start at the end of its last series, so a run is the
    autonomous forecast from there. One step of the system stands for one sample of the series. The defaults are those
    of Gauthier et al. (2021) for the Lorenz flow: 400 training samples after a warmup of 2, one sample every 0.025 from
    an explicit Runge-Kutta 3(2) integrator, and the ridge 2.5e-6 on features that are not standardized. A ridge this
    small on these features diverges within one Lyapunov time when a series comes from a much more accurate integrator,
    and such a series needs `standardize=True`.

    Args:
        series:
            The training series, one row per sample and one column per variable, or `None` for `samples` samples of the
            Lorenz flow every `step`.

        analysis:
            A fitted next-generation reservoir analysis to take the readout from, which replaces the fit.

        variables:
            The names of the columns of the series.

        delay:
            The number of consecutive samples the features read.

        order:
            The degree of the monomials.

        ridge:
            The strength of the ridge penalty.

        standardize:
            Whether the features read the standardized variables.

        samples:
            The number of samples of the Lorenz flow.

        step:
            The time between the samples of the Lorenz flow.

    Returns:
        system:
            The model, with the variables `reservoir.x` and their lags `reservoir.x_lag1` for the Lorenz flow.

    Raises:
        ValidationError:
            The analysis is not a next-generation reservoir.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.computer_science.neural_networks import next_generation_reservoir
        >>> values = np.sin(0.1 * np.arange(200.0))[:, None]
        >>> system = next_generation_reservoir(values, variables=['v'])
        >>> [interaction.mode for interaction in system.interactions]
        ['increment']
    """
    fitted = analysis or _fit_reservoir(
        series,
        variables,
        delay=delay,
        order=order,
        ridge=ridge,
        standardize=standardize,
        samples=samples,
        step=step,
    )
    if fitted.kind != REQUIRED_KIND:
        error_msg = f'The readout comes from a {REQUIRED_KIND!r} reservoir analysis, not from a {fitted.kind!r} one.'
        raise ValidationError(error_msg)
    names = fitted.states_[: len(fitted.labels_)]
    reservoir = fitted.step_
    memory = fitted.samples_.series[-1][-fitted.delay :][::-1]
    initial = {name: memory[:, index].tolist() for index, name in enumerate(names)}
    element = _reservoir_element(initial)
    weights = np.asarray(reservoir.readout.Wout, dtype=float)
    bias = np.ravel(np.asarray(reservoir.readout.bias, dtype=float))
    maps = [
        MapInteraction(
            name,
            target='reservoir',
            state=name,
            mode='increment',
            function=functools.partial(
                _increment,
                names=names,
                delay=fitted.delay,
                powers=reservoir.monomials(),
                weights=weights[:, index],
                bias=float(bias[index]),
                mean=reservoir.mean,
                scale=reservoir.scale,
                spread=float(reservoir.scale[index]),
            ),
        )
        for index, name in enumerate(names)
    ]
    return System(elements=[element], interactions=maps, time=DiscreteTime())
