"""Next-generation reservoir computer.

It uses lagged variables and the ridge readout of their polynomial features.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ....core.analysis import ReservoirResult, reservoir
from ....core.components.elements import DeclaredElement, Element
from ....core.components.interactions import MapInteraction, RateInteraction
from ....core.components.laws.utils import lag
from ....core.components.spaces import ContinuousTime, DiscreteTime
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.runtime import simulate
from ....core.engine.solvers import ScipyODESolver
from ....core.errors import ValidationError
from ....core.specs import DeclaredVariableSpec, StateSpec
from ....core.types import Array

LORENZ_START = (-8.0, 8.0, 27.0)
LORENZ_TRANSIENT = 5.0
LORENZ_RTOL, LORENZ_ATOL = 1e-3, 1e-6
LORENZ_VARIABLES = ('x', 'y', 'z')
REQUIRED_KIND = 'next_generation'


class LorenzFlowElement(Element):
    """Point of the Lorenz flow that trains a reservoir.

    It holds the three variables of the flow.

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
        >>> from skcomplex.systems.computer_science.neural_networks import LorenzFlowElement
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


class NextGenerationReservoirSystem(BaseSystem):
    """A next-generation reservoir computer.

    Its variables follow the ridge readout of their monomials.

    Every variable keeps `delay - 1` lags, and one map in the increment mode adds to it the readout of the constant, the
    variables with their lags, and the monomials of degree `order` of those. The readout comes from the result of
    `reservoir(kind='next_generation')`, and the variables start at the end of its last series, so a run is the
    autonomous forecast from there. One step of the system stands for one sample of the series. The defaults are those
    of Gauthier et al. (2021) for the Lorenz flow: 400 training samples after a warmup of 2, one sample every 0.025 from
    an explicit Runge-Kutta 3(2) integrator, and the ridge 2.5e-6 on features that are not standardized. A ridge this
    small on these features diverges within one Lyapunov time when a series comes from a much more accurate integrator,
    and such a series needs `standardize=True`. The system has the variables `reservoir.x` and their lags
    `reservoir.x_lag1` for the Lorenz flow.

    Args:
        series:
            The training series, one row per sample and one column per variable, or `None` for `samples` samples of the
            Lorenz flow every `step`.

        result:
            The result of a next-generation `reservoir` to take the readout from, which replaces the fit.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The result is not that of a next-generation reservoir.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.computer_science.neural_networks import NextGenerationReservoirSystem
        >>> values = np.sin(0.1 * np.arange(200.0))[:, None]
        >>> system = NextGenerationReservoirSystem(values, variables=['v']).prepare().system_
        >>> [interaction.mode for interaction in system.interactions]
        ['increment']
    """

    field = 'computer_science/neural_networks'
    topics = ('neural network', 'reservoir computing', 'next generation', 'lags', 'discrete map')
    references = ('Gauthier et al. 2021', 'Cestnik and Martens 2025')
    until = 200
    dt = 1

    def __init__(
        self,
        series: Array | None = None,
        *,
        result: ReservoirResult | None = None,
        variables: Sequence[str] = LORENZ_VARIABLES,
        delay: int = 2,
        order: int = 2,
        ridge: float = 2.5e-6,
        standardize: bool = False,
        samples: int = 403,
        step: float = 0.025,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.series = series
        self.result = result
        self.variables = variables
        self.delay = delay
        self.order = order
        self.ridge = ridge
        self.standardize = standardize
        self.samples = samples
        self.step = step

    @staticmethod
    def _lorenz_series(
        samples: int,
        step: float,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
    ) -> Array:
        """Return samples of the Lorenz flow on its attractor.

        There is one row per sample and one column per variable.
        """
        rates: dict[str, Callable[[SimpleNamespace], Expression]] = {
            'x': lambda s: sigma * (s.target.y - s.target.x),
            'y': lambda s: s.target.x * (rho - s.target.z) - s.target.y,
            'z': lambda s: s.target.x * s.target.y - beta * s.target.z,
        }
        system = CompositeSystem(
            elements=[LorenzFlowElement('flow')],
            interactions=[RateInteraction(name, target='flow', state=name, rate=rate) for name, rate in rates.items()],
            time=ContinuousTime(),
        )
        solver = ScipyODESolver(method='RK23', rtol=LORENZ_RTOL, atol=LORENZ_ATOL)
        trajectory = simulate(system, until=LORENZ_TRANSIENT + step * (samples - 1), solver=solver, dt=step)
        values: Array = np.column_stack(
            [np.asarray(trajectory.values(f'flow.{name}'), dtype=np.float64) for name in LORENZ_VARIABLES],
        )
        return values[-samples:]

    @staticmethod
    def _reservoir_element(initial: Mapping[str, Sequence[float]]) -> DeclaredElement:
        """Return the element of scalar states with lags.

        It is built from the values of every state and of its lags.
        """
        variables = {
            name: DeclaredVariableSpec(
                float(values[0]),
                StateSpec(lags=len(values) - 1, initial=tuple(float(value) for value in values[1:])),
            )
            for name, values in initial.items()
        }
        return DeclaredElement('reservoir', variables=variables)

    @staticmethod
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
        """Return the increment of a variable.

        It is the ridge readout of the monomials of the rescaled lagged variables.
        """
        linear = [
            (lag(getattr(s.target, name), steps) - mean[index]) / scale[index]
            for steps in range(delay)
            for index, name in enumerate(names)
        ]
        monomials = [np.prod([linear[int(column)] for column in row]) for row in powers]
        features = [*linear, *monomials]
        readout = sum(float(weight) * feature for weight, feature in zip(weights, features, strict=True) if weight)
        return spread * (readout + bias)

    @staticmethod
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
    ) -> ReservoirResult:
        """Return the next-generation reservoir fitted to a series.

        Without a series, it is fitted to samples of the Lorenz flow.
        """
        values = NextGenerationReservoirSystem._lorenz_series(samples, step) if series is None else series
        return reservoir(
            values,
            variables=list(variables),
            kind=REQUIRED_KIND,
            delay=delay,
            order=order,
            ridge=ridge,
            standardize=standardize,
            validation=0.0,
            dt=step,
        )

    def _build_system(self) -> CompositeSystem:
        """Return a next-generation reservoir computer.

        Its variables follow the ridge readout of their monomials.
        """
        fitted = self.result or self._fit_reservoir(
            self.series,
            self.variables,
            delay=self.delay,
            order=self.order,
            ridge=self.ridge,
            standardize=self.standardize,
            samples=self.samples,
            step=self.step,
        )
        if fitted.step.kind != REQUIRED_KIND:
            error_msg = f'The readout comes from a {REQUIRED_KIND!r} reservoir, not from a {fitted.step.kind!r} one.'
            raise ValidationError(error_msg)
        names = fitted.states[: len(fitted.labels)]
        readout = fitted.step
        delay = (len(fitted.states) - len(names)) // len(names)
        memory = fitted.samples.series[-1][-delay:][::-1]
        initial = {name: memory[:, index].tolist() for index, name in enumerate(names)}
        element = self._reservoir_element(initial)
        weights = np.asarray(readout.readout.Wout, dtype=float)
        bias = np.ravel(np.asarray(readout.readout.bias, dtype=float))
        maps = [
            MapInteraction(
                name,
                target='reservoir',
                state=name,
                mode='increment',
                function=functools.partial(
                    self._increment,
                    names=names,
                    delay=delay,
                    powers=readout.monomials(),
                    weights=weights[:, index],
                    bias=float(bias[index]),
                    mean=readout.mean,
                    scale=readout.scale,
                    spread=float(readout.scale[index]),
                ),
            )
            for index, name in enumerate(names)
        ]
        return CompositeSystem(self.label, elements=[element], interactions=maps, time=DiscreteTime())
