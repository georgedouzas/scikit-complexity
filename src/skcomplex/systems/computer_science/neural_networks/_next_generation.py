"""Next-generation reservoir computer.

It uses lagged variables and the ridge readout of their polynomial features.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ....core.analysis import ReservoirResult, reservoir
from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw, UpdateLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, ScipyODESolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
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
        x: float = LORENZ_START[0],
        y: float = LORENZ_START[1],
        z: float = LORENZ_START[2],
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The result is not that of a next-generation reservoir.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.computer_science.neural_networks import NextGenerationReservoirSystem
        >>> values = np.sin(0.1 * np.arange(200.0))[:, None]
        >>> NextGenerationReservoirSystem(values, variables=['v']).prepare().states_
        ('reservoir.v', 'reservoir.v[-1]')
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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
    def _get_lorenz_series(
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
        system = System(
            elements=[LorenzFlowElement('flow')],
            interactions=[
                Interaction(
                    'lorenz',
                    roles={'target': 'flow'},
                    laws=lambda s: [RateLaw(getattr(s.target, name), rate(s)) for name, rate in rates.items()],
                ),
            ],
            solver=ScipyODESolver(method='RK23', rtol=LORENZ_RTOL, atol=LORENZ_ATOL),
        )
        trajectory = system.prepare().simulate(until=LORENZ_TRANSIENT + step * (samples - 1), dt=step)
        values: Array = np.column_stack(
            [np.asarray(trajectory.get_values(f'flow.{name}'), dtype=np.float64) for name in LORENZ_VARIABLES],
        )
        return values[-samples:]

    @staticmethod
    def _get_reservoir_element(initial: Mapping[str, Sequence[float]]) -> Element:
        """Return the element of scalar states with their past values.

        It is built from the values of every state and of its past values, named `x[-1]`, `x[-2]`, and so on.
        """
        values = {
            f'{name}[-{steps}]' if steps else name: float(value)
            for name, past in initial.items()
            for steps, value in enumerate(past)
        }
        return Element('reservoir', values=values)

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
            (getattr(s.target, name)(t=s.t - steps) - mean[index]) / scale[index]
            for steps in range(delay)
            for index, name in enumerate(names)
        ]
        monomials = [np.prod([linear[int(column)] for column in row]) for row in powers]
        features = [*linear, *monomials]
        readout = sum(float(weight) * feature for weight, feature in zip(weights, features, strict=True) if weight)
        return spread * (readout + bias)

    @staticmethod
    def _get_step(
        s: SimpleNamespace,
        *,
        name: str,
        increment: Callable[[SimpleNamespace], Expression],
    ) -> list[BaseLaw]:
        """Return the update that adds the increment of a variable to it."""
        value = getattr(s.target, name)
        return [UpdateLaw(value, value + increment(s), increment=True)]

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
        values = NextGenerationReservoirSystem._get_lorenz_series(samples, step) if series is None else series
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
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
        element = self._get_reservoir_element(initial)
        weights = np.asarray(readout.readout.Wout, dtype=float)
        bias = np.ravel(np.asarray(readout.readout.bias, dtype=float))
        increments = {
            name: functools.partial(
                self._increment,
                names=names,
                delay=delay,
                powers=readout.get_monomials(),
                weights=weights[:, index],
                bias=float(bias[index]),
                mean=readout.mean,
                scale=readout.scale,
                spread=float(readout.scale[index]),
            )
            for index, name in enumerate(names)
        }
        maps = [
            Interaction(
                name,
                roles={'target': 'reservoir'},
                laws=functools.partial(self._get_step, name=name, increment=increments[name]),
            )
            for name in names
        ]
        self.system_ = System(self.label, elements=[element], interactions=maps)
