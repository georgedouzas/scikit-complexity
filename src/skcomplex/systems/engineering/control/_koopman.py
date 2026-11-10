"""Model predictive control of nonlinear plants through linear surrogates.

The surrogates are fitted by mode decomposition with control.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, sin
from scipy.optimize import lsq_linear
from sklearn.base import TransformerMixin
from sklearn.preprocessing import PolynomialFeatures

from ....core.analysis import ModeDecompositionResult, mode_decomposition
from ....core.components.elements import Element
from ....core.components.interactions import MapInteraction, RateInteraction
from ....core.components.spaces import ContinuousTime, DiscreteTime
from ....core.components.systems import BaseSystem, CompositeSystem, RunSettings
from ....core.engine.runtime import Trajectory, simulate
from ....core.errors import ValidationError
from ....core.specs import CallSpec, ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array

DUFFING_A, DUFFING_B, DUFFING_D = 1.0, -1.0, -0.3
DUFFING_FORCE, DUFFING_FREQUENCY = 0.5, 1.2
VAN_DER_POL_MU = 2.0
MOTOR_LA, MOTOR_RA, MOTOR_KM, MOTOR_J, MOTOR_B, MOTOR_TAU, MOTOR_UA = 0.314, 12.345, 0.253, 0.00441, 0.00732, 1.47, 60.0


def _duffing_rates(s: SimpleNamespace, *, limit: float, step: float) -> tuple[Expression, Expression]:
    """Return the rates per sample of the forced Duffing oscillator.

    The oscillator is that of the review of Brunton et al., figure 6.6. The second derivative is
    `x'' = a x + b x^3 + d x' + f cos(w t) + (3 + cos(x)) u`.
    """
    plant, control = s.target, limit * s.sources[0].u
    force = DUFFING_FORCE * cos(DUFFING_FREQUENCY * step * s.t)
    gain = 3 + cos(plant.x)
    acceleration = DUFFING_A * plant.x + DUFFING_B * plant.x**3 + DUFFING_D * plant.y + force + gain * control
    return step * plant.y, step * acceleration


def _van_der_pol_rates(s: SimpleNamespace, *, limit: float, step: float) -> tuple[Expression, Expression]:
    """Return the rates per sample of the Van der Pol oscillator.

    The oscillator is `x'' = mu (1 - x^2) x' - x + u`.
    """
    plant, control = s.target, limit * s.sources[0].u
    acceleration = VAN_DER_POL_MU * (1 - plant.x**2) * plant.y - plant.x + control
    return step * plant.y, step * acceleration


def _motor_rates(s: SimpleNamespace, *, limit: float, step: float) -> tuple[Expression, Expression]:
    """Return the rates per sample of a bilinear DC motor.

    They are the rates of the current and the angular velocity.
    """
    plant, control = s.target, limit * s.sources[0].u
    current = -MOTOR_RA / MOTOR_LA * plant.x - MOTOR_KM / MOTOR_LA * plant.y * control + MOTOR_UA / MOTOR_LA
    velocity = -MOTOR_B / MOTOR_J * plant.y + MOTOR_KM / MOTOR_J * plant.x * control - MOTOR_TAU / MOTOR_J
    return step * current, step * velocity


def _sine_reference(t: float) -> float:
    """Return the reference `-sin(t)` of the two oscillators."""
    return -math.sin(t)


def _cosine_reference(t: float) -> float:
    """Return the reference `cos(2 pi t / 3) / 2` of the motor."""
    return 0.5 * math.cos(2.0 * math.pi * t / 3.0)


def _symbolic_sine_reference(time: Expression) -> Expression:
    """Return the symbolic reference `-sin(t)` of the two oscillators."""
    return -sin(time)


def _symbolic_cosine_reference(time: Expression) -> Expression:
    """Return the symbolic reference `cos(2 pi t / 3) / 2` of the motor."""
    return cos(2 * math.pi * time / 3) / 2


@dataclass(frozen=True)
class _PlantCase:
    """Plant of the system with its rates and its control settings.

    The settings are the sampling, input limit, start domain, horizon, weight, and reference.
    """

    rates: Callable[..., tuple[Expression, Expression]]
    step: float
    limit: float
    low: float
    high: float
    horizon: int
    weight: float
    reference: Callable[[float], float]
    symbolic_reference: Callable[[Expression], Expression]


PLANTS = {
    'duffing': _PlantCase(_duffing_rates, 0.1, 0.5, -1.5, -1.0, 10, 0.1, _sine_reference, _symbolic_sine_reference),
    'van_der_pol': _PlantCase(
        _van_der_pol_rates,
        0.1,
        5.0,
        -2.0,
        2.0,
        30,
        0.01,
        _sine_reference,
        _symbolic_sine_reference,
    ),
    'motor': _PlantCase(
        _motor_rates,
        0.01,
        4.0,
        -1.0,
        1.0,
        10,
        0.1 / 16.0,
        _cosine_reference,
        _symbolic_cosine_reference,
    ),
}


class ControlInputElement(Element):
    """Control input that a plant reads, a bounded adaptive parameter.

    It is scaled to the interval from -1 to 1.

    Args:
        label:
            The label of the input.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The value of the input.

    Examples:
        >>> from skcomplex.systems.engineering.control import ControlInputElement
        >>> ControlInputElement('input', u=0.5).u
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, ParameterSpec(adaptive=True, bounds=(-1.0, 1.0))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u


class PlantStateElement(Element):
    """State of a plant with two variables, whose rates a pair of interactions gives.

    Args:
        label:
            The label of the plant.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The value of the first variable.

        y:
            The value of the second variable, the output that tracks the reference.

    Examples:
        >>> from skcomplex.systems.engineering.control import PlantStateElement
        >>> PlantStateElement('plant', x=0.5).x
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y


class _PredictiveProblem:
    """Quadratic tracking problem over a horizon on a fitted linear surrogate.

    It is condensed to a bounded least squares.
    """

    def __init__(
        self,
        surrogate: ModeDecompositionResult,
        *,
        output: int,
        horizon: int,
        weight: float,
        step: float,
        reference: Callable[[float], float],
    ) -> None:
        self.surrogate = surrogate
        self.output = output
        self.horizon = horizon
        self.weight = weight
        self.step = step
        self.reference = reference

    def prepare(self) -> "_PredictiveProblem":
        """Set the rows that give the outputs over the horizon.

        The outputs follow from the first state and from the inputs.

        Returns:
            problem:
                The problem.
        """
        operator, input_matrix = self.surrogate.operator, self.surrogate.input_matrix
        readout = np.eye(len(operator))[self.output]
        free = np.zeros((self.horizon, len(operator)))
        forced = np.zeros((self.horizon, self.horizon))
        for row in range(self.horizon):
            free[row] = readout
            forced[row, 0] = readout @ input_matrix[:, 0]
            forced[row, 1 : row + 1] = forced[row - 1, :row]
            readout = readout @ operator
        self.free_ = free @ operator
        self.forced_ = forced
        return self

    def get_control(self, t: float, *states: float) -> float:
        """Return the first input of the solution of the tracking problem.

        The problem is read from states at a time in samples.

        Args:
            t:
                The time in samples.

            *states:
                The values of the variables of the plant.

        Returns:
            control:
                The first input, between -1 and 1.
        """
        lifted = np.asarray(states, dtype=float)[None, :]
        dictionary = self.surrogate.dictionary
        current = lifted[0] if dictionary is None else np.asarray(dictionary.transform(lifted), dtype=float)[0]
        goals = np.asarray([self.reference((t + index + 1) * self.step) for index in range(self.horizon)])
        problem = np.vstack([self.forced_, math.sqrt(self.weight) * np.eye(self.horizon)])
        target = np.concatenate([goals - self.free_ @ current, np.zeros(self.horizon)])
        return float(lsq_linear(problem, target, bounds=(-1.0, 1.0)).x[0])


class KoopmanModelPredictiveControlSystem(BaseSystem):
    """A plant that a discrete-time controller steers with model predictive control.

    The control runs on a fitted surrogate. The system runs the plant under a random binary input at the limits of its
    range, drawn every sample, and fits `mode_decomposition` with the input to the runs. The controller lifts the states
    with the dictionary, condenses the quadratic tracking cost over the horizon with the matrices of the surrogate,
    solves the bounded least squares problem with `scipy.optimize.lsq_linear`, and sets the input to the first value of
    the solution. The plant reads the input through a role, and `CoSimulationSolver` holds it between samples. The
    plants are those of figure 6.6 of the review of Brunton et al.: the forced Duffing oscillator
    `x'' = a x + b x^3 + d x' + f cos(w t) + (3 + cos(x)) u`, the Van der Pol oscillator
    `x'' = -x - mu (x^2 - 1) x' + u`, and the bilinear DC motor, whose second variable is the output that tracks the
    reference. The motor reads four times the input, as Korda and Mezić scale it. One time unit of the system is one
    sample of the controller.

    The system has the variables `plant/state.x` and `plant/state.y`, the input
    `controller/input.u`, and the observable `reference`.

    Args:
        plant:
            `'duffing'`, `'van_der_pol'`, or `'motor'`.

        controlled:
            Whether the controller solves the tracking problem, or holds the input at 0 with no fit.

        dictionary:
            A scikit-learn transformer that computes the observables and keeps the variables among them, or `None` for
            the monomials up to degree 3.

        x:
            The initial value of the first variable.

        y:
            The initial value of the second variable.

        n_runs:
            The number of training runs.

        duration:
            The number of samples of every training run.

        random_state:
            The seed or generator of the training runs.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The plant is unknown.

    Examples:
        >>> from skcomplex.systems.engineering.control import KoopmanModelPredictiveControlSystem
        >>> system = KoopmanModelPredictiveControlSystem('motor', controlled=False).prepare().system_
        >>> [subsystem.label for subsystem in system.systems]
        ['controller', 'plant']
    """

    field = 'engineering/control'
    topics = ('control', 'model predictive control', 'koopman', 'co-simulation', 'surrogate')
    references = ('Korda and Mezić 2018', 'Brunton et al. 2022')
    until = 100.0
    dt = 1.0

    @staticmethod
    def _get_plant_interactions(case: _PlantCase, target: str, source: str) -> list[RateInteraction]:
        """Return the rates of the two variables of a plant.

        The plant reads its input from a source.
        """
        rates = functools.partial(case.rates, limit=case.limit, step=case.step)
        return [
            RateInteraction(
                name,
                target=target,
                state=name,
                sources=[source],
                rate=functools.partial(KoopmanModelPredictiveControlSystem._get_rate_at, rates=rates, index=index),
            )
            for index, name in enumerate(('x', 'y'))
        ]

    @staticmethod
    def _get_rate_at(
        s: SimpleNamespace,
        *,
        rates: Callable[[SimpleNamespace], tuple[Expression, Expression]],
        index: int,
    ) -> Expression:
        """Return one of the two rates of a plant."""
        return rates(s)[index]

    @staticmethod
    def _run_training(case: _PlantCase, duration: int, rng: np.random.Generator) -> Trajectory:
        """Return a run of the plant under a random binary input.

        The input is drawn every sample, from a random state.
        """
        system = CompositeSystem(
            elements=[
                PlantStateElement('plant', x=rng.uniform(case.low, case.high), y=rng.uniform(case.low, case.high)),
                ControlInputElement('input'),
            ],
            interactions=KoopmanModelPredictiveControlSystem._get_plant_interactions(case, 'plant', 'input'),
            time=ContinuousTime(),
        )
        values = rng.choice((-1.0, 1.0), duration + 1)
        return simulate(
            system,
            until=float(duration),
            dt=1.0,
            schedule={'input.u': functools.partial(KoopmanModelPredictiveControlSystem._get_held_value, values=values)},
        )

    @staticmethod
    def _get_held_value(t: float, *, values: Array) -> float:
        """Return the input of the sample at a time in samples."""
        return float(values[min(round(t), len(values) - 1)])

    @staticmethod
    def _get_reference(s: SimpleNamespace, *, step: float, symbolic: Callable[[Expression], Expression]) -> Expression:
        """Return the reference at the time of the system, which counts samples."""
        return symbolic(step * s.t)

    def __init__(
        self,
        plant: str = 'duffing',
        *,
        controlled: bool = True,
        dictionary: TransformerMixin | None = None,
        x: float = 0.5,
        y: float = 0.0,
        n_runs: int = 100,
        duration: int = 1000,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.plant = plant
        self.controlled = controlled
        self.dictionary = dictionary
        self.x = x
        self.y = y
        self.n_runs = n_runs
        self.duration = duration
        self.random_state = random_state

    @staticmethod
    def _hold(s: SimpleNamespace, *, problem: _PredictiveProblem | None) -> object:
        """Return the next input of the controller.

        It is the solution of the tracking problem, or 0 without one.
        """
        if problem is None:
            return 0.0
        plant = s.sources[0]
        return CallSpec(problem.get_control, (plant.x, plant.y))

    @staticmethod
    def _fit_problem(
        case: _PlantCase,
        dictionary: TransformerMixin | None,
        n_runs: int,
        duration: int,
        random_state: int | np.random.Generator | None,
    ) -> _PredictiveProblem:
        """Return the tracking problem of a plant on its fitted surrogate.

        The surrogate is fitted to its training runs.
        """
        rng = np.random.default_rng(random_state)
        surrogate = mode_decomposition(
            [KoopmanModelPredictiveControlSystem._run_training(case, duration, rng) for _ in range(n_runs)],
            variables=['plant.x', 'plant.y'],
            inputs=['input.u'],
            dictionary=PolynomialFeatures(3, include_bias=False) if dictionary is None else dictionary,
            validation=0.0,
        )
        output = surrogate.labels.index('plant.y')
        return _PredictiveProblem(
            surrogate,
            output=output,
            horizon=case.horizon,
            weight=case.weight * case.limit**2,
            step=case.step,
            reference=case.reference,
        ).prepare()

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        if self.plant not in PLANTS:
            error_msg = f'The plant {self.plant!r} is unknown. Use one of {", ".join(PLANTS)}.'
            raise ValidationError(error_msg)
        case = PLANTS[self.plant]
        problem = (
            self._fit_problem(case, self.dictionary, self.n_runs, self.duration, self.random_state)
            if self.controlled
            else None
        )
        controller = MapInteraction(
            'policy',
            target='controller/input',
            state='u',
            sources=['plant/state'],
            function=functools.partial(self._hold, problem=problem),
        )
        return CompositeSystem(
            self.label,
            systems=[
                CompositeSystem('controller', elements=[ControlInputElement('input')], time=DiscreteTime()),
                CompositeSystem(
                    'plant',
                    elements=[PlantStateElement('state', x=self.x, y=self.y)],
                    time=ContinuousTime(),
                ),
            ],
            interactions=[
                controller,
                *KoopmanModelPredictiveControlSystem._get_plant_interactions(case, 'plant/state', 'controller/input'),
            ],
        )

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings, with the reference of the plant as an observable."""
        case = PLANTS[self.plant]
        reference = functools.partial(
            KoopmanModelPredictiveControlSystem._get_reference,
            step=case.step,
            symbolic=case.symbolic_reference,
        )
        settings = super().get_run_settings()
        settings['observables'] = [ObservableSpec('reference', reference)]
        return settings
