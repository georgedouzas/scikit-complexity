"""Model predictive control of nonlinear plants through linear surrogates.

The surrogates are fitted by mode decomposition with control.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, sin
from sage.sets.real_set import RealSet
from scipy.optimize import lsq_linear
from sklearn.base import TransformerMixin
from sklearn.preprocessing import PolynomialFeatures

from ....core.analysis import ModeDecompositionResult, mode_decomposition
from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, CallSpec, RateLaw, UpdateLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.runtime import Trajectory
from ....core.engine.solvers import BaseSolver
from ....core.errors import ValidationError
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role

DUFFING_A, DUFFING_B, DUFFING_D = 1.0, -1.0, -0.3
DUFFING_FORCE, DUFFING_FREQUENCY = 0.5, 1.2
VAN_DER_POL_MU = 2.0
MOTOR_LA, MOTOR_RA, MOTOR_KM, MOTOR_J, MOTOR_B, MOTOR_TAU, MOTOR_UA = 0.314, 12.345, 0.253, 0.00441, 0.00732, 1.47, 60.0


def _duffing_rates(plant: SimpleNamespace, control: Expression, t: Expression, step: float) -> tuple[Expression, ...]:
    """Return the rates per sample of the forced Duffing oscillator.

    The oscillator is that of the review of Brunton et al., figure 6.6. The second derivative is
    `x'' = a x + b x^3 + d x' + f cos(w t) + (3 + cos(x)) u`.
    """
    force = DUFFING_FORCE * cos(DUFFING_FREQUENCY * step * t)
    gain = 3 + cos(plant.x)
    acceleration = DUFFING_A * plant.x + DUFFING_B * plant.x**3 + DUFFING_D * plant.y + force + gain * control
    return step * plant.y, step * acceleration


def _van_der_pol_rates(
    plant: SimpleNamespace,
    control: Expression,
    t: Expression,
    step: float,
) -> tuple[Expression, ...]:
    """Return the rates per sample of the Van der Pol oscillator.

    The oscillator is `x'' = mu (1 - x^2) x' - x + u`.
    """
    acceleration = VAN_DER_POL_MU * (1 - plant.x**2) * plant.y - plant.x + control
    return step * plant.y, step * acceleration


def _motor_rates(plant: SimpleNamespace, control: Expression, t: Expression, step: float) -> tuple[Expression, ...]:
    """Return the rates per sample of a bilinear DC motor.

    They are the rates of the current and the angular velocity.
    """
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

    rates: Callable[..., tuple[Expression, ...]]
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
    """Control input that a plant reads, scaled to the interval from -1 to 1.

    Args:
        label:
            The label of the input.

        u:
            The value of the input.

    Examples:
        >>> from skcomplex.systems.engineering.control import ControlInputElement
        >>> ControlInputElement('input', u=0.5).u
        0.5
    """

    def __init__(self, label: str, *, u: Annotated[float, RealSet.closed(-1, 1)] = 0.0) -> None:
        super().__init__(label)
        self.u = u


class PlantStateElement(Element):
    """State of a plant with two variables, whose rates `PlantInteraction` gives.

    Args:
        label:
            The label of the plant.

        x:
            The value of the first variable.

        y:
            The value of the second variable, the output that tracks the reference.

    Examples:
        >>> from skcomplex.systems.engineering.control import PlantStateElement
        >>> PlantStateElement('plant', x=0.5).x
        0.5
    """

    def __init__(self, label: str, *, x: float = 0.0, y: float = 0.0) -> None:
        super().__init__(label)
        self.x = x
        self.y = y


class PlantInteraction(Interaction):
    """Rates of the two variables of a plant that reads a bounded control input.

    The plant is the forced Duffing oscillator, the Van der Pol oscillator, or the bilinear DC motor of figure 6.6 of
    the review of Brunton et al., with one time unit per sample of the controller. The plant reads `limit * u`, so
    the input `u` of the control element stays between -1 and 1.

    Args:
        label:
            The label of the interaction.

        plant:
            The path of the plant, with the variables `x` and `y`.

        control:
            The path of the control input, with the value `u`.

        kind:
            `'duffing'`, `'van_der_pol'`, or `'motor'`.

    Examples:
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.engineering.control import ControlInputElement, PlantInteraction, PlantStateElement
        >>> elements = [PlantStateElement('plant'), ControlInputElement('input')]
        >>> rates = PlantInteraction('rates', plant='plant', control='input')
        >>> System(elements=elements, interactions=[rates]).prepare().states_
        ('plant.x', 'plant.y')
    """

    def __init__(self, label: str, *, plant: Role, control: Role, kind: str = 'duffing') -> None:
        super().__init__(label)
        self.plant = plant
        self.control = control
        self.kind = kind

    def _check_arguments(self) -> None:
        """Raise an error for an argument the interaction cannot take.

        Raises:
            ValidationError:
                The kind of plant is unknown.
        """
        super()._check_arguments()
        if self.kind not in PLANTS:
            error_msg = f'The plant {self.kind!r} is unknown. Use one of {", ".join(PLANTS)}.'
            raise ValidationError(error_msg)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rates of the two variables of the plant.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        case, plant, t = PLANTS[self.kind], context.plant, context.t
        rates = case.rates(plant, case.limit * context.control.u, t, case.step)
        self.laws_: list[BaseLaw] = [RateLaw(plant.x.diff(t), rates[0]), RateLaw(plant.y.diff(t), rates[1])]


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


class PredictiveControlInteraction(Interaction):
    """Controller that sets a control input once per sample from the two variables of a plant.

    The next input is `policy(t, x, y)`, such as the first input of a tracking problem solved over a horizon, or 0
    without a policy. The input is the target of an update, so a system that holds the controller steps it once per
    unit of time while a continuous plant beside it holds the input between samples.

    Args:
        label:
            The label of the interaction.

        control:
            The path of the control input, with the value `u`.

        plant:
            The path of the plant, with the variables `x` and `y`.

        policy:
            The function `f(t, x, y)` of the next input, or `None` to hold the input at 0.

    Examples:
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.engineering.control import ControlInputElement, PredictiveControlInteraction
        >>> from skcomplex.systems.engineering.control import PlantStateElement
        >>> elements = [ControlInputElement('input'), PlantStateElement('plant')]
        >>> policy = PredictiveControlInteraction('policy', control='input', plant='plant')
        >>> System(elements=elements, interactions=[policy]).prepare().form_
        'discrete map'
    """

    def __init__(
        self,
        label: str,
        *,
        control: Role,
        plant: Role,
        policy: Callable[[float, float, float], float] | None = None,
    ) -> None:
        super().__init__(label)
        self.control = control
        self.plant = plant
        self.policy = policy

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the update of the control input.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        plant = context.plant
        value = 0.0 if self.policy is None else CallSpec(self.policy, (plant.x, plant.y))
        self.laws_: list[BaseLaw] = [UpdateLaw(context.control.u, value)]


class KoopmanModelPredictiveControlSystem(BaseSystem):
    """A plant that a discrete-time controller steers with model predictive control.

    The control runs on a fitted surrogate. The system runs the plant under a random binary input at the limits of its
    range, drawn every sample, and fits `mode_decomposition` with the input to the runs. The controller lifts the states
    with the dictionary, condenses the quadratic tracking cost over the horizon with the matrices of the surrogate,
    solves the bounded least squares problem with `scipy.optimize.lsq_linear`, and sets the input to the first value of
    the solution. The controller updates the input, so the nested system `controller` steps once per unit of time,
    and the rates of the nested system `plant` run in continuous time between the steps, holding the input. The
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ValidationError:
            The plant is unknown.

    Examples:
        >>> from skcomplex.systems.engineering.control import KoopmanModelPredictiveControlSystem
        >>> system = KoopmanModelPredictiveControlSystem('motor', controlled=False).prepare()
        >>> [subsystem.label for subsystem in system.system_.systems], system.form_
        (['controller', 'plant'], 'co-simulation')
    """

    field = 'engineering/control'
    topics = ('control', 'model predictive control', 'koopman', 'co-simulation', 'surrogate')
    references = ('Korda and Mezić 2018', 'Brunton et al. 2022')
    until = 100.0
    dt = 1.0

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.plant = plant
        self.controlled = controlled
        self.dictionary = dictionary
        self.x = x
        self.y = y
        self.n_runs = n_runs
        self.duration = duration
        self.random_state = random_state

    def _run_training(self, case: _PlantCase, rng: np.random.Generator) -> Trajectory:
        """Return a run of the plant under a random binary input, drawn every sample."""
        system = System(
            elements=[
                PlantStateElement('plant', x=rng.uniform(case.low, case.high), y=rng.uniform(case.low, case.high)),
                ControlInputElement('input'),
            ],
            interactions=[PlantInteraction('rates', plant='plant', control='input', kind=self.plant)],
        )
        values = rng.choice((-1.0, 1.0), self.duration + 1)
        held = functools.partial(self._get_held_value, values=values)
        return system.prepare().simulate(until=float(self.duration), dt=1.0, schedule={'input.u': held})

    @staticmethod
    def _get_held_value(t: float, *, values: Array) -> float:
        """Return the input of the sample at a time in samples."""
        return float(values[min(round(t), len(values) - 1)])

    @staticmethod
    def _get_reference(s: SimpleNamespace, *, step: float, symbolic: Callable[[Expression], Expression]) -> Expression:
        """Return the reference at the time of the system, which counts samples."""
        return symbolic(step * s.t)

    def _fit_problem(self, case: _PlantCase) -> _PredictiveProblem:
        """Return the tracking problem of the plant on the surrogate fitted to its training runs."""
        rng = np.random.default_rng(self.random_state)
        surrogate = mode_decomposition(
            [self._run_training(case, rng) for _ in range(self.n_runs)],
            variables=['plant.x', 'plant.y'],
            inputs=['input.u'],
            dictionary=PolynomialFeatures(3, include_bias=False) if self.dictionary is None else self.dictionary,
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

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the controller and the plant built from the model parameters.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ValidationError:
                The plant is unknown.
        """
        super()._prepare(context)
        if self.plant not in PLANTS:
            error_msg = f'The plant {self.plant!r} is unknown. Use one of {", ".join(PLANTS)}.'
            raise ValidationError(error_msg)
        case = PLANTS[self.plant]
        policy = self._fit_problem(case).get_control if self.controlled else None
        reference = functools.partial(self._get_reference, step=case.step, symbolic=case.symbolic_reference)
        self.observables_ = (*self.observables_, Observable('reference', reference))
        self.system_ = System(
            self.label,
            systems=[
                System('controller', elements=[ControlInputElement('input')]),
                System('plant', elements=[PlantStateElement('state', x=self.x, y=self.y)]),
            ],
            interactions=[
                PredictiveControlInteraction(
                    'policy',
                    control='controller/input',
                    plant='plant/state',
                    policy=policy,
                ),
                PlantInteraction('rates', plant='plant/state', control='controller/input', kind=self.plant),
            ],
        )
