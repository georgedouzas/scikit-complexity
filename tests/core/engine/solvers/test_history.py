"""Test the `History` class."""

import math
import warnings
from typing import Annotated

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import CrossingLaw, NoiseLaw, RateLaw
from skcomplex.core.components.laws.utils import delayed, pulse_train, series, step
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import BaseObserver, simulate
from skcomplex.core.engine.solvers import (
    EulerMaruyamaSolver,
    GillespieSolver,
    HeunSolver,
    MilsteinSolver,
    RK4Solver,
    ScipyODESolver,
    VelocityVerletSolver,
)
from skcomplex.core.errors import SolverError, ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec, StepSpec, VisualSpec
from skcomplex.systems.biology.ecology import HutchinsonSystem
from skcomplex.systems.biology.physiology import MackeyGlassSystem
from skcomplex.systems.engineering.processes import BathtubSystem
from skcomplex.systems.physics.mechanics import BouncingBallSystem, HuygensClocksSystem


class ClockElement(Element):
    """Clock element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the laws of the clock element."""
        return [RateLaw(s.x, 1), CrossingLaw(s.x - 3.2, terminal=True)]


class FirstOrderElement(Element):
    """First order element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[object, StateSpec()] = 0.0,
        k: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y
        self.k = k

    def _get_laws(self, s):
        """Return the laws of the first order element."""
        return [RateLaw(s.y, -s.k * s.y + step(s.t, 5.0))]


class GeometricElement(Element):
    """Geometric Brownian motion whose noise follows a convention."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        convention: str = 'ito',
        x: Annotated[object, StateSpec()] = 1.0,
        mu: Annotated[object, ParameterSpec()] = 0.5,
        sigma: Annotated[object, ParameterSpec()] = 0.8,
    ) -> None:
        super().__init__(label, visual=visual)
        self.convention = convention
        self.x = x
        self.mu = mu
        self.sigma = sigma

    def _get_laws(self, s):
        """Return the drift and the noise of the motion."""
        return [RateLaw(s.x, s.mu * s.x), NoiseLaw(s.x, s.sigma * s.x, convention=self.convention)]


class DelayedElement(Element):
    """Delayed element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the laws of the delayed element."""
        return [RateLaw(s.x, -delayed(s.x, 1.0))]


def _impact_times(n_impacts, h=1.0, g=9.81, e=0.8):
    """Return the analytic impact times of a ball dropped from rest."""
    times, speed = [math.sqrt(2 * h / g)], math.sqrt(2 * g * h)
    while len(times) < n_impacts:
        speed *= e
        times.append(times[-1] + 2 * speed / g)
    return np.asarray(times)


def _divergence_rate(tau, until=1500.0, window=100.0):
    """Return the growth rate of the distance between two Mackey-Glass runs that start 1e-8 apart.

    The rate is the slope of the logarithm of the largest distance in successive windows, which estimates the largest
    Lyapunov exponent while the distance stays small.
    """
    solver = ScipyODESolver(rtol=1e-9, atol=1e-12)
    first = simulate(MackeyGlassSystem(tau=tau).prepare().system_, until=until, dt=0.5, solver=solver)
    second = simulate(
        MackeyGlassSystem(x=0.5 + 1e-8, tau=tau).prepare().system_,
        until=until,
        dt=0.5,
        solver=clone(solver),
    )
    distance = np.abs(first.get_values('cells.x') - second.get_values('cells.x'))
    starts = np.arange(0.0, until, window)
    peaks = [distance[(first.times >= start) & (first.times < start + window)].max() for start in starts]
    return np.polyfit(starts, np.log(peaks), 1)[0]


def test_mackey_glass_is_chaotic_at_seventeen():
    """Test that Mackey-Glass at tau = 17 separates nearby runs exponentially, a positive largest exponent."""
    assert _divergence_rate(17.0) > 0.003


def test_mackey_glass_is_periodic_at_six():
    """Test that Mackey-Glass at tau = 6 settles on a periodic orbit whose maxima repeat."""
    values = simulate(MackeyGlassSystem(tau=6.0).prepare().system_, until=600, dt=0.1).get_values('cells.x')[3000:]
    peaks = values[1:-1][(values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])]
    assert peaks.size > 10
    assert np.ptp(peaks) < 1e-3
    assert abs(_divergence_rate(6.0)) < 0.002


def test_hutchinson_oscillates_above_threshold():
    """Test that the delayed logistic oscillates around K with a stable amplitude when r tau is above pi/2."""
    values = simulate(HutchinsonSystem(r=1.0, tau=2.0).prepare().system_, until=200, dt=0.1).get_values('population.x')
    middle, late = values[1000:1500], values[1500:]
    assert np.ptp(late) > 50
    assert np.ptp(late) == pytest.approx(np.ptp(middle), rel=0.01)
    assert late.mean() == pytest.approx(100, rel=0.5)


def test_hutchinson_settles_below_threshold():
    """Test that the delayed logistic settles at K when r tau is below pi/2."""
    values = simulate(HutchinsonSystem(r=1.0, tau=1.0).prepare().system_, until=200, dt=0.1).get_values('population.x')
    assert values[-1] == pytest.approx(100, abs=1e-3)


@pytest.mark.parametrize('solver', [ScipyODESolver, RK4Solver])
def test_delayed_decay_matches_method_of_steps(solver):
    """Test that `x' = -x(t - 1)` with history 1 matches its exact solution `1 - t + (t - 1)^2 / 2` on [1, 2]."""
    trajectory = simulate(CompositeSystem(elements=[DelayedElement('d')]), until=2, dt=0.1, solver=solver())
    times = trajectory.times
    exact = np.where(times <= 1, 1 - times, 1 - times + (times - 1) ** 2 / 2)
    assert trajectory.get_values('d.x') == pytest.approx(exact, abs=1e-6)


def _with_history(s):
    """Return a decay driven by the state one unit of time ago."""
    return [RateLaw(s.x, -delayed(s.x, 1.0))]


def test_declared_history_is_read_before_the_start():
    """Test that a history the state declares as a function of time feeds the delayed state before the start."""

    class RememberedElement(Element):
        """Remembered element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec(history=lambda t: 1 + t)] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the remembered element."""
            return _with_history(s)

    trajectory = simulate(CompositeSystem(elements=[RememberedElement('d')]), until=1, dt=0.1)
    assert trajectory.get_values('d.x')[-1] == pytest.approx(1 - 0.5, abs=1e-5)


def test_history_is_keyed_by_state():
    """Test that the history a state declares is compiled under the name of its state."""

    class RememberedElement(Element):
        """Remembered element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec(history=2.0)] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the remembered element."""
            return _with_history(s)

    assert PassCompiler().prepare(CompositeSystem(elements=[RememberedElement('d')])).run().problem_.history == {
        'd.x': 2.0,
    }


@pytest.mark.parametrize('solver', [ScipyODESolver, RK4Solver])
def test_bouncing_ball_impact_times(solver):
    """Test that the impact times of a ball with restitution 0.8 match the analytic times within 1e-6."""
    trajectory = simulate(BouncingBallSystem().prepare().system_, until=3.3, dt=0.01, solver=solver())
    times = np.asarray([event.time for event in trajectory.events])
    assert times.size == 7
    assert times == pytest.approx(_impact_times(times.size), abs=1e-6)
    first = trajectory.events[0]
    assert first.label == 'impact'
    assert first.after[1] == pytest.approx(-0.8 * first.before[1])


def test_terminal_event_ends_the_run():
    """Test that a terminal event at 3.2 ends the run there with the state at the crossing."""
    trajectory = simulate(CompositeSystem(elements=[ClockElement('clock')]), until=10, dt=0.5)
    assert trajectory.times[-1] == pytest.approx(3.2)
    assert trajectory.get_values('clock.x')[-1] == pytest.approx(3.2)
    assert [event.label for event in trajectory.events] == ['clock']


def test_event_without_reset_fires_once():
    """Test that an event that keeps the state fires once at its crossing and the run goes on past it."""

    class MarkerElement(Element):
        """Marker element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the marker element."""
            return [RateLaw(s.x, -1), CrossingLaw(s.x, direction=-1)]

    trajectory = simulate(CompositeSystem(elements=[MarkerElement('m')]), until=2, dt=0.5)
    assert [event.time for event in trajectory.events] == pytest.approx([1.0])
    assert trajectory.get_values('m.x')[-1] == pytest.approx(-1.0)


@pytest.mark.parametrize('solver', [ScipyODESolver, RK4Solver])
def test_chattering_event_raises(solver):
    """Test that an event whose reset puts the state back just before the guard, so it fires again, raises an error."""

    class StickyElement(Element):
        """Sticky element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the sticky element."""
            return [RateLaw(s.x, -1), CrossingLaw(s.x, {s.x: 1e-12}, direction=-1)]

    with pytest.raises(SolverError, match='chattering'):
        simulate(CompositeSystem(elements=[StickyElement('c')]), until=2, dt=0.5, solver=solver())


@pytest.mark.parametrize('solver', [ScipyODESolver, RK4Solver])
def test_step_response(solver):
    """Test that a first-order system is unforced before a step at 5 and follows the step response after it."""
    trajectory = simulate(CompositeSystem(elements=[FirstOrderElement('f')]), until=10, dt=0.25, solver=solver())
    times, values = trajectory.times, trajectory.get_values('f.y')
    assert np.all(values[times < 5] == 0.0)
    assert values == pytest.approx(np.where(times >= 5, 1 - np.exp(-(times - 5)), 0.0), abs=1e-6)


def test_pulse_train_and_series_breakpoints():
    """Test that periodic pulses and interpolated series give the breakpoints of their discontinuities."""

    class DrivenElement(Element):
        """Driven element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            y: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.y = y

        def _get_laws(self, s):
            """Return the laws of the driven element."""
            return [
                RateLaw(s.y, pulse_train(s.t, 2.0, 0.5) + series(s.t, [1.0, 3.0], [0.0, 1.0], kind='previous')),
            ]

    system = CompositeSystem(elements=[DrivenElement('d')])
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.breakpoints.between(0.0, 5.0) == [0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 4.5]
    trajectory = simulate(system, until=5, dt=0.5)
    assert trajectory.get_values('d.y')[-1] == pytest.approx(1.5 + 2.0, abs=1e-6)


def test_schedule_ramps_a_parameter():
    """Test that a scheduled parameter takes the value of its function of time before every frame."""
    trajectory = simulate(
        CompositeSystem(elements=[FirstOrderElement('f')]),
        until=3,
        dt=1.0,
        schedule={'f.k': lambda t: 1.0 + t},
    )
    assert trajectory.parameters[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize('stepper', [EulerMaruyamaSolver, MilsteinSolver, HeunSolver])
def test_stratonovich_geometric_brownian_motion(stepper):
    """Test that the Stratonovich mean of geometric Brownian motion at 1 is exp(mu + sigma^2 / 2) within 3 SE."""
    system = (
        PassCompiler()
        .prepare(CompositeSystem(elements=[GeometricElement('g', convention='stratonovich')]))
        .run()
        .problem_
    )
    values = np.asarray(
        [
            stepper(dt=0.01, random_state=seed)
            .prepare(system)
            .run(StepSpec(0.0, system.initial_state, system.initial_parameters, 1.0))
            .state_[0]
            for seed in range(1000)
        ],
    )
    error = values.std() / math.sqrt(values.size)
    assert abs(values.mean() - math.exp(0.5 + 0.8**2 / 2)) < 3 * error


def test_noise_convention_adds_drift_correction():
    """Test that the Stratonovich and anti-Itô corrections add half and all of `g dg/dx` to the drift."""
    for noise, alpha in [('ito', 0.0), ('stratonovich', 0.5), ('anti-ito', 1.0)]:
        problem = (
            PassCompiler().prepare(CompositeSystem(elements=[GeometricElement('g', convention=noise)])).run().problem_
        )
        drift = problem.drift(0.0, np.asarray([2.0]), problem.initial_parameters)
        assert drift[0] == pytest.approx(0.5 * 2 + alpha * 0.8**2 * 2)


def test_unknown_noise_convention_is_rejected():
    """Test that a noise convention other than the three is rejected."""
    with pytest.raises(ValidationError, match='noise convention'):
        PassCompiler().prepare(CompositeSystem(elements=[GeometricElement('g', convention='marcus')])).run()


@pytest.mark.slow
def test_milstein_and_heun_converge_on_additive_noise():
    """Test that Milstein and Heun integrate an Ornstein-Uhlenbeck process to its stationary variance."""

    class OrnsteinElement(Element):
        """Ornstein element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the ornstein element."""
            return [RateLaw(s.x, -s.x), NoiseLaw(s.x, 1.0)]

    for stepper in (MilsteinSolver(dt=0.01), HeunSolver(dt=0.01)):
        trajectory = simulate(
            CompositeSystem(elements=[OrnsteinElement('o')]),
            until=1000,
            solver=stepper,
            dt=1.0,
            random_state=0,
        )
        assert np.var(trajectory.get_values('o.x')[100:]) == pytest.approx(0.5, rel=0.2)


def test_steppers_without_hybrid_path_reject_events():
    """Test that velocity Verlet and Gillespie name the steppers to use for events and delays."""
    problem = PassCompiler().prepare(BouncingBallSystem().prepare().system_).run().problem_
    with pytest.raises(SolverError, match='does not integrate delays or events'):
        VelocityVerletSolver(random_state=0).prepare(problem)
    with pytest.raises(SolverError):
        GillespieSolver(random_state=0).prepare(problem)


def test_fixed_step_longer_than_delay_is_rejected():
    """Test that a fixed step longer than the smallest delay raises an error."""
    with pytest.raises(SolverError, match='smallest delay'):
        simulate(CompositeSystem(elements=[DelayedElement('d')]), until=2, dt=2.0, solver=RK4Solver(dt=2.0))


def test_events_are_attached_to_frames():
    """Test that every frame carries the events since the previous one."""
    frames = []

    class Collector(BaseObserver):
        def _run(self, inputs):
            if inputs is not None:
                frames.append(inputs)

    simulate(BouncingBallSystem().prepare().system_, until=1.5, dt=0.5, observers=[Collector()])
    assert [len(frame.events) for frame in frames] == [0, 1, 0, 1]


def _short_delay(s):
    """Return a decay driven by the state a delay `tau` ago."""
    return [RateLaw(s.x, -delayed(s.x, s.tau))]


@pytest.mark.parametrize('tau', [1e-4, 1e-6])
def test_very_short_delay_with_scipy(tau):
    """Test that ScipyODE integrates a delay shorter than its first step and matches the delay-free decay."""

    class ShortElement(Element):
        """Short element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec(history=1.0)] = 1.0,
            tau: Annotated[object, ParameterSpec()] = tau,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.tau = tau

        def _get_laws(self, s):
            """Return the laws of the short element."""
            return _short_delay(s)

    system = CompositeSystem(elements=[ShortElement('d')])
    trajectory = simulate(system, until=0.2, solver=ScipyODESolver())
    assert trajectory.get_values('d.x')[-1] == pytest.approx(math.exp(-0.2), abs=1e-3)


def _monthly_dose(s):
    """Return a decay with a short dose every 21 days."""
    return [RateLaw(s.c, -s.k * s.c + pulse_train(s.t, 21, 1 / 24))]


def test_periodic_breakpoints_over_unbounded_runs():
    """Test that periodic breakpoints bound a persistent integrator, which looks for the next one without an end."""

    class DosedElement(Element):
        """Dosed element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            c: Annotated[object, StateSpec()] = 0.0,
            k: Annotated[object, ParameterSpec()] = 0.1,
        ) -> None:
            super().__init__(label, visual=visual)
            self.c = c
            self.k = k

        def _get_laws(self, s):
            """Return the laws of the dosed element."""
            return _monthly_dose(s)

    system = CompositeSystem(elements=[DosedElement('a')])
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.breakpoints.next_after(1.0) == 21.0
    trajectory = simulate(system, until=10, dt=0.5)
    expected = (1 - math.exp(-0.1 / 24)) / 0.1 * math.exp(-0.1 * (10 - 1 / 24))
    assert trajectory.get_values('a.c')[-1] == pytest.approx(expected, rel=1e-5)


def _scaled_delay(s):
    """Return a decay driven by the state twice the parameter `tau` ago."""
    return [RateLaw(s.x, -delayed(s.x, 2 * s.tau))]


def test_delay_of_parameter_expression():
    """Test that a delay that is an expression of parameters takes its smallest value from their values."""

    class ScaledElement(Element):
        """Scaled element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 1.0,
            tau: Annotated[object, ParameterSpec()] = 0.5,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.tau = tau

        def _get_laws(self, s):
            """Return the laws of the scaled element."""
            return _scaled_delay(s)

    system = CompositeSystem(elements=[ScaledElement('d')])
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.delays[0].minimum == 1.0
    trajectory = simulate(system, until=3, dt=0.5)
    assert trajectory.get_values('d.x')[-1] == pytest.approx(-1 / 6, abs=1e-6)


def test_state_conditions_restart_the_integrator_without_stalling():
    """Test that the bathtub drains without a stall warning, keeps its water nonnegative, and records no events."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        trajectory = simulate(BathtubSystem(water=20.0).prepare().system_, until=6, dt=1)
    assert np.all(np.asarray(trajectory.get_values('tub.water')) >= -1e-6)
    assert not trajectory.events


def test_state_conditions_compile_into_hidden_events():
    """Test that the condition of a rate law compiles into a hidden event without a reset."""
    problem = PassCompiler().prepare(BathtubSystem(water=20.0).prepare().system_).run().problem_
    hidden = [event for event in problem.events if event.hidden]
    assert len(hidden) == 1
    assert hidden[0].reset is None


def test_sign_conditions_compile_into_hidden_events():
    """Test that the escapement `sign(|theta| - theta_c)` of each clock compiles into a hidden event."""
    problem = PassCompiler().prepare(HuygensClocksSystem().prepare().system_).run().problem_
    assert sum(event.hidden for event in problem.events) == 2
