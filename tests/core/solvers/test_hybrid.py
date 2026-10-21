"""Test delays, events, drivers, and noise conventions through the steppers and the simulation."""

import math
import warnings

import numpy as np
import pytest
from sklearn.base import clone

from skcomplex.core.base import SolverError, ValidationError
from skcomplex.core.compile import compile_system
from skcomplex.core.model import Element, Event, Noise, Rate, System, delayed, pulse_train, series, step
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4, EulerMaruyama, Gillespie, Heun, Milstein, ScipyODE, VelocityVerlet
from skcomplex.models.biology.ecology import hutchinson
from skcomplex.models.biology.physiology import mackey_glass
from skcomplex.models.engineering.processes import bathtub
from skcomplex.models.physics.mechanics import bouncing_ball, huygens_clocks

Clock = Element.define('Clock', states={'x': 0.0}, laws=lambda s: [Rate(s.x, 1), Event(s.x - 3.2, terminal=True)])

FirstOrder = Element.define(
    'FirstOrder',
    states={'y': 0.0},
    parameters={'k': 1.0},
    laws=lambda s: [Rate(s.y, -s.k * s.y + step(s.t, 5.0))],
)

Stratonovich = Element.define(
    'Stratonovich',
    states={'x': 1.0},
    parameters={'mu': 0.5, 'sigma': 0.8},
    laws=lambda s: [Rate(s.x, s.mu * s.x), Noise(s.x, s.sigma * s.x)],
)

Delayed = Element.define(
    'Delayed',
    states={'x': 1.0},
    laws=lambda s: [Rate(s.x, -delayed(s.x, 1.0))],
)


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
    solver = ScipyODE(rtol=1e-9, atol=1e-12)
    first = Simulation(mackey_glass(tau=tau), dt=0.5, solver=solver).run(until=until).trajectory_
    second = Simulation(mackey_glass(x=0.5 + 1e-8, tau=tau), dt=0.5, solver=clone(solver)).run(until=until).trajectory_
    distance = np.abs(first['cells.x'] - second['cells.x'])
    starts = np.arange(0.0, until, window)
    peaks = [distance[(first.times >= start) & (first.times < start + window)].max() for start in starts]
    return np.polyfit(starts, np.log(peaks), 1)[0]


def test_mackey_glass_is_chaotic_at_seventeen():
    """Test that Mackey-Glass at tau = 17 separates nearby runs exponentially, a positive largest exponent."""
    assert _divergence_rate(17.0) > 0.003


def test_mackey_glass_is_periodic_at_six():
    """Test that Mackey-Glass at tau = 6 settles on a periodic orbit whose maxima repeat."""
    values = Simulation(mackey_glass(tau=6.0), dt=0.1).run(until=600).trajectory_['cells.x'][3000:]
    peaks = values[1:-1][(values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])]
    assert peaks.size > 10
    assert np.ptp(peaks) < 1e-3
    assert abs(_divergence_rate(6.0)) < 0.0005


def test_hutchinson_oscillates_above_threshold():
    """Test that the delayed logistic oscillates around K with a stable amplitude when r tau is above pi/2."""
    values = Simulation(hutchinson(r=1.0, tau=2.0), dt=0.1).run(until=200).trajectory_['population.n']
    middle, late = values[1000:1500], values[1500:]
    assert np.ptp(late) > 50
    assert np.ptp(late) == pytest.approx(np.ptp(middle), rel=0.01)
    assert late.mean() == pytest.approx(100, rel=0.5)


def test_hutchinson_settles_below_threshold():
    """Test that the delayed logistic settles at K when r tau is below pi/2."""
    values = Simulation(hutchinson(r=1.0, tau=1.0), dt=0.1).run(until=200).trajectory_['population.n']
    assert values[-1] == pytest.approx(100, abs=1e-3)


@pytest.mark.parametrize('solver', ['auto', 'rk4'])
def test_delayed_decay_matches_method_of_steps(solver):
    """Test that `x' = -x(t - 1)` with history 1 matches its exact solution `1 - t + (t - 1)^2 / 2` on [1, 2]."""
    trajectory = Simulation(System(elements=[Delayed('d')]), dt=0.1, solver=solver).run(until=2).trajectory_
    times = trajectory.times
    exact = np.where(times <= 1, 1 - times, 1 - times + (times - 1) ** 2 / 2)
    assert trajectory['d.x'] == pytest.approx(exact, abs=1e-6)


def test_declared_history_is_read_before_the_start():
    """Test that a history given as a function of time feeds the delayed state before the start."""
    system = System(elements=[Delayed('d')], history={'d.x': lambda t: 1 + t})
    trajectory = Simulation(system, dt=0.1).run(until=1).trajectory_
    assert trajectory['d.x'][-1] == pytest.approx(1 - 0.5, abs=1e-5)


def test_history_names_states():
    """Test that a history naming no state is rejected when compiling."""
    with pytest.raises(ValidationError, match='not states'):
        compile_system(System(elements=[Delayed('d')], history={'d.y': 1.0}))


@pytest.mark.parametrize('solver', ['auto', 'rk4'])
def test_bouncing_ball_impact_times(solver):
    """Test that the impact times of a ball with restitution 0.8 match the analytic times within 1e-6."""
    trajectory = Simulation(bouncing_ball(), dt=0.01, solver=solver).run(until=3.3).trajectory_
    times = np.asarray([event.time for event in trajectory.events])
    assert times.size == 7
    assert times == pytest.approx(_impact_times(times.size), abs=1e-6)
    first = trajectory.events[0]
    assert first.label == 'impact'
    assert first.after[1] == pytest.approx(-0.8 * first.before[1])


def test_terminal_event_ends_the_run():
    """Test that a terminal event at 3.2 ends the run there with the state at the crossing."""
    trajectory = Simulation(System(elements=[Clock('clock')]), dt=0.5).run(until=10).trajectory_
    assert trajectory.times[-1] == pytest.approx(3.2)
    assert trajectory['clock.x'][-1] == pytest.approx(3.2)
    assert [event.label for event in trajectory.events] == ['clock']


def test_event_without_reset_fires_once():
    """Test that an event that keeps the state fires once at its crossing and the run goes on past it."""
    Marker = Element.define('Marker', states={'x': 1.0}, laws=lambda s: [Rate(s.x, -1), Event(s.x, direction=-1)])
    trajectory = Simulation(System(elements=[Marker('m')]), dt=0.5).run(until=2).trajectory_
    assert [event.time for event in trajectory.events] == pytest.approx([1.0])
    assert trajectory['m.x'][-1] == pytest.approx(-1.0)


@pytest.mark.parametrize('solver', ['auto', 'rk4'])
def test_chattering_event_raises(solver):
    """Test that an event whose reset puts the state back just before the guard, so it fires again, raises an error."""
    Sticky = Element.define(
        'Sticky',
        states={'x': 1.0},
        laws=lambda s: [Rate(s.x, -1), Event(s.x, {s.x: 1e-12}, direction=-1)],
    )
    with pytest.raises(SolverError, match='chattering'):
        Simulation(System(elements=[Sticky('c')]), dt=0.5, solver=solver).run(until=2)


@pytest.mark.parametrize('solver', ['auto', 'rk4'])
def test_step_response(solver):
    """Test that a first-order system is unforced before a step at 5 and follows the step response after it."""
    trajectory = Simulation(System(elements=[FirstOrder('f')]), dt=0.25, solver=solver).run(until=10).trajectory_
    times, values = trajectory.times, trajectory['f.y']
    assert np.all(values[times < 5] == 0.0)
    assert values == pytest.approx(np.where(times >= 5, 1 - np.exp(-(times - 5)), 0.0), abs=1e-6)


def test_pulse_train_and_series_breakpoints():
    """Test that periodic pulses and interpolated series give the breakpoints of their discontinuities."""
    Driven = Element.define(
        'Driven',
        states={'y': 0.0},
        laws=lambda s: [Rate(s.y, pulse_train(s.t, 2.0, 0.5) + series(s.t, [1.0, 3.0], [0.0, 1.0], kind='previous'))],
    )
    problem = compile_system(System(elements=[Driven('d')]))
    assert problem.breakpoints.between(0.0, 5.0) == [0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 4.5]
    trajectory = Simulation(problem, dt=0.5).run(until=5).trajectory_
    assert trajectory['d.y'][-1] == pytest.approx(1.5 + 2.0, abs=1e-6)


def test_schedule_ramps_a_parameter():
    """Test that a scheduled parameter takes the value of its function of time before every frame."""
    simulation = Simulation(
        System(elements=[FirstOrder('f')]),
        dt=1.0,
        schedule={'f.k': lambda t: 1.0 + t},
    ).run(until=3)
    assert simulation.trajectory_.parameters[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize('stepper', [EulerMaruyama, Milstein, Heun])
def test_stratonovich_geometric_brownian_motion(stepper):
    """Test that the Stratonovich mean of geometric Brownian motion at 1 is exp(mu + sigma^2 / 2) within 2 SE."""
    system = compile_system(System(elements=[Stratonovich('g')], noise='stratonovich'))
    values = np.asarray(
        [
            Simulation(system, dt=1.0, solver=stepper(dt=0.01), random_state=seed, record=False).run(until=1).state_[0]
            for seed in range(1000)
        ],
    )
    error = values.std() / math.sqrt(values.size)
    assert abs(values.mean() - math.exp(0.5 + 0.8**2 / 2)) < 2 * error


def test_noise_convention_adds_drift_correction():
    """Test that the Stratonovich and anti-Itô corrections add half and all of `g dg/dx` to the drift."""
    for noise, alpha in [('ito', 0.0), ('stratonovich', 0.5), ('anti-ito', 1.0)]:
        problem = compile_system(System(elements=[Stratonovich('g')], noise=noise))
        drift = problem.drift(0.0, np.asarray([2.0]), problem.initial_parameters)
        assert drift[0] == pytest.approx(0.5 * 2 + alpha * 0.8**2 * 2)


def test_unknown_noise_convention_is_rejected():
    """Test that a noise convention other than the three is rejected."""
    with pytest.raises(ValidationError, match='noise'):
        compile_system(System(elements=[Stratonovich('g')], noise='marcus'))


def test_milstein_and_heun_converge_on_additive_noise():
    """Test that Milstein and Heun integrate an Ornstein-Uhlenbeck process to its stationary variance."""
    Ornstein = Element.define('Ornstein', states={'x': 0.0}, laws=lambda s: [Rate(s.x, -s.x), Noise(s.x, 1.0)])
    for stepper in (Milstein(dt=0.01), Heun(dt=0.01)):
        values = Simulation(System(elements=[Ornstein('o')]), solver=stepper, dt=1.0, random_state=0).run(1000)
        assert np.var(values.trajectory_['o.x'][100:]) == pytest.approx(0.5, rel=0.2)


def test_steppers_without_hybrid_path_reject_events():
    """Test that velocity Verlet and Gillespie name the steppers to use for events and delays."""
    problem = compile_system(bouncing_ball())
    with pytest.raises(SolverError, match='does not integrate delays or events'):
        VelocityVerlet().initialize(problem, np.random.default_rng(0))
    with pytest.raises(SolverError):
        Gillespie().initialize(problem, np.random.default_rng(0))


def test_fixed_step_longer_than_delay_is_rejected():
    """Test that a fixed step longer than the smallest delay raises an error."""
    with pytest.raises(SolverError, match='smallest delay'):
        Simulation(System(elements=[Delayed('d')]), dt=2.0, solver=RK4(dt=2.0)).run(until=2)


def test_events_are_attached_to_frames():
    """Test that every frame carries the events since the previous one."""
    frames = []

    class Collector:
        def on_start(self, problem):
            pass

        def on_frame(self, frame):
            frames.append(frame)

        def on_stop(self):
            pass

    Simulation(bouncing_ball(), dt=0.5, observers=[Collector()]).run(until=1.5)
    assert [len(frame.events) for frame in frames] == [0, 1, 0, 1]


def _short_delay(s):
    """Return a decay driven by the state a delay `tau` ago."""
    return [Rate(s.x, -delayed(s.x, s.tau))]


@pytest.mark.parametrize('tau', [1e-4, 1e-6])
def test_very_short_delay_with_scipy(tau):
    """Test that ScipyODE integrates a delay shorter than its first step and matches the delay-free decay."""
    Short = Element.define('Short', states={'x': 1.0}, parameters={'tau': tau}, laws=_short_delay)
    system = System(elements=[Short('d')], history={'d.x': 1.0})
    trajectory = Simulation(system, solver=ScipyODE()).run(until=0.2).trajectory_
    assert trajectory['d.x'][-1] == pytest.approx(math.exp(-0.2), abs=1e-3)


def _monthly_dose(s):
    """Return a decay with a short dose every 21 days."""
    return [Rate(s.c, -s.k * s.c + pulse_train(s.t, 21, 1 / 24))]


def test_periodic_breakpoints_over_unbounded_runs():
    """Test that periodic breakpoints bound a persistent integrator, which looks for the next one without an end."""
    Dosed = Element.define('Dosed', states={'c': 0.0}, parameters={'k': 0.1}, laws=_monthly_dose)
    problem = compile_system(System(elements=[Dosed('a')]))
    assert problem.breakpoints.next_after(1.0) == 21.0
    trajectory = Simulation(problem, dt=0.5).run(until=10).trajectory_
    expected = (1 - math.exp(-0.1 / 24)) / 0.1 * math.exp(-0.1 * (10 - 1 / 24))
    assert trajectory['a.c'][-1] == pytest.approx(expected, rel=1e-5)


def _scaled_delay(s):
    """Return a decay driven by the state twice the parameter `tau` ago."""
    return [Rate(s.x, -delayed(s.x, 2 * s.tau))]


def test_delay_of_parameter_expression():
    """Test that a delay that is an expression of parameters takes its smallest value from their values."""
    Scaled = Element.define('Scaled', states={'x': 1.0}, parameters={'tau': 0.5}, laws=_scaled_delay)
    problem = compile_system(System(elements=[Scaled('d')]))
    assert problem.delays[0].minimum == 1.0
    trajectory = Simulation(problem, dt=0.5).run(until=3).trajectory_
    assert trajectory['d.x'][-1] == pytest.approx(-1 / 6, abs=1e-6)


def test_state_conditions_restart_the_integrator_without_stalling():
    """Test that the bathtub drains without a stall warning, keeps its water nonnegative, and records no events."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        trajectory = Simulation(bathtub(water=20.0), dt=1).run(until=6).trajectory_
    assert np.all(np.asarray(trajectory['tub.water']) >= -1e-6)
    assert not trajectory.events


def test_state_conditions_compile_into_hidden_events():
    """Test that the condition of a rate law compiles into a hidden event without a reset."""
    problem = compile_system(bathtub(water=20.0))
    hidden = [event for event in problem.events if event.hidden]
    assert len(hidden) == 1
    assert hidden[0].reset is None


def test_sign_conditions_compile_into_hidden_events():
    """Test that the escapement `sign(|theta| - theta_c)` of each clock compiles into a hidden event."""
    problem = compile_system(huygens_clocks())
    assert sum(event.hidden for event in problem.events) == 2
