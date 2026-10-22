"""Test the renewal stepper and the renewals of fixed-step steppers."""

import itertools

import numpy as np
import pytest
import rerun as rr

from skcomplex.core.base import Form, SolverError
from skcomplex.core.compile import RenewalRecord, compile_system
from skcomplex.core.model import Element, Jump, Rate, System, bernoulli, draw, mean, normal
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4, RenewalStepper, ScipyODE
from skcomplex.core.viz import RerunObserver

Counter = Element.define(
    'Counter',
    states={'n': 0.0},
    laws=lambda s: [Jump(waiting=draw('wait', 'expon'), reset={s.n: s.n + 1}, label='ticks')],
)
Resetting = Element.define(
    'Resetting',
    states={'x': 0.0},
    laws=lambda s: [Jump(waiting=draw('wait', 'expon'), reset={s.x: draw('value', 'norm', loc=2, scale=3)})],
)
Kicked = Element.define(
    'Kicked',
    states={'a': 0.0, 'b': 0.0},
    laws=lambda s: [Jump(waiting=draw('wait', 'expon'), reset={s.a: normal('kick', 0, 1), s.b: -normal('kick', 0, 1)})],
)
Waiting = Element.define(
    'Waiting',
    states={'n': 0.0},
    parameters={'alpha': 0.5},
    laws=lambda s: [Jump(waiting=draw('wait', 'pareto', b=s.alpha), reset={s.n: s.n + 1})],
)
Flier = Element.define(
    'Flier',
    states={'x': 0.0, 'v': 1.0},
    laws=lambda s: [
        Rate(s.x, s.v),
        Jump(waiting=draw('flight', 'pareto', b=1.5, scale=0.5), reset={s.v: 2 * bernoulli('sign', 0.5) - 1}),
    ],
)
Following = Element.define(
    'Following',
    states={'x': 0.0},
    laws=lambda s: [
        Rate(s.x, mean(s.x) - s.x + 1),
        Jump(waiting=draw('wait', 'expon'), reset={s.x: s.x + normal('jump')}),
    ],
)


def _trajectory(system, until, dt, **settings):
    """Return the trajectory of a system run with a seed."""
    return Simulation(system, dt=dt, random_state=0, **settings).run(until).trajectory_


def _member_times(trajectory, label, member):
    """Return the renewal times of one member of a renewal law."""
    return np.asarray(
        [
            time
            for record in trajectory.events
            if record.label == label
            for time, renewing in zip(record.times, record.members, strict=True)
            if renewing == member
        ],
    )


def test_renewal_stepper_draws_reset_values_from_scipy():
    """Test that 100000 reset values of a SciPy normal draw have mean 2 and scale 3."""
    trajectory = _trajectory(System(components=[Resetting.group('r', 1000)]), 100, 100)
    values = np.concatenate([record.after[:, 0] for record in trajectory.events])
    assert values.size > 99000
    assert abs(values.mean() - 2) < 0.05
    assert abs(values.std() - 3) < 0.05


def test_renewal_stepper_shares_a_draw_within_a_renewal():
    """Test that two uses of one draw name in a reset read the same value at every renewal."""
    trajectory = _trajectory(System(components=[Kicked.group('k', 20)]), 50, 1)
    after = np.concatenate([record.after for record in trajectory.events])
    assert after.shape[0] > 500
    assert np.all(after[:, 0] + after[:, 1] == 0)


def test_renewal_stepper_reproduces_runs_with_one_seed():
    """Test that two runs with random state 0 give equal trajectories and records."""
    first = _trajectory(System(components=[Kicked.group('k', 20)]), 20, 1)
    second = _trajectory(System(components=[Kicked.group('k', 20)]), 20, 1)
    assert np.array_equal(first.states, second.states)
    assert all(np.array_equal(a.times, b.times) for a, b in zip(first.events, second.events, strict=True))


def test_renewal_stepper_counts_exponential_renewals():
    """Test that exponential waits of mean 1 give 1000 renewals by time 1000 within 2 standard errors over 100 runs."""
    problem = compile_system(System(components=[Counter('c')]))
    counts = [Simulation(problem, dt=1000, random_state=seed).run(1000).trajectory_['c.n'][-1] for seed in range(100)]
    assert abs(np.mean(counts) - 1000) < 2 * np.std(counts) / np.sqrt(len(counts))


def test_renewal_stepper_counts_power_law_renewals():
    """Test that Pareto waits of tail exponent 0.5 for 10000 members give a mean count growing as t^0.5."""
    trajectory = _trajectory(System(components=[Waiting.group('w', 10000)]), 10000, 100)
    times, counts = trajectory.times[10:], trajectory['w.n'][10:].mean(axis=1)
    assert np.polyfit(np.log(times), np.log(counts), 1)[0] == pytest.approx(0.5, abs=0.05)


def test_renewal_stepper_holds_states_between_renewals():
    """Test that every frame holds the states the last renewal at or before its time set."""
    trajectory = _trajectory(System(components=[Counter('c')]), 20, 0.1)
    times = np.concatenate([record.times for record in trajectory.events])
    expected = np.searchsorted(times, trajectory.times, side='right')
    assert np.array_equal(trajectory['c.n'], expected)


def test_renewal_stepper_exposes_the_renewal_count_observable():
    """Test that the observable of a renewal law counts the renewals recorded up to every frame."""
    trajectory = _trajectory(System(components=[Counter.group('c', 5)]), 10, 0.5)
    times = np.concatenate([record.times for record in trajectory.events])
    assert np.array_equal(trajectory['ticks.renewals'], np.searchsorted(np.sort(times), trajectory.times, 'right'))


def test_renewal_stepper_streams_renewals_to_rerun(monkeypatch, tmp_path):
    """Test that a recording logs the renewal count observable and a text entry for the renewals of every frame."""
    logged = []
    original = rr.RecordingStream.log

    def log(self, path, *args, **kwargs):
        logged.append((path, type(args[0]).__name__))
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(rr.RecordingStream, 'log', log)
    observer = RerunObserver('renewals', save_path=tmp_path / 'run.rrd')
    Simulation(System(components=[Counter.group('c', 5)]), dt=1, observers=[observer], random_state=0).run(5)
    assert logged.count(('ticks.renewals', 'Scalars')) == 6
    assert ('renewals/ticks', 'TextLog') in logged


@pytest.mark.parametrize('system', [Flier.group('f', 1000), Following.group('f', 1000)])
def test_rk4_renews_members_on_their_own_clocks(system):
    """Test that members renew at their own times, with independent and interacting members."""
    trajectory = _trajectory(System(components=[system]), 10, 1, solver=RK4(dt=0.1))
    clocks = [tuple(_member_times(trajectory, 'f', member)) for member in range(1000)]
    renewing = [clock for clock in clocks if clock]
    assert len(renewing) > 950
    assert len(set(renewing)) == len(renewing)


def test_rk4_moves_ballistically_between_renewals():
    """Test that the position at every renewal is the previous one plus the velocity times the waiting time."""
    trajectory = _trajectory(System(components=[Flier.group('f', 50)]), 50, 1, solver=RK4(dt=0.25))
    for member in range(50):
        rows = [
            (time, before, after)
            for record in trajectory.events
            for time, renewing, before, after in zip(
                record.times,
                record.members,
                record.before,
                record.after,
                strict=True,
            )
            if renewing == member
        ]
        for (start, _, earlier), (end, later, _) in itertools.pairwise(rows):
            assert abs(later[0] - (earlier[0] + earlier[1] * (end - start))) < 1e-12


def test_renewal_records_name_their_states():
    """Test that a renewal record holds the time, the member, and the member's states before and after it."""
    record = _trajectory(System(components=[Flier.group('f', 3)]), 5, 5, solver=RK4(dt=0.5)).events[0]
    assert isinstance(record, RenewalRecord)
    assert record.states == ('f.x', 'f.v')
    assert record.before.shape == record.after.shape == (record.members.size, 2)


def test_stepper_rejects_renewals_it_does_not_fire():
    """Test that a stepper without renewals names the steppers that fire them."""
    problem = compile_system(System(components=[Flier('f')]))
    with pytest.raises(SolverError, match=r'ScipyODE does not fire renewal laws. Use RK4'):
        ScipyODE().initialize(problem, np.random.default_rng(0))


def test_renewal_stepper_solves_the_renewal_form():
    """Test that the renewal stepper is the default solver of the renewal form."""
    simulation = Simulation(System(components=[Counter('c')]), random_state=0).run(1)
    assert simulation.problem_.form is Form.RENEWAL
    assert isinstance(simulation.stepper_, RenewalStepper)
