"""Test the renewal stepper and the renewals of fixed-step steppers."""

import itertools
from typing import Annotated

import numpy as np
import pytest
import rerun as rr

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import JumpLaw, RateLaw
from skcomplex.core.components.laws.utils import bernoulli, draw, mean, normal
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RenewalSolver, RK4Solver, ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.errors import SolverError
from skcomplex.core.specs import ParameterSpec, RenewalRecordSpec, StateSpec, VisualSpec
from skcomplex.core.types import FormKind


class CounterElement(Element):
    """Counter element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n

    def _get_laws(self, s):
        """Return the laws of the counter element."""
        return [JumpLaw(waiting=draw('wait', 'expon'), reset={s.n: s.n + 1}, label='ticks')]


class ResettingElement(Element):
    """Resetting element."""

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
        """Return the laws of the resetting element."""
        return [JumpLaw(waiting=draw('wait', 'expon'), reset={s.x: draw('value', 'norm', loc=2, scale=3)})]


class KickedElement(Element):
    """Kicked element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[object, StateSpec()] = 0.0,
        b: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.b = b

    def _get_laws(self, s):
        """Return the laws of the kicked element."""
        return [
            JumpLaw(waiting=draw('wait', 'expon'), reset={s.a: normal('kick', 0, 1), s.b: -normal('kick', 0, 1)}),
        ]


class WaitingElement(Element):
    """Waiting element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 0.0,
        alpha: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.alpha = alpha

    def _get_laws(self, s):
        """Return the laws of the waiting element."""
        return [JumpLaw(waiting=draw('wait', 'pareto', b=s.alpha), reset={s.n: s.n + 1})]


class FlierElement(Element):
    """Flier element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        v: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v

    def _get_laws(self, s):
        """Return the laws of the flier element."""
        return [
            RateLaw(s.x, s.v),
            JumpLaw(waiting=draw('flight', 'pareto', b=1.5, scale=0.5), reset={s.v: 2 * bernoulli('sign', 0.5) - 1}),
        ]


class FollowingElement(Element):
    """Following element."""

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
        """Return the laws of the following element."""
        return [
            RateLaw(s.x, mean(s.x) - s.x + 1),
            JumpLaw(waiting=draw('wait', 'expon'), reset={s.x: s.x + normal('jump')}),
        ]


def _trajectory(system, until, dt, **settings):
    """Return the trajectory of a system run with a seed."""
    return Simulation(system, dt=dt, random_state=0, **settings).prepare().run(until).trajectory_


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
    trajectory = _trajectory(System(elements=[GroupElement('r', element=ResettingElement, size=1000)]), 100, 100)
    values = np.concatenate([record.after[:, 0] for record in trajectory.events])
    assert values.size > 99000
    assert abs(values.mean() - 2) < 0.05
    assert abs(values.std() - 3) < 0.05


def test_renewal_stepper_shares_a_draw_within_a_renewal():
    """Test that two uses of one draw name in a reset read the same value at every renewal."""
    trajectory = _trajectory(System(elements=[GroupElement('k', element=KickedElement, size=20)]), 50, 1)
    after = np.concatenate([record.after for record in trajectory.events])
    assert after.shape[0] > 500
    assert np.all(after[:, 0] + after[:, 1] == 0)


def test_renewal_stepper_reproduces_runs_with_one_seed():
    """Test that two runs with random state 0 give equal trajectories and records."""
    first = _trajectory(System(elements=[GroupElement('k', element=KickedElement, size=20)]), 20, 1)
    second = _trajectory(System(elements=[GroupElement('k', element=KickedElement, size=20)]), 20, 1)
    assert np.array_equal(first.states, second.states)
    assert all(np.array_equal(a.times, b.times) for a, b in zip(first.events, second.events, strict=True))


def test_renewal_stepper_counts_exponential_renewals():
    """Test that exponential waits of mean 1 give 1000 renewals by time 1000 within 2 standard errors over 100 runs."""
    problem = compile_system(System(elements=[CounterElement('c')]))
    counts = [
        (Simulation(problem, dt=1000, random_state=seed).prepare().run(1000).trajectory_.values('c.n')[-1])
        for seed in range(100)
    ]
    assert abs(np.mean(counts) - 1000) < 2 * np.std(counts) / np.sqrt(len(counts))


def test_renewal_stepper_counts_power_law_renewals():
    """Test that Pareto waits of tail exponent 0.5 for 10000 members give a mean count growing as t^0.5."""
    trajectory = _trajectory(System(elements=[GroupElement('w', element=WaitingElement, size=10000)]), 10000, 100)
    times, counts = trajectory.times[10:], trajectory.values('w.n')[10:].mean(axis=1)
    assert np.polyfit(np.log(times), np.log(counts), 1)[0] == pytest.approx(0.5, abs=0.05)


def test_renewal_stepper_holds_states_between_renewals():
    """Test that every frame holds the states the last renewal at or before its time set."""
    trajectory = _trajectory(System(elements=[CounterElement('c')]), 20, 0.1)
    times = np.concatenate([record.times for record in trajectory.events])
    expected = np.searchsorted(times, trajectory.times, side='right')
    assert np.array_equal(trajectory.values('c.n'), expected)


def test_renewal_stepper_exposes_the_renewal_count_observable():
    """Test that the observable of a renewal law counts the renewals recorded up to every frame."""
    trajectory = _trajectory(System(elements=[GroupElement('c', element=CounterElement, size=5)]), 10, 0.5)
    times = np.concatenate([record.times for record in trajectory.events])
    assert np.array_equal(
        trajectory.values('ticks.renewals'),
        np.searchsorted(np.sort(times), trajectory.times, 'right'),
    )


def test_renewal_stepper_streams_renewals_to_rerun(monkeypatch, tmp_path):
    """Test that a recording logs the renewal count observable and a text entry for the renewals of every frame."""
    logged = []
    original = rr.RecordingStream.log

    def log(self, path, *args, **kwargs):
        logged.append((path, type(args[0]).__name__))
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(rr.RecordingStream, 'log', log)
    observer = RerunObserver('renewals', save_path=tmp_path / 'run.rrd')
    Simulation(
        System(elements=[GroupElement('c', element=CounterElement, size=5)]),
        dt=1,
        observers=[observer],
        random_state=0,
    ).prepare().run(5)
    assert logged.count(('ticks.renewals', 'Scalars')) == 6
    assert ('renewals/ticks', 'TextLog') in logged


@pytest.mark.parametrize(
    'system',
    [GroupElement('f', element=FlierElement, size=1000), GroupElement('f', element=FollowingElement, size=1000)],
)
def test_rk4_renews_members_on_their_own_clocks(system):
    """Test that members renew at their own times, with independent and interacting members."""
    trajectory = _trajectory(System(elements=[system]), 10, 1, solver=RK4Solver(dt=0.1))
    clocks = [tuple(_member_times(trajectory, 'f', member)) for member in range(1000)]
    renewing = [clock for clock in clocks if clock]
    assert len(renewing) > 950
    assert len(set(renewing)) == len(renewing)


def test_rk4_moves_ballistically_between_renewals():
    """Test that the position at every renewal is the previous one plus the velocity times the waiting time."""
    trajectory = _trajectory(
        System(elements=[GroupElement('f', element=FlierElement, size=50)]),
        50,
        1,
        solver=RK4Solver(dt=0.25),
    )
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
    record = _trajectory(
        System(elements=[GroupElement('f', element=FlierElement, size=3)]),
        5,
        5,
        solver=RK4Solver(dt=0.5),
    ).events[0]
    assert isinstance(record, RenewalRecordSpec)
    assert record.states == ('f.x', 'f.v')
    assert record.before.shape == record.after.shape == (record.members.size, 2)


def test_stepper_rejects_renewals_it_does_not_fire():
    """Test that a stepper without renewals names the steppers that fire them."""
    problem = compile_system(System(elements=[FlierElement('f')]))
    with pytest.raises(SolverError, match=r'ScipyODESolver does not fire renewal laws. Use RK4Solver'):
        ScipyODESolver(random_state=0).prepare(problem)


def test_renewal_stepper_solves_the_renewal_form():
    """Test that the renewal stepper is the default solver of the renewal form."""
    simulation = Simulation(System(elements=[CounterElement('c')]), random_state=0).prepare().run(1)
    assert simulation.problem_.form == FormKind.RENEWAL
    assert isinstance(simulation.stepper_, RenewalSolver)
