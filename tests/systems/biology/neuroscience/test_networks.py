"""Test the `_networks` module of `skcomplex.systems.biology.neuroscience`."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import laplacian, logistic
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.biology.neuroscience import (
    DelayedLIFNetworkSystem,
    InhibitorDecayInteraction,
    InhibitorDriveInteraction,
    InhibitorElement,
    LegionSystem,
    TermanWangElement,
    TermanWangInteraction,
)


class _PreviousOscillationInteraction(Interaction):
    """Oscillation with the excitation by the neighbours in the same law, as before the coupling interaction."""

    def __init__(self, label, *, target: Role):
        super().__init__(label)
        self.target = target

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the fast variable with its excitation by the neighbours, and the slow recovery."""
        super()._prepare(context)
        s = context.target
        active = s.m * logistic(s.x - s.theta_x, 0, s.kappa)
        excitation = s.W * s.m * (laplacian(active) + 4 * active)
        self.laws_ = [
            RateLaw(s.x, 3 * s.x - s.x**3 + 2 - s.y + s.I + excitation),
            RateLaw(s.y, s.epsilon * (s.gamma * (1 + tanh(s.x / s.beta)) - s.y)),
        ]


def test_delayed_lif_network_rates():
    """Test that uncoupled neurons fire at the leaky integrate-and-fire period and that inhibition slows them."""
    period = 20.0 * math.log((25.0 - 10.0) / (25.0 - 20.0))
    counts = []
    for J in (0.0, -2.0):
        trajectory = DelayedLIFNetworkSystem(5, J=J, probability=0.5).prepare().system_.simulate(until=60, dt=1.0)
        events = trajectory.events
        counts.append(len(events))
        if J == 0.0:
            index = list(trajectory.problem.states.names).index('neuron_0.v')
            times = [event.time for event in events if event.before[index] != event.after[index]]
            assert np.diff(times) == pytest.approx(period, rel=1e-3)
    assert counts[1] < counts[0]


def test_legion_segments_objects():
    """Test that LEGION synchronizes each object, alternates between objects, and keeps the background silent."""
    trajectory = LegionSystem(epsilon=0.1).prepare().simulate(dt=0.5, until=120)
    active = trajectory.get_values('oscillators.x').reshape(len(trajectory.times), -1) > 0
    first, second = active[:, [0, 1, 4]], active[:, [11, 15]]
    assert first.mean() > 0.1
    assert second.mean() > 0.1
    assert (first[:, 0] == first[:, 1]).mean() > 0.97
    assert (second[:, 0] == second[:, 1]).mean() > 0.97
    after = trajectory.times > 20
    assert not (first.any(axis=1) & second.any(axis=1))[after].any()
    assert first[after].any()
    assert second[after].any()
    assert active[:, [2, 3, 5, 6, 7]].mean() < 0.02


def test_legion_defaults_fire_both_objects_in_a_short_run():
    """Test that both objects of the default LEGION network start an active phase within 100 time units."""
    trajectory = LegionSystem().prepare().simulate(dt=0.5, until=100)
    active = trajectory.get_values('oscillators.x').reshape(len(trajectory.times), -1) > 0
    for sites in ([0, 1, 4], [11, 15]):
        on = active[:, sites].any(axis=1)
        assert (on[1:] & ~on[:-1]).any()


def _previous_legion():
    """Return `LegionSystem` with its coupling written in the oscillation, as `laplacian(active) + 4 * active`."""
    system = LegionSystem().prepare().system_
    oscillation = _PreviousOscillationInteraction('oscillation', target='oscillators')
    interactions = [oscillation, system.interactions[1], *system.interactions[3:]]
    return System(elements=system.elements, interactions=interactions)


def test_legion_rates_equal_the_coupling_written_in_the_element():
    """Test that the rates of the rebuilt coupling equal those of the previous element at many random states."""
    problem = PassCompiler().prepare(LegionSystem().prepare().system_).run().problem_
    previous = PassCompiler().prepare(_previous_legion()).run().problem_
    assert list(problem.states.names) == list(previous.states.names)
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = rng.uniform(-3.0, 6.0, problem.initial_state.shape)
        rates = problem.drift(0.0, state, problem.initial_parameters)
        expected = previous.drift(0.0, state, previous.initial_parameters)
        assert np.abs(rates - expected).max() < 1e-12


def test_legion_short_trajectory_equals_the_coupling_written_in_the_element():
    """Test that the trajectory of the rebuilt coupling equals the previous one before rounding grows."""
    trajectory = LegionSystem().prepare().simulate(dt=0.5, until=1.5)
    expected = _previous_legion().prepare().simulate(dt=0.5, until=1.5)
    assert np.abs(trajectory.get_values('oscillators.x') - expected.get_values('oscillators.x')).max() < 1e-10
    assert np.abs(trajectory.get_values('oscillators.y') - expected.get_values('oscillators.y')).max() < 1e-10


def test_inhibitor_drive_interaction_follows_most_active_oscillator():
    """Test that the inhibitor relaxes toward the largest activity of the oscillators, 1 when one is active."""
    oscillators = TermanWangElement(
        'oscillators',
        x=np.array([[2.0, -2.0]]),
        m=np.array([[1.0, 1.0]]),
        epsilon=1e-9,
        space=LatticeSpace((1, 2)),
    )
    interactions = [
        TermanWangInteraction('oscillation', target='oscillators'),
        InhibitorDecayInteraction('decay', target='inhibitor'),
        InhibitorDriveInteraction('drive', inhibitor='inhibitor', oscillators='oscillators'),
    ]
    system = System(elements=[oscillators, InhibitorElement('inhibitor')], interactions=interactions)
    trajectory = system.prepare().simulate(dt=0.01, until=0.1)
    assert trajectory.get_values('inhibitor.z')[-1] == pytest.approx(1 - np.exp(-0.3), rel=1e-3)
