"""Test the `_networks` module of `skcomplex.systems.biology.neuroscience`."""

import math

import numpy as np
import pytest
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import laplacian, logistic
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation, simulate
from skcomplex.systems.biology.neuroscience import (
    DelayedLIFNetworkSystem,
    InhibitorDriveInteraction,
    InhibitorElement,
    LegionSystem,
    TermanWangElement,
)


class _PreviousOscillatorElement(TermanWangElement):
    """Oscillator with the excitation by its neighbours in its own law, as before the coupling interaction."""

    def _get_laws(self, s):
        """Return the fast variable with its excitation by the neighbours, and the slow recovery."""
        active = s.m * logistic(s.x - s.theta_x, 0, s.kappa)
        excitation = s.W * s.m * (laplacian(active) + 4 * active)
        return [
            RateLaw(s.x, 3 * s.x - s.x**3 + 2 - s.y + s.I + excitation),
            RateLaw(s.y, s.epsilon * (s.gamma * (1 + tanh(s.x / s.beta)) - s.y)),
        ]


def test_delayed_lif_network_rates():
    """Test that uncoupled neurons fire at the leaky integrate-and-fire period and that inhibition slows them."""
    period = 20.0 * math.log((25.0 - 10.0) / (25.0 - 20.0))
    counts = []
    for J in (0.0, -2.0):
        simulation = (
            Simulation(DelayedLIFNetworkSystem(5, J=J, probability=0.5).prepare().system_, dt=1.0)
            .prepare()
            .run(until=60)
        )
        events = simulation.trajectory_.events
        counts.append(len(events))
        if J == 0.0:
            index = list(simulation.problem_.states.names).index('neuron_0.v')
            times = [event.time for event in events if event.before[index] != event.after[index]]
            assert np.diff(times) == pytest.approx(period, rel=1e-3)
    assert counts[1] < counts[0]


def test_legion_segments_objects():
    """Test that LEGION synchronizes each object, alternates between objects, and keeps the background silent."""
    trajectory = simulate(LegionSystem(epsilon=0.1), dt=0.5, until=120)
    active = trajectory.values('oscillators.x').reshape(len(trajectory.times), -1) > 0
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
    trajectory = simulate(LegionSystem(), dt=0.5, until=100)
    active = trajectory.values('oscillators.x').reshape(len(trajectory.times), -1) > 0
    for sites in ([0, 1, 4], [11, 15]):
        on = active[:, sites].any(axis=1)
        assert (on[1:] & ~on[:-1]).any()


def _previous_legion():
    """Return `LegionSystem` with its coupling written in the element, as `laplacian(active) + 4 * active`."""
    system = LegionSystem().prepare().system_
    field, inhibitor = system.elements
    oscillators = FieldElement(
        'oscillators',
        element=_PreviousOscillatorElement,
        space=field.space,
        boundary=field.boundary,
        values=field.values,
    )
    return System(elements=[oscillators, inhibitor], interactions=system.interactions[1:])


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
    trajectory = simulate(LegionSystem(), dt=0.5, until=1.5)
    expected = simulate(_previous_legion(), dt=0.5, until=1.5)
    assert np.abs(trajectory.values('oscillators.x') - expected.values('oscillators.x')).max() < 1e-10
    assert np.abs(trajectory.values('oscillators.y') - expected.values('oscillators.y')).max() < 1e-10


def test_inhibitor_drive_interaction_follows_most_active_oscillator():
    """Test that the inhibitor relaxes toward the largest activity of the oscillators, 1 when one is active."""
    values = {'x': np.array([[2.0, -2.0]]), 'm': np.array([[1.0, 1.0]]), 'epsilon': 1e-9}
    oscillators = FieldElement('oscillators', element=TermanWangElement, space=LatticeSpace((1, 2)), values=values)
    drive = InhibitorDriveInteraction('drive', inhibitor='inhibitor', oscillators='oscillators')
    system = System(elements=[oscillators, InhibitorElement('inhibitor')], interactions=[drive])
    trajectory = simulate(system, dt=0.01, until=0.1)
    assert trajectory.values('inhibitor.z')[-1] == pytest.approx(1 - np.exp(-0.3), rel=1e-3)
