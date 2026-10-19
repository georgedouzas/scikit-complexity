"""Test the cellular network models: oscillators, power-law systems, Boolean networks, and microbial communities."""

import numpy as np
import pytest

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.models.biology.ecology import generalized_lotka_volterra
from skcomplex.models.biology.systems_biology import (
    biochemical_systems_theory,
    boolean_repressilator,
    goodwin_oscillator,
    mammalian_cell_cycle,
    phosphorylation_oscillator,
    three_gene_boolean_network,
)

PRECISE = ScipyODE(rtol=1e-9, atol=1e-11)
CELL_CYCLE = ('CycD', 'Rb', 'E2F', 'CycE', 'CycA', 'p27', 'Cdc20', 'Cdh1', 'Ubc', 'CycB')


def _run(system, until, dt, solver='auto'):
    """Return the trajectory of a system."""
    return Simulation(system, dt=dt, random_state=0, solver=solver).run(until=until).trajectory_


def _states(trajectory, names):
    """Return the Boolean states of a network as strings of bits, one per step."""
    values = np.array([trajectory[f'{name}.x'] for name in names]).T.astype(int)
    return [''.join(str(bit) for bit in row) for row in values]


def _period(states):
    """Return the period of the cycle that a sequence of states ends on."""
    last = states[-1]
    return next(lag for lag in range(1, len(states)) if states[-1 - lag] == last)


def test_goodwin_oscillator_switches_at_hill_coefficient_eight():
    """Test that three stages settle on the fixed point for a Hill coefficient of 6 and oscillate for 12."""
    amplitudes = {}
    for n in (6.0, 12.0):
        trajectory = _run(goodwin_oscillator(3, n=n), until=200.0, dt=0.1, solver=PRECISE)
        late = trajectory['x3.x'][trajectory.times >= 150.0]
        amplitudes[n] = late.max() - late.min()
    assert amplitudes[6.0] < 1e-4
    assert amplitudes[12.0] > 0.5


def test_phosphorylation_oscillator_cycles():
    """Test that the three switches turn on in turn and keep oscillating with a constant amplitude."""
    trajectory = _run(phosphorylation_oscillator(), until=40.0, dt=0.05, solver=PRECISE)
    times = trajectory.times
    for name in ('A', 'B', 'C'):
        late = trajectory[f'loop.{name}'][times >= 20.0]
        assert late.max() > 0.9
        assert late.min() < 0.1


def test_biochemical_systems_theory_reaches_log_linear_steady_state():
    """Test that both forms reach the steady state that solves `(g - h) log x = log(beta / alpha)`."""
    g = np.array([[0.0, 0.0, -0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    h = 0.5 * np.eye(3)
    expected = np.exp(np.linalg.solve(g - h, np.log(np.array([1.0, 1.0, 1.0]) / np.array([2.0, 1.0, 1.0]))))
    for form in ('s-system', 'gma'):
        trajectory = _run(biochemical_systems_theory(form=form), until=60.0, dt=0.5, solver=PRECISE)
        final = np.array([trajectory[f'X{index}.x'][-1] for index in (1, 2, 3)])
        assert np.allclose(final, expected, rtol=1e-4)


def test_three_gene_boolean_network_matches_the_transition_table():
    """Test that every state goes to the successor the book lists, so `011` is fixed and `001` starts a 3-cycle."""
    successors = {
        '000': '011',
        '001': '010',
        '010': '111',
        '011': '011',
        '100': '001',
        '101': '000',
        '110': '101',
        '111': '001',
    }
    for state, successor in successors.items():
        trajectory = _run(three_gene_boolean_network(tuple(int(bit) for bit in state)), until=1, dt=1)
        assert _states(trajectory, ('g1', 'g2', 'g3')) == [state, successor]
    trajectory = _run(three_gene_boolean_network((1, 0, 0)), until=10, dt=1)
    assert _period(_states(trajectory, ('g1', 'g2', 'g3'))) == 3


def test_mammalian_cell_cycle_rests_or_cycles_with_the_growth_signal():
    """Test that the quiescent state is fixed without CycD and that CycD drives a cycle of period 7."""
    quiescent = _run(mammalian_cell_cycle(('Rb', 'p27', 'Cdh1')), until=5, dt=1)
    assert set(_states(quiescent, CELL_CYCLE)) == {'0100010100'}
    cycling = _run(mammalian_cell_cycle(), until=30, dt=1)
    assert _period(_states(cycling, CELL_CYCLE)) == 7
    assert cycling['CycB.x'][10:].max() == 1


def test_mammalian_cell_cycle_knockout_stays_off():
    """Test that a knocked-out cyclin stays off and that the cycle stops without CycE and CycA."""
    trajectory = _run(mammalian_cell_cycle(knockouts=('CycE', 'CycA')), until=30, dt=1)
    assert trajectory['CycA.x'].max() == 0
    assert _period(_states(trajectory, CELL_CYCLE)) == 1


@pytest.mark.parametrize('delay', [0, 1, 2])
def test_boolean_repressilator_oscillates_with_delay(delay):
    """Test that the repressilator, with its relays, oscillates with period `6 (delay + 1)`."""
    system = boolean_repressilator((1, 0, 0), delay=delay)
    trajectory = _run(system, until=60, dt=1)
    assert _period(_states(trajectory, [node.label for node in system.elements])) == 6 * (delay + 1)


def test_generalized_lotka_volterra_reaches_coexistence():
    """Test that the community settles on the interior fixed point `x = -a^-1 b`."""
    b = np.array([1.0, 0.8, 0.6])
    a = np.array([[-1.0, -0.3, 0.2], [-0.2, -1.0, -0.3], [0.1, -0.2, -1.0]])
    trajectory = _run(generalized_lotka_volterra(), until=60.0, dt=0.5, solver=PRECISE)
    assert np.allclose(trajectory['species.x'][-1], -np.linalg.solve(a, b), rtol=1e-4)
