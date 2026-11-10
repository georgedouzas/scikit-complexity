"""Test the `_boolean_systems` module of `skcomplex.systems.biology.systems_biology`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.systems_biology import (
    BooleanRepressilatorSystem,
    MammalianCellCycleSystem,
    NoisyBooleanNetworkSystem,
    ThreeGeneBooleanNetworkSystem,
)

from ..conftest import _run_with_solver

CELL_CYCLE = ('CycD', 'Rb', 'E2F', 'CycE', 'CycA', 'p27', 'Cdc20', 'Cdh1', 'Ubc', 'CycB')


def _states(trajectory, names):
    """Return the Boolean states of a network as strings of bits, one per step."""
    values = np.array([trajectory.get_values(f'{name}.x') for name in names]).T.astype(int)
    return [''.join(str(bit) for bit in row) for row in values]


def _period(states):
    """Return the period of the cycle that a sequence of states ends on."""
    last = states[-1]
    return next(lag for lag in range(1, len(states)) if states[-1 - lag] == last)


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
        trajectory = _run_with_solver(
            ThreeGeneBooleanNetworkSystem(tuple(int(bit) for bit in state)),
            until=1,
            dt=1,
        )
        assert _states(trajectory, ('g1', 'g2', 'g3')) == [state, successor]
    trajectory = _run_with_solver(ThreeGeneBooleanNetworkSystem((1, 0, 0)), until=10, dt=1)
    assert _period(_states(trajectory, ('g1', 'g2', 'g3'))) == 3


def test_mammalian_cell_cycle_rests_or_cycles_with_the_growth_signal():
    """Test that the quiescent state is fixed without CycD and that CycD drives a cycle of period 7."""
    quiescent = _run_with_solver(MammalianCellCycleSystem(('Rb', 'p27', 'Cdh1')), until=5, dt=1)
    assert set(_states(quiescent, CELL_CYCLE)) == {'0100010100'}
    cycling = _run_with_solver(MammalianCellCycleSystem(), until=30, dt=1)
    assert _period(_states(cycling, CELL_CYCLE)) == 7
    assert cycling.get_values('CycB.x')[10:].max() == 1


def test_mammalian_cell_cycle_knockout_stays_off():
    """Test that a knocked-out cyclin stays off and that the cycle stops without CycE and CycA."""
    trajectory = _run_with_solver(MammalianCellCycleSystem(knockouts=('CycE', 'CycA')), until=30, dt=1)
    assert trajectory.get_values('CycA.x').max() == 0
    assert _period(_states(trajectory, CELL_CYCLE)) == 1


@pytest.mark.parametrize('delay', [0, 1, 2])
def test_boolean_repressilator_oscillates_with_delay(delay):
    """Test that the repressilator, with its relays, oscillates with period `6 (delay + 1)`."""
    system = BooleanRepressilatorSystem((1, 0, 0), delay=delay).prepare().system_
    trajectory = _run_with_solver(system, until=60, dt=1)
    assert _period(_states(trajectory, [node.label for node in system.elements])) == 6 * (delay + 1)


def _order(eta: float) -> float:
    """Return the time average of the absolute mean state over the last half of a run."""
    trajectory = simulate(NoisyBooleanNetworkSystem(eta=eta), random_state=0, until=200)
    return float(np.abs(trajectory.get_values('nodes.sigma').mean(axis=1)[100:]).mean())


def test_order_vanishes_above_the_mean_field_critical_noise():
    """Test Huepe and Aldana-González: order below `eta_c = 1/6` for three inputs, disorder above it."""
    assert _order(0.05) > 0.8
    assert _order(0.12) > 0.4
    assert _order(0.25) < 0.1


def test_even_number_of_inputs_raises():
    """Test that an even number of inputs, which leaves ties, raises an error."""
    with pytest.raises(ValidationError):
        NoisyBooleanNetworkSystem(k=2).prepare()
