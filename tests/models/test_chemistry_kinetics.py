"""Test the chemical kinetics models against their known behavior."""

import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.chemistry.kinetics import (
    brusselator_ode,
    catalyst_pellet,
    chemical_langevin_equation,
    cubic_autocatalysis,
    cyclic_cellular_automaton,
    michaelis_menten,
    oregonator,
    pulsed_electrochemical_reactor,
)


def _run(system, until, dt, **settings):
    """Return the trajectory of a system."""
    return Simulation(system, dt=dt, random_state=0, **settings).run(until=until).trajectory_


def _upward_crossings(times, values, level):
    """Return the interpolated times where a series crosses a level upwards."""
    below, above = values[:-1], values[1:]
    rising = np.flatnonzero((below < level) & (above >= level))
    fractions = (level - below[rising]) / (above[rising] - below[rising])
    return times[rising] + fractions * (times[rising + 1] - times[rising])


def test_oregonator_oscillates():
    """Test that the Oregonator with its standard parameters settles on a relaxation limit cycle of stable period."""
    solver = describe('oregonator').defaults['solver']
    trajectory = _run(oregonator(), until=25.0, dt=0.02, solver=solver)
    late = trajectory.times > 5.0
    x = trajectory['reactor.x'][late]
    periods = np.diff(_upward_crossings(trajectory.times[late], x, 0.1))
    assert len(periods) >= 2
    assert np.ptp(periods) < 0.01 * periods.mean()
    assert x.max() / x.min() > 100


def test_brusselator_ode_has_hopf_bifurcation():
    """Test that the Brusselator rests at `(a, b / a)` below `b = 1 + a**2` and oscillates above it."""
    resting = _run(brusselator_ode(b=1.5), until=60.0, dt=0.5)
    assert np.allclose([resting['reactor.u'][-1], resting['reactor.v'][-1]], [1.0, 1.5], atol=1e-3)
    oscillating = _run(brusselator_ode(b=3.0), until=60.0, dt=0.1)
    assert np.ptp(oscillating['reactor.u'][oscillating.times > 30.0]) > 1.0


def test_cubic_autocatalysis_is_bistable():
    """Test that cubic autocatalysis settles on the washout or on the reacting state depending on its start."""
    washout = _run(cubic_autocatalysis(a=1.0, b=0.05), until=500.0, dt=5.0)
    reacting = _run(cubic_autocatalysis(a=0.5, b=0.5), until=500.0, dt=5.0)
    assert washout['reactor.b'][-1] < 1e-3
    assert np.isclose(reacting['reactor.a'][-1], (1 - np.sqrt(1 - 4 * 0.09**2 / 0.04)) / 2, atol=1e-3)


def test_michaelis_menten_conserves_enzyme_and_substrate():
    """Test that the enzyme reaction conserves the total enzyme and substrate as rate equations and as jumps."""
    for stochastic in (False, True):
        trajectory = _run(michaelis_menten(stochastic=stochastic), until=20.0, dt=1.0)
        enzyme = trajectory['E.n'] + trajectory['C.n']
        substrate = trajectory['S.n'] + trajectory['C.n'] + trajectory['P.n']
        assert np.allclose(enzyme, 100.0)
        assert np.allclose(substrate, 300.0)
        assert trajectory['P.n'][-1] > 100.0
    assert np.array_equal(trajectory['P.n'], np.round(trajectory['P.n']))


def test_chemical_langevin_equation_keeps_both_states():
    """Test that the Schlögl reaction fluctuates around its low or its high state, depending on its start."""
    low = _run(
        chemical_langevin_equation(x=85.0),
        until=1.0,
        dt=0.05,
        solver=describe('chemical_langevin_equation').defaults['solver'],
    )
    high = _run(
        chemical_langevin_equation(x=570.0),
        until=1.0,
        dt=0.05,
        solver=describe('chemical_langevin_equation').defaults['solver'],
    )
    assert abs(low['reactor.x'].mean() - 85.0) < 30.0
    assert abs(high['reactor.x'].mean() - 570.0) < 60.0
    assert low['reactor.x'].std() > 1.0


def test_catalyst_pellet_is_bistable():
    """Test that a catalyst pellet settles on a reacting or an inhibited state depending on its start."""
    reacting = _run(catalyst_pellet(c=0.0), until=200.0, dt=10.0)
    inhibited = _run(catalyst_pellet(c=1.0), until=200.0, dt=10.0)
    for trajectory in (reacting, inhibited):
        c = trajectory['pellet.c'][-1]
        assert abs(0.1 * (1 - c) - 10.0 * c / (1 + 20.0 * c) ** 2) < 1e-6
    assert reacting['pellet.c'][-1] < 0.05
    assert inhibited['pellet.c'][-1] > 0.5


def test_pulsed_electrochemical_reactor_follows_the_pulses():
    """Test that every pulse raises the overpotential and depletes the reactant, and the response becomes periodic."""
    trajectory = _run(pulsed_electrochemical_reactor(), until=100.0, dt=0.5)
    times, eta, c = trajectory.times, trajectory['electrode.eta'], trajectory['electrode.c']
    starts, ends, recovered = (np.isin(times, 10.0 * np.arange(6) + phase) for phase in (0.0, 2.0, 9.5))
    assert (eta[ends] > eta[starts]).all()
    assert (c[ends] < c[starts]).all()
    assert (c[recovered] > c[ends]).all()
    assert np.isclose(c[times == 99.5][0], c[times == 89.5][0], atol=1e-3)


def test_cyclic_cellular_automaton_advances_around_the_cycle():
    """Test that every cell of the cyclic automaton stays or moves to the following level."""
    trajectory = _run(cyclic_cellular_automaton((20, 20), n=8), until=30, dt=1)
    levels = trajectory.lattice('cells.level')
    steps = np.mod(levels[1:] - levels[:-1], 8)
    assert set(np.unique(steps)) <= {0.0, 1.0}
    assert steps[-1].sum() > 0
