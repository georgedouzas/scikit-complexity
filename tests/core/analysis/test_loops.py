"""Test the feedback loops of the signed dependency graph of a system, their dominance, and the flow matrix."""

import numpy as np
import pytest

from skcomplex.core.analysis import LoopAnalysis
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import AnalysisError
from skcomplex.systems.biology.ecology import generalized_lotka_volterra, lotka_volterra
from skcomplex.systems.biology.epidemiology import sir
from skcomplex.systems.biology.systems_biology import repressilator
from skcomplex.systems.economics.innovation import bass_diffusion
from skcomplex.systems.mathematics.dynamical_systems import logistic_map
from skcomplex.systems.physics.mechanics import harmonic_oscillator


def _loops(source, **settings):
    """Return the loop analysis of a source, run."""
    return LoopAnalysis(**settings).prepare(source).run()


def _found(analysis):
    """Return the polarity of every loop by its nodes."""
    return dict(zip(analysis.loops_, analysis.polarities_, strict=True))


def test_stock_and_flow_loops_pass_through_their_flows():
    """Test that adoption closes a reinforcing loop through adopters and a balancing one through potential adopters."""
    analysis = _loops(bass_diffusion())
    assert _found(analysis) == {
        ('market.A', 'market.adoption'): 'reinforcing',
        ('market.P', 'market.adoption'): 'balancing',
    }
    assert analysis.graph_.edges['market.adoption', 'market.P']['sign'] == -1
    assert analysis.graph_.edges['market.adoption', 'market.A']['sign'] == 1


def test_lotka_volterra_has_one_balancing_loop_through_predation():
    """Test that predation closes a balancing loop, prey growth a reinforcing self-loop, and predator death a balancing.

    At one prey and one predator, the rate of the prey grows with the prey by `r - a = 0.7`, the rate of the predators
    falls with the predators by `b - d = -0.3`, and the loop through predation has the gain `(-a) * b = -0.04`.
    """
    analysis = _loops(lotka_volterra(), at=[1.0, 1.0])
    assert _found(analysis) == {
        ('predators.n',): 'balancing',
        ('prey.n',): 'reinforcing',
        ('predators.n', 'prey.n'): 'balancing',
    }
    assert analysis.gains_ == pytest.approx([-0.3, 0.7, -0.04])


def test_sir_infection_reinforces_and_depletion_balances():
    """Test that infection above its threshold is a reinforcing self-loop, balanced through the susceptibles."""
    found = _found(_loops(sir()))
    assert found == {('I.n',): 'reinforcing', ('S.n',): 'balancing', ('I.n', 'S.n'): 'balancing'}


def test_repressilator_closes_one_balancing_loop_of_three_repressions():
    """Test that the cycle of three repressions through the six gene products is a balancing loop."""
    analysis = _loops(repressilator(), at=np.ones(6))
    cycle = ('gene_0.m', 'gene_0.p', 'gene_1.m', 'gene_1.p', 'gene_2.m', 'gene_2.p')
    assert _found(analysis)[cycle] == 'balancing'
    assert sum(len(nodes) == 1 for nodes in analysis.loops_) == 6


def test_states_of_second_order_link_through_their_velocity():
    """Test that the position and velocity of an oscillator close one balancing loop with the gain `-k / m`."""
    analysis = _loops(harmonic_oscillator())
    assert _found(analysis) == {('mass.x[0]', 'mass.x_d1[0]'): 'balancing'}
    assert analysis.gains_ == pytest.approx([-1.0])


def test_map_loops_follow_the_change_in_one_step():
    """Test that the self-loop of the logistic map has the gain of its change in one step, `r (1 - 2 x) - 1`."""
    analysis = _loops(logistic_map(), at=[0.1])
    assert analysis.loops_ == [('map.x',)]
    assert analysis.gains_ == pytest.approx([3.7 * 0.8 - 1])
    assert analysis.polarities_ == ['reinforcing']


def test_members_of_a_group_are_nodes_with_the_signs_of_the_jacobian():
    """Test that the members of a community are nodes linked by the entries of the Jacobian `x_i a_ij`.

    At abundances of one, the rate of the first species falls with itself by `1 - 2 - 0.5 = -1.5`, that of the second
    by `1 + 0.5 - 2 = -0.5`, and the second harms the first while the first feeds the second, a balancing loop of gain
    `0.5 * (-0.5) = -0.25`.
    """
    system = generalized_lotka_volterra([1.0, 1.0], b=[1.0, 1.0], a=[[-1.0, -0.5], [0.5, -1.0]])
    analysis = _loops(system)
    assert _found(analysis) == {
        ('species.x[0]',): 'balancing',
        ('species.x[1]',): 'balancing',
        ('species.x[0]', 'species.x[1]'): 'balancing',
    }
    assert analysis.gains_ == pytest.approx([-1.5, -0.5, -0.25])


def test_dominance_follows_the_gains_of_any_system():
    """Test that the gains of an oscillator stay `-k / m` and those of predation start at the gains of its loops."""
    trajectory = Simulation(harmonic_oscillator(), dt=0.1).prepare().run(1).trajectory_
    dominance = _loops(trajectory, kind='dominance')
    assert dominance.gains_ == pytest.approx(np.full((len(trajectory.times), 1), -1.0))
    trajectory = Simulation(lotka_volterra(), dt=0.1).prepare().run(1).trajectory_
    dominance = _loops(trajectory, kind='dominance')
    assert dominance.gains_[0] == pytest.approx(_loops(lotka_volterra()).gains_)
    assert dominance.polarities_[int(dominance.dominant_[0])] == 'balancing'


def test_flow_matrix_needs_flows():
    """Test that the flow matrix of a system without flows raises an error that names the Jacobian instead."""
    trajectory = Simulation(lotka_volterra(), dt=0.1).prepare().run(1).trajectory_
    with pytest.raises(AnalysisError, match='declares no flows'):
        _loops(trajectory, kind='flow-matrix')
