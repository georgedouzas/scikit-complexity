"""Test the `_flux_balance` module of `skcomplex.systems.biology.common.metabolism`."""

import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.biology.common.metabolism import FluxBalanceInteraction, ReactionSpec
from skcomplex.systems.biology.systems_biology import FluxBalanceSystem, MetabolicPerturbationsSystem


def _get_growth(network, **values):
    """Return the greatest weighted flux of a network whose cell starts with some values replaced."""
    cell = Element(network.cell, values=network.get_values() | values)
    trajectory = System(elements=[cell], interactions=[network]).prepare().simulate(until=0.0)
    return -float(trajectory.get_values(f'{network.cell}.objective_0')[0])


def test_network_rule_gates_the_bounds_of_a_reaction():
    """Test that a rule closes its reaction only when it evaluates to false."""
    reactions = {
        'in': ReactionSpec({'A': 1.0}, upper=5.0),
        'use': ReactionSpec({'A': -1.0}, rule='g1 and (g2 or g3)'),
    }
    network = FluxBalanceInteraction('fba', cell='cell', reactions=reactions, objective={'use': 1.0})
    values = [(1.0, 0.0, 1.0, 5.0), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 1.0, 0.0)]
    for first, second, third, expected in values:
        assert _get_growth(network, g_g1=first, g_g2=second, g_g3=third) == pytest.approx(expected)


def test_network_rejects_a_rule_with_another_operator():
    """Test that a rule with an operator other than `and` and `or` raises an error that names the interaction."""
    network = FluxBalanceInteraction('fba', cell='cell', reactions={'r': ReactionSpec({'A': 1.0}, rule='a and not b')})
    with pytest.raises(ValidationError, match='gene rule of fba'):
        _get_growth(network)


def test_network_without_an_objective_holds_its_balances_and_bounds():
    """Test that a network with no objective takes any fluxes its balances and bounds allow."""
    reactions = {'in': ReactionSpec({'A': 1.0}, lower=2.0, upper=2.0), 'use': ReactionSpec({'A': -1.0})}
    network = FluxBalanceInteraction('fba', cell='cell', reactions=reactions)
    cell = Element('cell', values=network.get_values())
    trajectory = System(elements=[cell], interactions=[network]).prepare().simulate(until=0.0)
    assert float(trajectory.get_values('cell.v_use')[0]) == pytest.approx(2.0)


def test_parsimonious_network_takes_the_smallest_total_flux_at_its_optimum():
    """Test that a parsimonious network takes the shorter of two routes of equal growth."""
    reactions = {
        'in': ReactionSpec({'A': 1.0}, upper=4.0),
        'short': ReactionSpec({'A': -1.0, 'B': 1.0}),
        'first': ReactionSpec({'A': -1.0, 'C': 1.0}),
        'second': ReactionSpec({'C': -1.0, 'B': 1.0}),
        'grow': ReactionSpec({'B': -1.0}),
    }
    network = FluxBalanceInteraction('fba', reactions=reactions, objective={'grow': 1.0}, parsimonious=True)
    cell = Element('cell', values=network.get_values())
    trajectory = System(elements=[cell], interactions=[network]).prepare().simulate(until=0.0)
    assert float(trajectory.get_values('cell.v_short')[0]) == pytest.approx(4.0)
    assert float(trajectory.get_values('cell.v_first')[0]) == pytest.approx(0.0)
    assert float(trajectory.get_values('cell.objective_1')[0]) == pytest.approx(12.0)


def test_network_keeps_its_reference_values():
    """Test the toy and core networks against the values recorded before they moved to declared states."""
    trajectory = FluxBalanceSystem().prepare().simulate(until=0.0)
    assert float(trajectory.get_values('cell.v_vbio')[0]) == pytest.approx(50.0, rel=1e-9)
    knocked = MetabolicPerturbationsSystem(genes=('b4025',)).prepare().simulate(until=0.0)
    assert -float(knocked.get_values('cell.objective_0')[0]) == pytest.approx(0.8631595522084183, rel=1e-9)
