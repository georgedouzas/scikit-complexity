"""Test the `_community` module of `skcomplex.systems.biology.common.metabolism`."""

import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.systems import System
from skcomplex.systems.biology.common.metabolism import (
    FluxBalanceInteraction,
    ReactionSpec,
    community_growth,
    load_network,
)

AEROBIC = 0.8739215


def _get_community(networks):
    """Return the system of some networks, each with its cell."""
    cells = [Element(network.cell, values=network.get_values()) for network in networks]
    return System(elements=cells, interactions=networks)


def _get_members(reactions):
    """Return the system of two members with the same reactions, which grow by `grow`."""
    return _get_community(
        [
            FluxBalanceInteraction(f'fba_{name}', cell=name, reactions=reactions, objective={'grow': 1.0})
            for name in ('one', 'two')
        ],
    )


def _get_core_members():
    """Return the system of two members of the E. coli core system."""
    return _get_community([load_network('textbook', name, label=f'fba_{name}') for name in ('first', 'second')])


def test_community_steadycom_of_identical_members():
    """Test that identical members share the common rate 0.8739 with abundances that sum to 1."""
    analysis = community_growth(_get_core_members(), medium={'EX_glc__D_e': -20.0})
    assert analysis.growth == pytest.approx(AEROBIC, abs=1e-4)
    assert sum(analysis.abundances.values()) == pytest.approx(1.0)


@pytest.mark.parametrize('rate', [7.0, 50.0])
def test_community_steadycom_finds_the_rate_at_any_scale(rate):
    """Test a rate that failed at the boundary of the bisection, and one that a limit of 10 would clip."""
    reactions = {'EX_a': ReactionSpec({'a': 1.0}, upper=rate), 'grow': ReactionSpec({'a': -1.0})}
    analysis = community_growth(_get_members(reactions))
    assert analysis.growth == pytest.approx(rate, rel=1e-6)
    assert sum(analysis.abundances.values()) == pytest.approx(1.0)


def test_community_tradeoff_at_full_growth():
    """Test that the cooperative trade-off at fraction 1 reaches the optimum of the community."""
    analysis = community_growth(_get_core_members(), kind='tradeoff', fraction=1.0)
    assert analysis.growth == pytest.approx(AEROBIC, abs=1e-5)
    assert set(analysis.members) == {'first', 'second'}


def test_community_steadycom_keeps_its_reference_values():
    """Test the common rate against the value recorded before the element moved to declared states."""
    reactions = {'EX_a': ReactionSpec({'a': 1.0}, upper=7.0), 'grow': ReactionSpec({'a': -1.0})}
    analysis = community_growth(_get_members(reactions))
    assert analysis.growth == pytest.approx(6.99999975040555, rel=1e-9)
