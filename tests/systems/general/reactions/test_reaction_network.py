"""Test the `_reaction_network` module of `skcomplex.systems.general.reactions`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.general.reactions import ReactionNetworkSystem

DIMERIZATION = {'binding': ({'A': 2}, {'D': 1}, 0.01), 'unbinding': ({'D': 1}, {'A': 2}, 0.1)}


def test_reaction_network_builds_an_element_per_species_and_a_reaction_per_row():
    """Test that every species is an element and every row a reaction between their states `x`."""
    system = ReactionNetworkSystem(species={'A': 100.0, 'D': 0.0}, reactions=DIMERIZATION).prepare().system_
    assert [element.label for element in system.elements] == ['A', 'D']
    binding = system.interactions[0]
    assert binding.label == 'binding'
    assert binding.reactants == {('A', 'x'): 2}
    assert binding.products == {('D', 'x'): 1}
    assert binding.k == pytest.approx(0.01)


def test_reaction_network_rate_equations_reach_the_equilibrium_of_product_propensities():
    """Test that the dimerization `2 A <-> D` settles where `k_binding A^2 = k_unbinding D` and conserves `A + 2 D`."""
    trajectory = simulate(
        ReactionNetworkSystem(species={'A': 100.0, 'D': 0.0}, reactions=DIMERIZATION),
        until=200.0,
        dt=1.0,
    )
    monomers, dimers = trajectory.get_values('A.x'), trajectory.get_values('D.x')
    np.testing.assert_allclose(monomers + 2 * dimers, 100.0, rtol=1e-6)
    assert 0.01 * monomers[-1] ** 2 == pytest.approx(0.1 * dimers[-1], rel=1e-3)


def test_reaction_network_takes_sources_and_sinks_as_empty_sides():
    """Test that a reaction without reactants is a source and one without products a sink."""
    reactions = {'birth': ({}, {'X': 1}, 2.0), 'death': ({'X': 1}, {}, 0.5)}
    trajectory = simulate(ReactionNetworkSystem(species={'X': 0.0}, reactions=reactions), until=40.0, dt=1.0)
    assert trajectory.get_values('X.x')[-1] == pytest.approx(4.0, rel=1e-3)


def test_reaction_network_jumps_move_whole_units():
    """Test that the mode `'jump'` gives integer amounts of a stochastic system that conserve the total."""
    system = ReactionNetworkSystem(species={'A': 100.0, 'D': 0.0}, reactions=DIMERIZATION, mode='jump')
    assert system.get_tags().randomness == 'stochastic'
    states = simulate(system, until=20.0, dt=1.0, random_state=0).states
    np.testing.assert_array_equal(states, np.round(states))
    np.testing.assert_allclose(states @ np.array([1.0, 2.0]), 100.0)


def test_reaction_network_rebuilds_its_reactions_from_changed_parameters():
    """Test that `set_params` changes the reactions of the next `prepare`."""
    system = ReactionNetworkSystem(species={'A': 1.0}, reactions={'decay': ({'A': 1}, {}, 1.0)}).prepare()
    system.set_params(reactions={'decay': ({'A': 1}, {}, 3.0)}).prepare()
    assert system.system_.interactions[0].k == pytest.approx(3.0)


class DecaySystem(ReactionNetworkSystem):
    """A decay `X -> 0` that gives its species, reactions, and mode from parameters of its own names."""

    def __init__(self, amount=5.0, *, rate=0.5, stochastic=False, label='system'):
        super().__init__(label=label)
        self.amount = amount
        self.rate = rate
        self.stochastic = stochastic

    def _get_species(self):
        return {'X': self.amount}

    def _get_reactions(self):
        return {'decay': ({'X': 1}, {}, self.rate)}

    def _get_mode(self):
        return 'jump' if self.stochastic else 'rate'


def test_reaction_network_subclass_gives_its_components_from_hooks():
    """Test that a subclass with parameters of its own names builds its components from the hooks."""
    system = DecaySystem(rate=0.25, stochastic=True)
    assert system.get_params() == {'amount': 5.0, 'label': 'system', 'rate': 0.25, 'stochastic': True}
    reaction = system.prepare().system_.interactions[0]
    assert (reaction.k, reaction.mode) == (0.25, 'jump')
