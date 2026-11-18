"""Test the `_rewriting_interaction` module of `skcomplex.systems.general.rewriting`."""

from types import SimpleNamespace

import pytest

from skcomplex.core.components.laws import RateLaw
from skcomplex.core.errors import RewritingError
from skcomplex.core.systems import System
from skcomplex.systems.general.rewriting import ProductionLaw, RewritingInteraction, WordElement


def test_prepare_sets_the_productions_for_the_word():
    """Test that the interaction prepared on the namespace of a word sets the productions it is given."""
    laws = [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]
    interaction = RewritingInteraction('growth', target='algae', laws=laws).prepare(SimpleNamespace(target=None))
    assert interaction.laws_ == laws


def test_prepare_rejects_a_law_that_rewrites_nothing():
    """Test that a law that is neither a production nor a substitution is rejected."""
    with pytest.raises(RewritingError, match='ProductionLaw or SubstitutionLaw'):
        RewritingInteraction('growth', target='algae', laws=[RateLaw(1, 1)]).prepare()


def test_simulate_rewrites_two_words_by_their_own_interactions():
    """Test that each word is rewritten only by the interaction that binds it."""
    system = System(
        elements=[WordElement('algae', axiom='A'), WordElement('morse', axiom='0')],
        interactions=[
            RewritingInteraction('growth', target='algae', laws=[ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]),
            RewritingInteraction('doubling', target='morse', laws=[ProductionLaw('0', '01'), ProductionLaw('1', '10')]),
        ],
    )
    trajectory = system.prepare().simulate(until=3)
    assert trajectory.get_structure('algae', 3).text == 'ABAAB'
    assert trajectory.get_structure('morse', 3).text == '01101001'
