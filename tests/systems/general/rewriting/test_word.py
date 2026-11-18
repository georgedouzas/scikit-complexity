"""Test the `_word` module of `skcomplex.systems.general.rewriting`."""

import pytest

from skcomplex.core.errors import RewritingError
from skcomplex.core.systems import System
from skcomplex.systems.general.rewriting import (
    GeometrySpec,
    GeometryVisual,
    ProductionLaw,
    RewritingInteraction,
    WordElement,
)


@pytest.mark.parametrize(
    'word',
    [
        WordElement('algae', max_size=0),
        WordElement('algae', max_size=True),
        WordElement('algae', geometry=GeometrySpec(dimension=4)),
    ],
)
def test_word_checks_its_maximum_size_and_geometry(word):
    """Test that a word rejects a maximum size that is not a positive integer and an invalid geometry."""
    with pytest.raises(RewritingError):
        word.prepare()


def test_word_is_drawn_as_its_geometry():
    """Test that a prepared word is drawn as the geometry of its letters."""
    assert WordElement('algae', axiom='A').prepare().default_visual_.kind is GeometryVisual


def test_word_is_rewritten_in_parallel():
    """Test that a word is rewritten by its productions at every step, its size growing as the Fibonacci numbers."""
    growth = RewritingInteraction('growth', target='algae', laws=[ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')])
    system = System(elements=[WordElement('algae', axiom='A')], interactions=[growth])
    trajectory = system.prepare().simulate(until=4)
    assert trajectory.get_values('algae.size').tolist() == [1.0, 2.0, 3.0, 5.0, 8.0]
