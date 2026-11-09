"""Test the `_word` module of `skcomplex.core.components.elements`."""

import pytest

from skcomplex.core.components.elements import WordElement
from skcomplex.core.components.laws import ProductionLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import RewritingError
from skcomplex.core.specs import GeometrySpec


class AlgaeElement(WordElement):
    """Word of Lindenmayer's algae."""

    def _get_laws(self, s):
        """Return the productions of the algae."""
        return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]


@pytest.mark.parametrize(
    'word',
    [
        AlgaeElement('algae', max_size=0),
        AlgaeElement('algae', max_size=True),
        AlgaeElement('algae', geometry=GeometrySpec(dimension=4)),
    ],
)
def test_word_checks_its_maximum_size_and_geometry(word):
    """Test that a word rejects a maximum size that is not a positive integer and an invalid geometry."""
    with pytest.raises(RewritingError):
        word.prepare()


def test_word_is_drawn_as_its_geometry():
    """Test that a prepared word is drawn as the geometry of its letters."""
    assert AlgaeElement('algae', axiom='A').prepare().default_visual_.kind == 'geometry'


def test_word_is_rewritten_in_parallel():
    """Test that a word is rewritten by its productions at every step, its size growing as the Fibonacci numbers."""
    trajectory = simulate(CompositeSystem(elements=[AlgaeElement('algae', axiom='A')]), until=4)
    assert trajectory.values('algae.size').tolist() == [1.0, 2.0, 3.0, 5.0, 8.0]
