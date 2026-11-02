"""Test the `word_element` module of `skcomplex.models.mathematics.fractals`."""

import pytest

from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.models.mathematics.fractals import (
    GeometrySpec,
    GeometryVisual,
    ProductionLaw,
    RewritingCompiler,
    RewritingError,
    WordElement,
)


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
        word.check()


def test_word_is_drawn_as_its_geometry():
    """Test that a prepared word is drawn as the geometry of its letters."""
    assert AlgaeElement('algae', axiom='A').check().prepare().default_visual_.kind is GeometryVisual


def test_word_is_rewritten_in_parallel():
    """Test that a word is rewritten by its productions at every step, its size growing as the Fibonacci numbers."""
    trajectory = (
        Simulation(System(elements=[AlgaeElement('algae', axiom='A')]), compiler=RewritingCompiler())
        .check()
        .prepare()
        .run(until=4)
        .trajectory_
    )
    assert trajectory.values('algae.size').tolist() == [1.0, 2.0, 3.0, 5.0, 8.0]
