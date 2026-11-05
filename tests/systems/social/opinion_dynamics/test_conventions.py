"""Test the `_conventions` module of `skcomplex.systems.social.opinion_dynamics`."""

from skcomplex.systems.social.opinion_dynamics import (
    naming_game,
)

from ..conftest import _run, _values


def test_naming_game_agrees_on_one_word():
    """Test that the naming game ends with every agent knowing the same single word."""
    words = _values(_run(naming_game(30), 200), 'namers', 'words')
    assert words.sum(axis=1).tolist() == [1.0] * 30
    assert len(set(words.argmax(axis=1).tolist())) == 1
