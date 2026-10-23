"""Test the `_groups` module of `skcomplex.models.computer_science.automata`."""

from skcomplex.core.formalisms.groups import GroupSpace, automaton_group, parse_word
from skcomplex.core.space import Memory
from skcomplex.models.computer_science.automata import (
    grigorchuk_group,
    heisenberg,
)

from .conftest import GRIGORCHUK_BALLS


def _site_of(space, word):
    """Return the site reached from the identity along the letters of a word."""
    site = 0
    for letter in parse_word(word, space.group.generators):
        site = int(space.generator_table[site, 2 * (abs(letter) - 1) + (letter < 0)])
    return site


def test_grigorchuk_group_is_an_automaton_group():
    """Test that the Grigorchuk group of the library is an automaton group of the core with generators a to d."""
    group = grigorchuk_group()
    recursion = {
        'a': ((None, None), (1, 0)),
        'b': (('a', 'c'), None),
        'c': (('a', 'd'), None),
        'd': ((None, 'b'), None),
    }
    assert group == automaton_group(recursion)
    assert group.generators == ('a', 'b', 'c', 'd')


def test_grigorchuk_growth():
    """Test that the balls of radii 0 to 5 hold 1 to 68 sites, all distinct elements as their level-12 actions show."""
    ball = GroupSpace(grigorchuk_group(), ball=5)
    assert ball.ball_sizes() == GRIGORCHUK_BALLS[:6]
    action = grigorchuk_group().level_quotient(12).word_problem()
    keys = set()
    for word in ball.words:
        element = action.identity()
        for letter in parse_word(word, ball.group.generators):
            element = action.multiply(element, letter)
        keys.add(action.key(element))
    assert len(keys) == ball.size


def test_grigorchuk_level_quotients_run_periodic_rules():
    """Test that the quotients of levels 1 to 4 hold 2, 8, 128, and 4096 sites and every cell reads inside them."""
    sizes = []
    for level in (1, 2, 3, 4):
        space = GroupSpace(grigorchuk_group(), quotient=level)
        sizes.append(space.size)
        assert (space.index_table(Memory(('1', 'a', 'b', 'c', 'd'))) < space.size).all()
    assert sizes == [2, 8, 128, 4096]


def test_heisenberg_commutator_is_central():
    """Test that the commutator of the Heisenberg generators commutes with both generators."""
    space = GroupSpace(heisenberg(), ball=6)
    commutator = 'a*b*a^-1*b^-1'
    for generator in ('a', 'b'):
        assert _site_of(space, f'{commutator}*{generator}') == _site_of(space, f'{generator}*{commutator}')
