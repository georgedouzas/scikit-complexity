"""Test the `_tags` module of `skcomplex.core.specs`."""

import pytest

from skcomplex.core.enums import DomainTag, LawTag, RandomnessTag, SpaceTag, StructureTag, TimeTag
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import Tags


def test_tags_reject_an_unknown_value():
    """Test that a tag outside the values of its field is refused, with the field and its values named."""
    with pytest.raises(ValidationError, match=r"The space tag 'torus' is unknown\. Use one of 'none'"):
        Tags(space=frozenset({'torus'}))._check()


def test_tags_accept_unknown_law_kinds():
    """Test that law kinds and randomness may be unknown."""
    tags = Tags(laws=None, randomness=None)
    tags._check()
    assert tags.laws is None
    assert tags.randomness is None


def test_missing_names_every_unsupported_field():
    """Test that supported tags name the spaces and law kinds of a system they do not support, and no time it shares."""
    supported = Tags(time=frozenset({'continuous'}), space=frozenset({'none'}), laws=frozenset({'rate'}))
    system = Tags(
        time=frozenset({'continuous', 'static'}),
        space=frozenset({'lattice'}),
        laws=frozenset({'rate', 'jump'}),
    )
    assert supported.get_missing(system) == {'space': ['lattice'], 'laws': ['jump']}


def test_missing_rejects_a_time_it_does_not_share():
    """Test that a system none of whose times is supported is reported."""
    supported = Tags(time=frozenset({'discrete'}))
    assert supported.get_missing(Tags(time=frozenset({'continuous'}))) == {'time': ['continuous']}


def test_empty_fields_support_anything_and_unknown_laws_are_not_checked():
    """Test that empty supported fields support any value, and that unknown law kinds of a system pass."""
    assert Tags().get_missing(Tags(space=frozenset({'network'}), laws=frozenset({'noise'}))) == {}
    assert Tags(laws=frozenset({'rate'})).get_missing(Tags(laws=None)) == {}


def test_tags_accept_every_value_of_the_public_vocabularies():
    """Test that the enums of the tags hold the values `Tags` accepts."""
    tags = Tags(
        time=frozenset(TimeTag),
        domains=frozenset(DomainTag),
        space=frozenset(SpaceTag),
        laws=frozenset(LawTag),
        randomness=RandomnessTag.STOCHASTIC,
        structure=frozenset(StructureTag),
    )
    tags._check()
    assert tags.time == {'continuous', 'discrete', 'static'}


def test_tags_differentiable_covers_reparameterized_draws():
    """Test that stochastic update laws are differentiable only when their draws are reparameterized."""
    stochastic = Tags(laws=frozenset({'update'}), randomness='stochastic')
    assert not stochastic.differentiable
    assert Tags(laws=frozenset({'update'}), randomness='stochastic', reparameterized=True).differentiable
    assert not Tags(laws=frozenset({'noise'}), randomness='stochastic', reparameterized=True).differentiable
