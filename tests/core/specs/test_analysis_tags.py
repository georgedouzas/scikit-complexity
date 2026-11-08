"""Test the `_analysis_tags` module of `skcomplex.core.specs`."""

import pytest

from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import AnalysisTags, Tags


def test_analysis_tags_reject_an_unknown_value():
    """Test that a data kind or a tag outside the values of its field is refused, with the field named."""
    with pytest.raises(ValidationError, match=r"The data tag 'table' is unknown"):
        AnalysisTags('table')._check()
    with pytest.raises(ValidationError, match=r"The space tag 'torus' is unknown"):
        AnalysisTags('series', space=frozenset({'torus'}))._check()


def test_missing_names_the_fields_a_system_lacks():
    """Test that the needed tags name the time, space, and randomness of a system that miss what is needed."""
    needed = AnalysisTags(
        'system',
        time=frozenset({'continuous'}),
        space=frozenset({'network'}),
        randomness='deterministic',
    )
    system = Tags(time=frozenset({'discrete'}), space=frozenset({'none'}), randomness='stochastic')
    assert needed.missing(system) == {'time': ['discrete'], 'space': ['none'], 'randomness': ['stochastic']}


def test_missing_accepts_one_needed_value_and_unknown_tags():
    """Test that a system with one of the needed spaces, or with unknown tags, misses nothing."""
    needed = AnalysisTags('network', space=frozenset({'network', 'hypergraph'}), randomness='deterministic')
    assert not needed.missing(Tags(space=frozenset({'network', 'lattice'})))
    assert not needed.missing(Tags(laws=None, randomness=None))
