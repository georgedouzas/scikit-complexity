"""Test the `_noise` module of `skcomplex.core.laws`."""

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.laws import NoiseLaw
from skcomplex.errors import ValidationError


@pytest.mark.parametrize('process', ['market', ('market', 'rates'), None])
def test_noise_takes_a_label_labels_or_no_process(process):
    """Test that a noise law takes a label, a tuple of labels, or no process, and runs into its amplitude."""
    n = var('n')
    law = NoiseLaw(n, 0.1 * n, process=process).check().prepare()
    assert law.run({'n': 2.0}) == pytest.approx(0.2)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'process': 3}, 'Give a label'),
        ({'process': ()}, 'Give a label'),
        ({'conserved': True, 'process': 'market'}, 'one process per link'),
        ({'convention': 'ito-stratonovich'}, 'Unknown noise convention'),
    ],
)
def test_noise_rejects_invalid_processes_and_conventions(arguments, message):
    """Test that a noise law rejects an invalid process, a process of conserved noise, and an unknown convention."""
    with pytest.raises(ValidationError, match=message):
        NoiseLaw(var('n'), 1.0, **arguments).check()
