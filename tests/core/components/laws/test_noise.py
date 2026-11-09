"""Test the `_noise` module of `skcomplex.core.components.laws`."""

import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import NoiseLaw
from skcomplex.core.errors import ValidationError


@pytest.mark.parametrize('process', ['market', ('market', 'rates'), None])
def test_noise_takes_a_label_labels_or_no_process(process):
    """Test that a noise law takes a label, a tuple of labels, or no process, and runs into its amplitude."""
    n = var('n')
    law = NoiseLaw(n, 0.1 * n, process=process).prepare()
    assert law.run({'n': 2.0}).value_ == pytest.approx(0.2)


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
        NoiseLaw(var('n'), 1.0, **arguments).prepare()


@pytest.mark.parametrize('hurst', [0.3, 0.7, var('H')])
def test_noise_takes_a_hurst_index(hurst):
    """Test that a noise law takes a Hurst index in (0, 1) or a parameter."""
    law = NoiseLaw(var('n'), 1.0, hurst=hurst).prepare()
    assert law.hurst is hurst


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'hurst': 1.0}, r'Give a number in \(0, 1\)'),
        ({'hurst': 0}, r'Give a number in \(0, 1\)'),
        ({'hurst': 0.7, 'process': 'shared'}, r'n .*Remove the process'),
        ({'hurst': 0.7, 'conserved': True}, r'n .*a conserved noise is Brownian'),
    ],
)
def test_noise_rejects_invalid_hurst_indices(arguments, message):
    """Test that a noise law rejects a Hurst index out of range, with a process, or on a conserved noise."""
    with pytest.raises(ValidationError, match=message):
        NoiseLaw(var('n'), 1.0, **arguments).prepare()
