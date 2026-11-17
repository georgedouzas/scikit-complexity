"""Test the `_observable_spec` module of `skcomplex.core.components`."""

from multiprocessing.reduction import ForkingPickler

import numpy as np
import pytest

from skcomplex.core.components import Observable
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError

from ...conftest import _total, _two_populations


def test_observable_lambda_is_rejected():
    """Test that checking an observable whose expression or reduction is a lambda raises a named error."""
    with pytest.raises(ValidationError, match="expression of the observable 'total'"):
        Observable('total', lambda s: s.a.n + s.b.n)._check()
    with pytest.raises(ValidationError, match='reduction'):
        Observable('spread', group='herd', variable='n', reduce=lambda values: values.ptp())._check()


def test_observable_module_function_pickles():
    """Test that a module-level observable pickles and records the same values once restored."""
    observables = [Observable('total', _total)]
    restored = ForkingPickler.loads(ForkingPickler.dumps(observables))
    expected = simulate(_two_populations(), until=1, dt=0.5, observables=observables).get_values('total')
    values = simulate(_two_populations(), until=1, dt=0.5, observables=restored).get_values('total')
    assert np.allclose(values, expected)
