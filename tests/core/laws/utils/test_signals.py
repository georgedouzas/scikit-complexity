"""Test the `_signals` module of `skcomplex.core.laws.utils`."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element
from skcomplex.core.laws import RateLaw
from skcomplex.core.laws.utils import series
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.compile import compile_system
from skcomplex.specs import StateSpec, VisualSpec


def _driven_by(data):
    """Return a system whose one rate reads a series input."""

    class DrivenElement(Element):
        """Driven element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            y: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.y = y

        def _get_laws(self, s):
            """Return the laws of the driven element."""
            return [RateLaw(s.y, data(s.t))]

    driven = DrivenElement
    return System(elements=[driven('d')])


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_series_compiles_long_held_data(backend):
    """Test that a held series of 1000 points compiles and holds each value until the next time."""
    times, values = np.arange(1000.0), np.sin(np.arange(1000.0))
    problem = compile_system(_driven_by(lambda t: series(t, times, values, 'previous')), backend=backend)
    rates = [problem.drift(k + 0.5, problem.initial_state, problem.initial_parameters)[0] for k in range(1000)]
    np.testing.assert_allclose(rates, values, rtol=0, atol=1e-12)


@pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
def test_series_interpolates_long_linear_data(backend):
    """Test that a linear series of 1000 points matches linear interpolation and breaks at every data time."""
    times, values = np.arange(1000.0), np.sin(np.arange(1000.0))
    problem = compile_system(_driven_by(lambda t: series(t, times, values)), backend=backend)
    probes = np.linspace(-3.0, 1003.0, 2501)
    rates = [problem.drift(time, problem.initial_state, problem.initial_parameters)[0] for time in probes]
    np.testing.assert_allclose(rates, np.interp(probes, times, values), rtol=0, atol=1e-12)
    assert problem.breakpoints.between(-1.0, 1000.0) == list(times)
