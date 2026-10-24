"""Test the passes and checks of the fractional formalism: fractional noise, the fractional form, and orders."""

import numpy as np
import pytest

from skcomplex.core.base import Form, OrderKind, Supports, ValidationError, state
from skcomplex.core.components import Element
from skcomplex.core.laws import Noise, Rate
from skcomplex.core.space import DiscreteTime
from skcomplex.core.system import System
from skcomplex.engine.backends import NumpyBackend
from skcomplex.engine.compile import compile_system, validate
from skcomplex.formalisms.fractional import FractionalNoise, GrunwaldLetnikov


def _fractional(amplitude, hurst):
    """Return an element type driven by one fractional noise law."""
    return Element.define(
        'Fractional',
        states={'x': 0.0},
        parameters={'D': 1.0},
        laws=lambda s: [FractionalNoise(s.x, amplitude(s), hurst=hurst)],
    )


class Relaxing(Element):
    """Element whose state relaxes with a Caputo derivative of order 0.6."""

    x: float = state(1.0, order=0.6)

    def laws(self, s):
        """Return the relaxation rate."""
        return [Rate(s.x, -s.x)]


class Overordered(Element):
    """Element whose state declares an order outside the fractional interval."""

    x: float = state(1.0, order=1.5)


class Remembering(Element):
    """Element with a state of a fractional order and no laws."""

    x: float = state(1.0, order=0.5)


class Shaken(Element):
    """Element whose fractional state is driven by noise."""

    x: float = state(1.0, order=0.6)

    def laws(self, s):
        """Return the noise on the state."""
        return [Noise(s.x, 0.1)]


def test_validate_fractional_state_with_rk4_names_fractional_solvers():
    """Test that a fractional state with the solver rk4 gives one error naming the state and the fractional solvers."""
    report = validate(System(components=[Relaxing('r')]), solver='rk4')
    assert [issue.code for issue in report.errors] == ['order-unsupported']
    assert 'r.x' in report.errors[0].message
    assert 'fractional-abm, grunwald-letnikov, l1' in report.errors[0].fix


def test_validate_order_out_of_range_names_interval():
    """Test that a state of order 1.5 gives one error naming the state and the interval of fractional orders."""
    report = validate(System(components=[Overordered('a')]))
    assert [issue.code for issue in report.errors] == ['fractional-order']
    assert 'a.x' in report.errors[0].message
    assert '(0, 1)' in report.errors[0].fix


def test_validate_fractional_state_under_discrete_time_suggests_continuous_time():
    """Test that a fractional state in a system of discrete time gives one error suggesting continuous time."""
    report = validate(System(components=[Remembering('a')], time=DiscreteTime()))
    assert [issue.code for issue in report.errors] == ['fractional-time']
    assert 'a.x' in report.errors[0].message
    assert 'continuous time' in report.errors[0].fix


def test_validate_fractional_state_with_backend_without_orders():
    """Test that a backend that declares no fractional orders is reported for a fractional state."""

    class IntegerBackend(NumpyBackend):
        """NumPy backend that integrates integer orders alone."""

        supports = Supports(orders={OrderKind.INTEGER})

    report = validate(System(components=[Relaxing('r')]), backend=IntegerBackend())
    assert [issue.code for issue in report.errors] == ['order-unsupported']
    assert 'numba, numpy' in report.errors[0].fix
    assert validate(System(components=[Relaxing('r')]), backend='numba').ok


def test_validate_noise_on_fractional_state_is_unsupported():
    """Test that a noise law on a fractional state is reported, since no fractional solver runs noise laws."""
    report = validate(System(components=[Shaken('a')]))
    assert [issue.code for issue in report.errors] == ['solver-unsupported']
    assert 'a.x' in report.errors[0].message


def test_validate_fractional_state_with_fractional_solver():
    """Test that a fractional state with a fractional solver and default settings is valid."""
    system = System(components=[Relaxing('r')])
    assert validate(system).ok
    assert validate(system, solver='l1').ok
    assert validate(system, solver=GrunwaldLetnikov()).ok


def test_compile_system_rejects_a_hurst_index_out_of_range():
    """Test that a Hurst index of 1 names the law and the value and states the open interval."""
    rough = _fractional(lambda s: 1, 1.0)
    with pytest.raises(ValidationError, match=r'fractional noise law of f.x of f has the Hurst index 1.*\(0, 1\)'):
        compile_system(System(components=[rough('f')]))


def test_compile_system_rejects_a_state_dependent_fractional_amplitude():
    """Test that a fractional amplitude that reads its own state names the state and the supported case."""
    multiplicative = _fractional(lambda s: s.x, 0.7)
    with pytest.raises(ValidationError, match=r'depends on the state f.x. Only state-independent amplitudes'):
        compile_system(System(components=[multiplicative('f')]))


def test_compile_system_lowers_a_hurst_index_of_one_half_to_noise():
    """Test that fractional noise of Hurst index 1/2 compiles to an ordinary noise row with no fractional rows."""
    problem = compile_system(System(components=[_fractional(lambda s: s.x, 0.5).group('f', 3)]))
    plain = compile_system(
        System(components=[Element.define('Plain', states={'x': 0.0}, laws=lambda s: [Noise(s.x, s.x)]).group('f', 3)]),
    )
    assert problem.form is Form.SDE
    assert problem.fractional == ()
    assert np.array_equal(problem.noise_rows, plain.noise_rows)


def test_compile_system_records_the_hurst_index_of_fractional_rows():
    """Test that fractional rows of a group record the Hurst index of every row from the parameters."""
    problem = compile_system(System(components=[_fractional(lambda s: (2 * s.D) ** 0.5, 0.3).group('f', 4)]))
    rows = problem.fractional[0]
    assert rows.positions.tolist() == [0, 1, 2, 3]
    assert np.allclose(rows.hurst(problem.initial_parameters), 0.3)
