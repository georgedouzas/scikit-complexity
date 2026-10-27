"""Test the passes and checks of the fractional formalism: fractional noise, the fractional form, and orders."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.errors import ValidationError
from skcomplex.core.laws import FractionalNoiseLaw, NoiseLaw, RateLaw
from skcomplex.core.spaces import DiscreteTime
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.backends import NumpyBackend
from skcomplex.engine.compile import compile_system, validate
from skcomplex.engine.problems import Form, OrderKind, SupportSpec
from skcomplex.engine.solvers import GrunwaldLetnikovSolver


def _fractional(amplitude, hurst):
    """Return an element type driven by one fractional noise law."""
    return type(
        'FractionalElement',
        (Element,),
        {
            'x': state(0.0),
            'D': parameter(1.0),
            'laws': lambda self, s: [FractionalNoiseLaw(s.x, amplitude(s), hurst=hurst)],
        },
    )


class RelaxingElement(Element):
    """Element whose state relaxes with a Caputo derivative of order 0.6."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=0.6)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the relaxation rate."""
        return [RateLaw(s.x, -s.x)]


class OverorderedElement(Element):
    """Element whose state declares an order outside the fractional interval."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=1.5)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


class RememberingElement(Element):
    """Element with a state of a fractional order and no laws."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=0.5)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


class ShakenElement(Element):
    """Element whose fractional state is driven by noise."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(order=0.6)] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the noise on the state."""
        return [NoiseLaw(s.x, 0.1)]


def test_validate_fractional_state_with_rk4_names_fractional_solvers():
    """Test that a fractional state with the solver rk4 gives one error naming the state and the fractional solvers."""
    report = validate(System(elements=[RelaxingElement('r')]), solver='rk4')
    assert [issue.code for issue in report.errors] == ['order-unsupported']
    assert 'r.x' in report.errors[0].message
    assert 'fractional-abm, grunwald-letnikov, l1' in report.errors[0].fix


def test_validate_order_out_of_range_names_interval():
    """Test that a state of order 1.5 gives one error naming the state and the interval of fractional orders."""
    report = validate(System(elements=[OverorderedElement('a')]))
    assert [issue.code for issue in report.errors] == ['fractional-order']
    assert 'a.x' in report.errors[0].message
    assert '(0, 1)' in report.errors[0].fix


def test_validate_fractional_state_under_discrete_time_suggests_continuous_time():
    """Test that a fractional state in a system of discrete time gives one error suggesting continuous time."""
    report = validate(System(elements=[RememberingElement('a')], time=DiscreteTime()))
    assert [issue.code for issue in report.errors] == ['fractional-time']
    assert 'a.x' in report.errors[0].message
    assert 'continuous time' in report.errors[0].fix


def test_validate_fractional_state_with_backend_without_orders():
    """Test that a backend that declares no fractional orders is reported for a fractional state."""

    class IntegerBackend(NumpyBackend):
        """NumPy backend that integrates integer orders alone."""

        supports = SupportSpec(orders={OrderKind.INTEGER})

    report = validate(System(elements=[RelaxingElement('r')]), backend=IntegerBackend())
    assert [issue.code for issue in report.errors] == ['order-unsupported']
    assert 'numba, numpy' in report.errors[0].fix
    assert validate(System(elements=[RelaxingElement('r')]), backend='numba').ok


def test_validate_noise_on_fractional_state_is_unsupported():
    """Test that a noise law on a fractional state is reported, since no fractional solver runs noise laws."""
    report = validate(System(elements=[ShakenElement('a')]))
    assert [issue.code for issue in report.errors] == ['solver-unsupported']
    assert 'a.x' in report.errors[0].message


def test_validate_fractional_state_with_fractional_solver():
    """Test that a fractional state with a fractional solver and default settings is valid."""
    system = System(elements=[RelaxingElement('r')])
    assert validate(system).ok
    assert validate(system, solver='l1').ok
    assert validate(system, solver=GrunwaldLetnikovSolver()).ok


def test_compile_system_rejects_a_hurst_index_out_of_range():
    """Test that a Hurst index of 1 names the law and the value and states the open interval."""
    rough = _fractional(lambda s: 1, 1.0)
    with pytest.raises(ValidationError, match=r'fractional noise law of f.x of f has the Hurst index 1.*\(0, 1\)'):
        compile_system(System(elements=[rough('f')]))


def test_compile_system_rejects_a_state_dependent_fractional_amplitude():
    """Test that a fractional amplitude that reads its own state names the state and the supported case."""
    multiplicative = _fractional(lambda s: s.x, 0.7)
    with pytest.raises(ValidationError, match=r'depends on the state f.x. Only state-independent amplitudes'):
        compile_system(System(elements=[multiplicative('f')]))


def test_compile_system_lowers_a_hurst_index_of_one_half_to_noise():
    """Test that fractional noise of Hurst index 1/2 compiles to an ordinary noise row with no fractional rows."""
    problem = compile_system(System(elements=[GroupElement('f', element=_fractional(lambda s: s.x, 0.5), size=3)]))
    plain = compile_system(
        System(
            elements=[
                GroupElement(
                    'f',
                    element=type(
                        'PlainElement',
                        (Element,),
                        {'x': state(0.0), 'laws': lambda self, s: [NoiseLaw(s.x, s.x)]},
                    ),
                    size=3,
                ),
            ],
        ),
    )
    assert problem.form == Form.SDE
    assert problem.fractional == ()
    assert np.array_equal(problem.noise_rows, plain.noise_rows)


def test_compile_system_records_the_hurst_index_of_fractional_rows():
    """Test that fractional rows of a group record the Hurst index of every row from the parameters."""
    problem = compile_system(
        System(elements=[GroupElement('f', element=_fractional(lambda s: (2 * s.D) ** 0.5, 0.3), size=4)]),
    )
    rows = problem.fractional[0]
    assert rows.positions.tolist() == [0, 1, 2, 3]
    assert np.allclose(rows.hurst(problem.initial_parameters), 0.3)
