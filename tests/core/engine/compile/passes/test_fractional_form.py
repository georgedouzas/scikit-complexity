"""Test the passes and checks of the fractional formalism: fractional noise, the fractional form, and orders."""

from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import FractionalNoiseLaw, NoiseLaw, RateLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.solvers import GrunwaldLetnikovSolver, L1Solver, RK4Solver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec, SupportSpec, VisualSpec
from skcomplex.core.types import FormKind, OrderKind


class RoughElement(Element):
    """Element driven by fractional noise of the Hurst index 1, outside the open interval."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.D = D

    def _get_laws(self, s):
        """Return the noise on the state."""
        return [FractionalNoiseLaw(s.x, 1, hurst=1.0)]


class MultiplicativeElement(Element):
    """Element driven by fractional noise whose amplitude reads its own state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.D = D

    def _get_laws(self, s):
        """Return the noise on the state."""
        return [FractionalNoiseLaw(s.x, s.x, hurst=0.7)]


class BrownianElement(Element):
    """Element driven by fractional noise of the Hurst index 1/2, which is ordinary noise."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.D = D

    def _get_laws(self, s):
        """Return the noise on the state."""
        return [FractionalNoiseLaw(s.x, s.x, hurst=0.5)]


class PlainElement(Element):
    """Element driven by ordinary noise whose amplitude reads its own state."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.D = D

    def _get_laws(self, s):
        """Return the noise on the state."""
        return [NoiseLaw(s.x, s.x)]


class PersistentElement(Element):
    """Element driven by fractional noise of the Hurst index 0.3 with an amplitude of a parameter."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.D = D

    def _get_laws(self, s):
        """Return the noise on the state."""
        return [FractionalNoiseLaw(s.x, (2 * s.D) ** 0.5, hurst=0.3)]


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
    report = PassCompiler(solver=RK4Solver()).prepare(System(elements=[RelaxingElement('r')])).report_
    assert [issue.code for issue in report.errors] == ['order-unsupported']
    assert 'r.x' in report.errors[0].message
    assert 'FractionalABMSolver' in report.errors[0].fix


def test_validate_order_out_of_range_names_interval():
    """Test that a state of order 1.5 gives one error naming the element and the interval of fractional orders."""
    report = PassCompiler().prepare(System(elements=[OverorderedElement('a')])).report_
    assert [issue.code for issue in report.errors] == ['structure']
    assert 'a: The order 1.5' in report.errors[0].message
    assert '(0, 1)' in report.errors[0].message


def test_validate_fractional_state_under_discrete_time_suggests_continuous_time():
    """Test that a fractional state in a system of discrete time gives one error suggesting continuous time."""
    report = PassCompiler().prepare(System(elements=[RememberingElement('a')], time=DiscreteTime())).report_
    assert [issue.code for issue in report.errors] == ['fractional-time']
    assert 'a.x' in report.errors[0].message
    assert 'continuous time' in report.errors[0].fix


def test_validate_fractional_state_with_backend_without_orders():
    """Test that a backend that declares no fractional orders is reported for a fractional state."""

    class IntegerBackend(NumpyBackend):
        """NumPy backend that integrates integer orders alone."""

        @classmethod
        def _get_support(cls):
            return SupportSpec(orders={OrderKind.INTEGER})

    report = PassCompiler(backend=IntegerBackend()).prepare(System(elements=[RelaxingElement('r')])).report_
    assert [issue.code for issue in report.errors] == ['order-unsupported']
    assert 'fractional orders' in report.errors[0].fix
    assert PassCompiler(backend=NumbaBackend()).prepare(System(elements=[RelaxingElement('r')])).report_.ok


def test_validate_noise_on_fractional_state_is_unsupported():
    """Test that a noise law on a fractional state is reported, since no fractional solver runs noise laws."""
    report = PassCompiler().prepare(System(elements=[ShakenElement('a')])).report_
    assert [issue.code for issue in report.errors] == ['solver-unsupported']
    assert 'a.x' in report.errors[0].message


def test_validate_fractional_state_with_fractional_solver():
    """Test that a fractional state with a fractional solver and default settings is valid."""
    system = System(elements=[RelaxingElement('r')])
    assert PassCompiler().prepare(system).report_.ok
    assert PassCompiler(solver=L1Solver()).prepare(system).report_.ok
    assert PassCompiler(solver=GrunwaldLetnikovSolver()).prepare(system).report_.ok


def test_compiler_rejects_a_hurst_index_out_of_range():
    """Test that a Hurst index of 1 names the element and the value and states the open interval."""
    with pytest.raises(
        ValidationError,
        match=r'f: The Hurst index of the noise of x_f is 1.0. Give a number in \(0, 1\)',
    ):
        PassCompiler().prepare(System(elements=[RoughElement('f')])).run()


def test_compiler_rejects_a_state_dependent_fractional_amplitude():
    """Test that a fractional amplitude that reads its own state names the state and the supported case."""
    with pytest.raises(ValidationError, match=r'depends on the state f.x. Only state-independent amplitudes'):
        PassCompiler().prepare(System(elements=[MultiplicativeElement('f')])).run()


def test_compiler_lowers_a_hurst_index_of_one_half_to_noise():
    """Test that fractional noise of Hurst index 1/2 compiles to an ordinary noise row with no fractional rows."""
    problem = (
        PassCompiler().prepare(System(elements=[GroupElement('f', element=BrownianElement, size=3)])).run().problem_
    )
    plain = PassCompiler().prepare(System(elements=[GroupElement('f', element=PlainElement, size=3)])).run().problem_
    assert problem.form == FormKind.SDE
    assert problem.fractional == ()
    assert np.array_equal(problem.noise_rows, plain.noise_rows)


def test_compiler_records_the_hurst_index_of_fractional_rows():
    """Test that fractional rows of a group record the Hurst index of every row from the parameters."""
    problem = (
        PassCompiler().prepare(System(elements=[GroupElement('f', element=PersistentElement, size=4)])).run().problem_
    )
    rows = problem.fractional[0]
    assert rows.positions.tolist() == [0, 1, 2, 3]
    assert np.allclose(rows.hurst(problem.initial_parameters), 0.3)
