"""Test the `_fractional` module of `skcomplex.core.components.laws.utils`."""

from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var
from scipy.special import erfcx, gamma

from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.components.laws.utils import fractional
from skcomplex.core.components.laws.utils._fractional import fractional_of
from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.engine.solvers import FractionalABMSolver, L1Solver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class AnomalousRelaxation(Interaction):
    """Interaction of the anomalous relaxation `D^alpha x = -k x`, with another law chosen by `extra`."""

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'cell',
        k: float = 1.0,
        alpha: float = 0.5,
        extra: str = '',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.k = k
        self.alpha = alpha
        self.extra = extra

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the fractional relaxation, and an ordinary rate or a noise of the state for `extra`."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            laws: list[BaseComponent] = [RateLaw(fractional(x, self.alpha), -context.k * x)]
            if self.extra == 'rate':
                laws.append(RateLaw(x, 1.0))
            if self.extra == 'noise':
                laws = [NoiseLaw(fractional(x, self.alpha), 0.1)]
            self.laws_ = laws


def relaxation(**params: object) -> System:
    """Return a system of one state that relaxes anomalously."""
    return System(elements=[Element('cell', values={'x': 1.0})], interactions=[AnomalousRelaxation('relax', **params)])


def mittag_leffler(alpha: float, z: np.ndarray) -> np.ndarray:
    """Return the Mittag-Leffler function `E_alpha(z)` from its power series, for moderate arguments."""
    return sum(z**k / gamma(alpha * k + 1) for k in range(150))


def test_fractional_holds_the_state_and_the_order():
    """Test that the derivative keeps the state and the order and prints both."""
    x = var('x')
    derivative = fractional(x, 0.5)
    assert str(derivative) == 'fractional(x, 0.5)'
    assert fractional_of(derivative) == (x, 0.5)
    assert fractional_of(x) is None


@pytest.mark.parametrize('alpha', [0, -0.5, 1.5, 'alpha', var('alpha')])
def test_fractional_rejects_an_order_out_of_range(alpha):
    """Test that an order that is no number in `(0, 1]` raises an error."""
    with pytest.raises(ValidationError, match=r'number in \(0, 1\]'):
        fractional(var('x'), alpha)


def test_fractional_rejects_a_target_that_is_not_a_state():
    """Test that the derivative of an expression of a state raises an error."""
    x = var('x')
    with pytest.raises(ValidationError, match='is taken of a state'):
        fractional(x + 1, 0.5)


def test_relaxation_follows_the_mittag_leffler_solution_of_order_one_half():
    """Test that `D^0.5 x = -k x` follows `E_0.5(-k t^0.5) = erfcx(k t^0.5)`."""
    system = relaxation(k=0.8, alpha=0.5).prepare()
    trajectory = system.simulate(5.0, dt=0.5)
    exact = erfcx(0.8 * np.sqrt(trajectory.times))
    np.testing.assert_allclose(trajectory.get_values('cell.x'), exact, atol=1e-4)


def test_relaxation_follows_the_mittag_leffler_series():
    """Test that `D^0.7 x = -x` follows `E_0.7(-t^0.7)`, also with the L1 solver."""
    exact = mittag_leffler(0.7, -np.linspace(0.0, 3.0, 7) ** 0.7)
    for solver in ('auto', L1Solver(dt=0.001)):
        system = System(
            elements=[Element('cell', values={'x': 1.0})],
            interactions=[AnomalousRelaxation('relax', k=1.0, alpha=0.7)],
            solver=solver,
        )
        trajectory = system.prepare().simulate(3.0, dt=0.5)
        np.testing.assert_allclose(trajectory.get_values('cell.x'), exact, atol=1e-3)


def test_prepare_picks_the_caputo_solver():
    """Test that a fractional target makes fractional equations solved by the predictor-corrector."""
    system = relaxation(alpha=0.5).prepare()
    assert system.form_ == 'fractional differential equations'
    assert isinstance(system.solver_, FractionalABMSolver)
    assert system.states_ == ('cell.x',)


def test_order_one_is_the_ordinary_derivative():
    """Test that the order 1 makes ordinary equations of exponential decay."""
    system = relaxation(alpha=1.0).prepare()
    trajectory = system.simulate(2.0, dt=1.0)
    assert system.form_ == 'ordinary differential equations'
    np.testing.assert_allclose(trajectory.get_values('cell.x'), np.exp(-trajectory.times), rtol=1e-4)


def test_fractional_and_ordinary_rates_of_one_state_raise_an_error():
    """Test that a state with a fractional and an ordinary rate law raises an error naming both orders."""
    with pytest.raises(ValidationError, match=r'fractional derivative of order 0\.5 and a derivative of order 1'):
        relaxation(extra='rate').prepare()


def test_noise_of_a_fractional_derivative_raises_an_error():
    """Test that a noise law cannot target a fractional derivative."""
    with pytest.raises(ValidationError, match='only a RateLaw of a state takes one'):
        relaxation(extra='noise').prepare()
