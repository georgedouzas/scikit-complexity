"""Test the integrals of fields against two-point kernels and along one axis."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from sage.all__sagemath_symbolics import Expression, exp, tanh, var

from skcomplex.core.components.elements import FieldElement, NonnegativeElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws.utils import (
    coordinate,
    cumulative_integral,
    integral_transform,
    integration_variable,
)
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import AxisSpec, ObservableSpec


def product_integral(s: SimpleNamespace) -> object:
    """Return the integral of the density against the kernel `z u`."""
    return integral_transform(s.x, coordinate('z') * integration_variable('z'), ('z',))


def tail_integral(s: SimpleNamespace) -> object:
    """Return the integral of the density from every cell to the end of its class axis."""
    return cumulative_integral(s.x, 1, reverse=True)


def head_integral(s: SimpleNamespace) -> object:
    """Return the integral of the density from the start of its class axis to every cell."""
    return cumulative_integral(s.x, 1)


def light(s: SimpleNamespace) -> object:
    """Return the Beer-Lambert light under the canopy of the density weighted by the squared height."""
    return exp(-cumulative_integral(s.x * coordinate('h') ** 2, 1, reverse=True))


def _observed(values: np.ndarray, axis: AxisSpec, function: Callable[[SimpleNamespace], object]) -> np.ndarray:
    """Return an observable of a constant density on one class axis at every cell."""
    space = PhaseSpace(LatticeSpace((1,)), [axis]).prepare()
    field = FieldElement('g', element=NonnegativeElement, space=space, values={'x': values})
    observables = [ObservableSpec('value', function, group='g', reduce=np.asarray)]
    trajectory = simulate(System(elements=[field]), until=0.1, dt=0.1, observables=observables)
    return np.asarray(trajectory.values('value'), dtype=float)[-1].ravel()


def test_integration_variable_rejects_a_name_that_is_no_identifier() -> None:
    """Raise a named error for an axis name that is no identifier."""
    with pytest.raises(ValidationError, match='identifier'):
        integration_variable('not a name')


def test_cumulative_integral_rejects_an_axis_that_is_no_integer() -> None:
    """Raise a named error for an axis that is not a nonnegative integer."""
    for axis in (-1, 1.5, True):
        with pytest.raises(ValidationError, match='nonnegative integer'):
            cumulative_integral(var('x'), axis)


def test_integral_transform_samples_a_kernel_of_both_coordinates() -> None:
    """Integrate a density of one against the kernel `z u` into `z / 2` at every cell centre."""
    axis = AxisSpec('z', 20, 0.0, 1.0)
    rate = _observed(np.ones((1, 20)), axis, product_integral)
    centres = (np.arange(20) + 0.5) / 20
    np.testing.assert_allclose(rate, centres / 2, rtol=1e-12)


def test_integral_transform_with_an_antisymmetric_kernel_conserves_the_total() -> None:
    """Keep the total of a density whose rate integrates it against `rho(z, u) - rho(u, z)` within 1e-10."""
    z, u = coordinate('z'), integration_variable('z')

    def victory(first: Expression, second: Expression) -> Expression:
        return (1 + tanh(2 * ((first - second) + (first**2 - second**2)))) / 2

    def rate(s: SimpleNamespace) -> object:
        return s.target.x * integral_transform(s.target.x, victory(z, u) - victory(u, z), ('z',))

    space = PhaseSpace(LatticeSpace((1,)), [AxisSpec('z', 50, 0.0, 1.0)]).prepare()
    start = 1.0 + np.sin(3 * space.coordinates_['z'])[None, :]
    field = FieldElement('g', element=NonnegativeElement, space=space, values={'x': start})
    system = System(elements=[field], interactions=[RateInteraction('r', target='g', state='x', rate=rate)])
    final = simulate(system, until=5.0, dt=1.0, solver=RK4Solver(dt=0.05))
    values = final.values('g.x')
    assert abs(values[-1].sum() / values[0].sum() - 1.0) < 1e-10
    assert not np.allclose(values[-1], values[0])


def test_cumulative_integral_from_every_site_to_the_end() -> None:
    """Integrate a field of one along a class axis of 10 cells into `1 - h` at the cell of centre `h`."""
    axis = AxisSpec('h', 10, 0.0, 1.0)
    rate = _observed(np.ones((1, 10)), axis, tail_integral)
    centres = (np.arange(10) + 0.5) / 10
    np.testing.assert_allclose(rate, 1 - centres, rtol=1e-12)


def test_cumulative_integral_from_the_start_to_every_site() -> None:
    """Integrate a field of one along a class axis from its low end into the distance from it."""
    axis = AxisSpec('h', 10, 2.0, 3.0)
    rate = _observed(np.ones((1, 10)), axis, head_integral)
    np.testing.assert_allclose(rate, (np.arange(10) + 0.5) / 10, rtol=1e-12)


def test_cumulative_integral_shades_by_beer_lambert() -> None:
    """Give a light of one above the canopy that decreases downward through it."""
    axis = AxisSpec('h', 40, 0.0, 4.0)
    centres = 0.05 + 0.1 * np.arange(40)
    canopy = np.where(centres < 3.0, 1.0, 0.0)[None, :]
    rate = _observed(canopy, axis, light)
    assert np.all(rate[centres > 3.0] == 1.0)
    assert np.all(np.diff(rate) >= 0)
    assert rate[0] < 1e-3
