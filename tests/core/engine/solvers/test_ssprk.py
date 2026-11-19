"""Test the `SSPRKSolver` class."""

import math
import warnings

import numpy as np
import pytest

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import divergence, laplacian
from skcomplex.core.components.spaces import FiniteVolumeSpec, LatticeSpace
from skcomplex.core.engine.solvers import SSPRKSolver
from skcomplex.core.systems import System


def _shock_laws(s):
    """Return the Burgers flux of the line."""
    u = s.target.u
    return [RateLaw(u.diff(s.t), -divergence([u**2 / 2]))]


def _viscous_burgers_laws(s):
    """Return the Burgers flux and its viscous Laplacian."""
    u = s.target.u
    return [*_shock_laws(s), RateLaw(u.diff(s.t), s.target.nu * laplacian(u))]


@pytest.mark.parametrize('riemann', ['rusanov', 'godunov'])
def test_numerical_fluxes_conserve(riemann):
    """Test that the Rusanov and Godunov fluxes conserve the total of a shock without overshoots."""
    initial = 1.5 + np.sin((np.arange(100) + 0.5) * 2 * math.pi / 100)
    space = LatticeSpace(
        100,
        spacing=2 * math.pi / 100,
        discretization='finite-volume',
        scheme=FiniteVolumeSpec(riemann=riemann),
    )
    line = Element('line', values={'u': initial}, space=space)
    flux = Interaction('flux', roles={'target': 'line'}, laws=_shock_laws)
    system = System(elements=[line], interactions=[flux], solver=SSPRKSolver()).prepare()
    values = system.simulate(until=2.0, dt=2.0).states[-1]
    assert values.sum() == pytest.approx(initial.sum(), rel=1e-12)
    assert initial.min() - 1e-12 <= values.min() <= values.max() <= initial.max() + 1e-12


def test_viscous_burgers_respects_diffusive_limit():
    """Test that SSPRK keeps viscous Burgers within its initial range with no stability warning."""
    n = 200
    spacing = 2 * math.pi / n
    initial = 1.5 + np.sin((np.arange(n) + 0.5) * spacing)
    space = LatticeSpace(n, spacing=spacing, discretization='finite-volume')
    line = Element('line', values={'u': initial, 'nu': 0.5}, space=space)
    flux = Interaction('flux', roles={'target': 'line'}, laws=_viscous_burgers_laws)
    system = System(elements=[line], interactions=[flux], solver=SSPRKSolver()).prepare()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        values = system.simulate(until=0.5, dt=0.05).get_values('line.u')
    assert values.min() >= initial.min() - 1e-12
    assert values.max() <= initial.max() + 1e-12
