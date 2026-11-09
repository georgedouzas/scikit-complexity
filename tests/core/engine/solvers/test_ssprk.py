"""Test the `SSPRKSolver` class."""

import math
import warnings
from typing import Annotated

import numpy as np
import pytest

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import FluxLaw, RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import (
    SSPRKSolver,
)
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec


@pytest.mark.parametrize('riemann', ['rusanov', 'godunov'])
def test_numerical_fluxes_conserve(riemann):
    """Test that the Rusanov and Godunov fluxes conserve the total of a shock without overshoots."""

    class ShockElement(Element):
        """Shock element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            u: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.u = u

        def _get_laws(self, s):
            """Return the laws of the shock element."""
            return [FluxLaw(s.u, s.u**2 / 2, riemann=riemann)]

    initial = 1.5 + np.sin((np.arange(100) + 0.5) * 2 * math.pi / 100)
    line = FieldElement(
        'line',
        element=ShockElement,
        space=LatticeSpace(100, spacing=2 * math.pi / 100),
        values={'u': initial},
        discretization='finite-volume',
    )
    values = simulate(CompositeSystem(elements=[line]), until=2.0, dt=2.0, solver=SSPRKSolver()).states[-1]
    assert values.sum() == pytest.approx(initial.sum(), rel=1e-12)
    assert initial.min() - 1e-12 <= values.min() <= values.max() <= initial.max() + 1e-12


def _viscous_burgers_laws(s):
    """Return the Burgers flux and its viscous Laplacian."""
    return [FluxLaw(s.u, s.u**2 / 2), RateLaw(s.u, s.nu * laplacian(s.u))]


def test_viscous_burgers_respects_diffusive_limit():
    """Test that SSPRK keeps viscous Burgers within its initial range with no stability warning."""
    n = 200
    spacing = 2 * math.pi / n
    initial = 1.5 + np.sin((np.arange(n) + 0.5) * spacing)

    class ViscousElement(Element):
        """Viscous element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            u: Annotated[object, StateSpec()] = 0.0,
            nu: Annotated[object, ParameterSpec()] = 0.5,
        ) -> None:
            super().__init__(label, visual=visual)
            self.u = u
            self.nu = nu

        def _get_laws(self, s):
            """Return the laws of the viscous element."""
            return _viscous_burgers_laws(s)

    space = LatticeSpace(n, spacing=spacing)
    field = FieldElement(
        'line',
        element=ViscousElement,
        space=space,
        values={'u': initial},
        discretization='finite-volume',
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        values = simulate(CompositeSystem(elements=[field]), until=0.5, solver=SSPRKSolver(), dt=0.05).values('line.u')
    assert values.min() >= initial.min() - 1e-12
    assert values.max() <= initial.max() + 1e-12
