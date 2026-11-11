"""Absorbing-state transitions.

The module holds the Langevin field theory of directed percolation coupled to an energy field.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import sqrt

from ....core.components.elements import Element, FieldElement
from ....core.components.laws import BaseLaw, BoundaryLaw, NoiseLaw, RateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ActiveFieldElement(Element):
    """Site of the Langevin equation of directed percolation with a conserved field.

    The site has an activity `rho` and a field `nu`.

    The activity grows at the rate `a + b nu`, saturates, diffuses, and fluctuates with the multiplicative noise
    `e sqrt(rho)` of an absorbing state, `rho_t = (a + b nu) rho - c rho^2 + d laplacian(rho) + e sqrt(rho) eta`. The
    field diffuses, gains `f`, and is consumed by the activity, `nu_t = laplacian(nu) + f - g rho`. Both stay
    nonnegative.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        rho:
            The initial activity.

        nu:
            The initial field.

        a:
            The linear growth rate of the activity.

        b:
            The growth of the activity per unit of the field.

        c:
            The saturation of the activity.

        d:
            The diffusion coefficient of the activity.

        e:
            The amplitude of the noise.

        f:
            The input of the field.

        g:
            The consumption of the field by the activity.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.criticality import ActiveFieldElement
        >>> ActiveFieldElement('site').e
        0.2
    """

    law_classes = (
        BoundaryLaw,
        NoiseLaw,
        RateLaw,
    )

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        rho: Annotated[float, StateSpec()] = 0.5,
        nu: Annotated[float, StateSpec()] = 1.0,
        a: Annotated[float, ParameterSpec()] = -1.0,
        b: Annotated[float, ParameterSpec()] = 1.0,
        c: Annotated[float, ParameterSpec()] = 1.0,
        d: Annotated[float, ParameterSpec()] = 1.0,
        e: Annotated[float, ParameterSpec()] = 0.2,
        f: Annotated[float, ParameterSpec()] = 0.0,
        g: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.rho = rho
        self.nu = nu
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates, the noise, and the bounds of the activity and the field.

        The bounds keep both nonnegative.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of `rho` and `nu`.
        """
        s = namespace
        return [
            RateLaw(s.rho, (s.a + s.b * s.nu) * s.rho - s.c * s.rho**2 + s.d * laplacian(s.rho)),
            NoiseLaw(s.rho, s.e * sqrt(s.rho)),
            RateLaw(s.nu, laplacian(s.nu) + s.f - s.g * s.rho),
            BoundaryLaw(s.rho, (s.rho + abs(s.rho)) / 2),
            BoundaryLaw(s.nu, (s.nu + abs(s.nu)) / 2),
        ]


class DirectedPercolationLangevinSystem(BaseSystem):
    """The Langevin field theory of an absorbing-state transition.

    An active density is fed by an energy field.

    The active density `rho` grows at the local rate `a + b nu`, saturates through `-c rho^2`, spreads with the
    diffusivity `d`, and fluctuates with the multiplicative noise `e sqrt(rho)`, which vanishes in the absorbing state
    `rho = 0`. The energy field `nu` diffuses, gains the drive `f`, and loses the dissipation `g rho`, so `rho_t = (a +
    b nu) rho - c rho^2 + d laplacian(rho) + e sqrt(rho) xi` and `nu_t = laplacian(nu) + f - g rho` (Tian et al., Eqs.
    3-4). Both fields stay nonnegative. Without drive and dissipation the energy is a fixed control parameter: below the
    threshold `nu_c = -a / b` activity dies out, and above it the density settles near `(a + b nu) / c`. The system has
    the activity `neurons.rho` and the energy `neurons.nu`.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        a:
            The bare growth rate of the activity.

        b:
            The gain of the growth rate per unit of energy.

        c:
            The saturation of the activity.

        d:
            The diffusivity of the activity.

        e:
            The amplitude of the multiplicative noise.

        f:
            The drive of the energy.

        g:
            The dissipation of the energy per unit of activity.

        energy:
            The initial energy at every site.

        density:
            The mean initial activity, drawn uniformly between 0 and twice this value.

        random_state:
            The seed or generator of the initial activity.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.criticality import DirectedPercolationLangevinSystem
        >>> system = DirectedPercolationLangevinSystem((8, 8), energy=0.5).prepare().system_
        >>> float(system.elements[0].values['nu'].mean())
        0.5
    """

    field = 'physics/criticality'
    topics = ('field', 'absorbing state', 'directed percolation', 'multiplicative noise', 'spde')
    references = ('Tian et al. 2022', 'Henkel, Hinrichsen, and Lübeck 2008', 'Bonachela and Muñoz 2009')
    until = 100.0
    dt = 1.0
    solver = EulerMaruyamaSolver(dt=0.01)

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        a: float = -1.0,
        b: float = 1.0,
        c: float = 1.0,
        d: float = 1.0,
        e: float = 0.2,
        f: float = 0.0,
        g: float = 0.0,
        energy: float = 1.2,
        density: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.energy = energy
        self.density = density
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system, built from its model parameters."""
        shape = tuple(self.shape)
        rng = np.random.default_rng(self.random_state)
        values = {
            'rho': rng.uniform(0.0, 2 * self.density, shape),
            'nu': np.full(shape, self.energy),
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'd': self.d,
            'e': self.e,
            'f': self.f,
            'g': self.g,
        }
        return CompositeSystem(
            self.label,
            elements=[FieldElement('neurons', element=ActiveFieldElement, space=LatticeSpace(shape), values=values)],
        )
