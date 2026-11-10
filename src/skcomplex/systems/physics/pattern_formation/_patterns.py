"""Pattern formation systems.

The module covers amplitude equations and rolls.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import GradientInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import ETDRK4Solver
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array


class GinzburgLandauElement(Element):
    """Site of the complex Ginzburg-Landau equation.

    The equation is `A_t = A + (1 + i b) laplacian(A) - (1 - i c) |A|^2 A`.

    The amplitude `A = u + i v` is held as its real and imaginary parts. Without the Laplacian every site is a
    Stuart-Landau oscillator, and the dispersion `b` and the nonlinear frequency shift `c` set whether plane waves are
    stable or break into spiral defect chaos.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        u:
            The initial real part of the amplitude.

        v:
            The initial imaginary part of the amplitude.

        b:
            The linear dispersion.

        c:
            The nonlinear frequency shift.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import GinzburgLandauElement
        >>> GinzburgLandauElement('site').c
        -1.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        v: Annotated[float, StateSpec()] = 0.0,
        b: Annotated[float, ParameterSpec()] = 1.0,
        c: Annotated[float, ParameterSpec()] = -1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v
        self.b = b
        self.c = c

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the real and imaginary parts of the amplitude.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rates of `u` and `v`.
        """
        s = namespace
        square = s.u**2 + s.v**2
        return [
            RateLaw(s.u, s.u + laplacian(s.u) - s.b * laplacian(s.v) - square * (s.u - s.c * s.v)),
            RateLaw(s.v, s.v + s.b * laplacian(s.u) + laplacian(s.v) - square * (s.c * s.u + s.v)),
        ]


class PatternFieldElement(Element):
    """Site of a real field `u` whose patterns a free energy selects.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        u:
            The initial value of the field.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import PatternFieldElement
        >>> PatternFieldElement('site', u=0.1).u
        0.1
    """

    def __init__(self, label: str, visual: VisualSpec | None = None, *, u: Annotated[float, StateSpec()] = 0.0) -> None:
        super().__init__(label, visual=visual)
        self.u = u


class SwiftHohenbergEnergyInteraction(GradientInteraction):
    """Gradient flow of a field in the Swift-Hohenberg free energy.

    The flow is `u_t = r u - (1 + laplacian)^2 u + g u^2 - u^3`.

    The free energy is `F = integral -r u^2 / 2 + ((1 + laplacian) u)^2 / 2 - g u^3 / 3 + u^4 / 4`. Above onset,
    `r > 0`, the modes of wavenumber near 1 grow into rolls, or into hexagons for a large enough quadratic term `g`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the field.

        state:
            The name of the state of the field.

        r:
            The distance above onset.

        g:
            The strength of the quadratic term, which breaks the symmetry `u -> -u`.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.systems.physics.pattern_formation._patterns import SwiftHohenbergEnergyInteraction
        >>> energy = SwiftHohenbergEnergyInteraction('energy', target='cells').prepare()
        >>> namespace = SimpleNamespace(target=SimpleNamespace(u=var('u')), r=var('r'), g=0)
        >>> [law.value for law in energy.run(namespace).laws_]
        [-u^3 + r*u - u - 2*laplacian(u) - laplacian(laplacian(u))]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'u',
        r: Annotated[float, ParameterSpec()] = 0.3,
        g: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, target=target, states=state, variational=True)
        self.state = state
        self.r = r
        self.g = g

    def _get_potential(self, namespace: SimpleNamespace) -> Expression:
        """Return the Swift-Hohenberg free energy density.

        The density is `-r u^2 / 2 + (u + laplacian(u))^2 / 2 - g u^3 / 3 + u^4 / 4`.

        Args:
            namespace:
                The namespace of the symbols the density can use.

        Returns:
            density:
                The density of the free energy.
        """
        u = getattr(namespace.target, self.state)
        return -namespace.r * u**2 / 2 + (u + laplacian(u)) ** 2 / 2 - namespace.g * u**3 / 3 + u**4 / 4


class ComplexGinzburgLandauSystem(BaseSystem):
    """The complex Ginzburg-Landau equation.

    The equation is `A_t = A + (1 + i b) laplacian(A) - (1 + i c) |A|^2 A`.

    The amplitude `A = u + i v` is held as its real and imaginary parts, and the linear part couples them per Fourier
    mode. With `1 + b c < 0` the uniform oscillation is unstable and spiral defects turn into turbulence. The system has
    the real part `medium.u` and the imaginary part `medium.v`.

    Args:
        shape:
            The number of sites along both axes of the periodic lattice.

        length:
            The side of the square.

        b:
            The linear dispersion.

        c:
            The nonlinear dispersion.

        random_state:
            The seed or generator of the small random initial amplitude.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import ComplexGinzburgLandauSystem
        >>> ComplexGinzburgLandauSystem((8, 8)).prepare().system_.elements[0].values['c']
        -1.5
    """

    field = 'physics/pattern_formation'
    topics = ('field', 'chaos', 'spirals', 'spectral', 'pde')
    references = ('Aranson and Kramer 2002', 'Gros 2015')
    until = 50.0
    dt = 1.0
    solver = ETDRK4Solver(dt=0.1)

    @staticmethod
    def _get_noise(shape: tuple[int, ...], scale: float, random_state: int | np.random.Generator | None) -> Array:
        """Return small uniform noise of a shape."""
        return scale * np.random.default_rng(random_state).uniform(-1.0, 1.0, size=shape)

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        length: float = 64.0,
        b: float = 1.0,
        c: float = -1.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.length = length
        self.b = b
        self.c = c
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the complex Ginzburg-Landau equation.

        The equation is `A_t = A + (1 + i b) laplacian(A) - (1 + i c) |A|^2 A`.
        """
        shape = tuple(self.shape)
        rng = np.random.default_rng(self.random_state)
        values = {
            'u': ComplexGinzburgLandauSystem._get_noise(shape, 0.01, rng),
            'v': ComplexGinzburgLandauSystem._get_noise(shape, 0.01, rng),
            'b': self.b,
            'c': self.c,
        }
        space = LatticeSpace(shape, spacing=self.length / shape[0])
        return CompositeSystem(
            self.label,
            elements=[
                FieldElement(
                    'medium',
                    element=GinzburgLandauElement,
                    space=space,
                    values=values,
                    discretization='spectral',
                ),
            ],
        )


class SwiftHohenbergSystem(BaseSystem):
    """The Swift-Hohenberg equation `u_t = r u - (1 + laplacian)^2 u - u^3`.

    The equation forms rolls of length 2 pi.

    Above onset, `r > 0`, small noise grows into rolls whose wavenumber is near 1, the fastest growing one. The system
    has the field `cells.u`.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        length:
            The side of the domain.

        r:
            The distance above onset.

        random_state:
            The seed or generator of the noise.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import SwiftHohenbergSystem
        >>> SwiftHohenbergSystem((8, 8), r=0.1).prepare().system_.interactions[0].r
        0.1
    """

    field = 'physics/pattern_formation'
    topics = ('field', 'rolls', 'convection', 'spectral', 'pde')
    references = ('Swift and Hohenberg 1977', 'Cross and Hohenberg 1993', 'Datseris and Parlitz 2022')
    until = 200.0
    dt = 5.0
    solver = ETDRK4Solver(dt=0.5)

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        length: float = 16 * math.pi,
        r: float = 0.3,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.length = length
        self.r = r
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        shape = tuple(self.shape)
        values = {'u': ComplexGinzburgLandauSystem._get_noise(shape, 0.01, self.random_state)}
        space = LatticeSpace(shape, spacing=self.length / shape[0])
        return CompositeSystem(
            self.label,
            elements=[
                FieldElement(
                    'cells',
                    element=PatternFieldElement,
                    space=space,
                    values=values,
                    discretization='spectral',
                ),
            ],
            interactions=[SwiftHohenbergEnergyInteraction('energy', target='cells', r=self.r)],
        )
