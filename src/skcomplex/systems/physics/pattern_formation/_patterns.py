"""Pattern formation systems: amplitude equations and rolls."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.components.elements import Element, FieldElement
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import ETDRK4Solver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system
from ..common.fields import SwiftHohenbergEnergyInteraction


class GinzburgLandauElement(Element):
    """Site of the complex Ginzburg-Landau equation `A_t = A + (1 + i b) laplacian(A) - (1 - i c) |A|^2 A`.

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


def _noise(shape: tuple[int, ...], scale: float, random_state: int | np.random.Generator | None) -> Array:
    """Return small uniform noise of a shape."""
    return scale * np.random.default_rng(random_state).uniform(-1.0, 1.0, size=shape)


@register_system(
    field='physics/pattern_formation',
    tags=('field', 'chaos', 'spirals', 'spectral', 'pde'),
    references=('Aranson and Kramer 2002', 'Gros 2015'),
    defaults={'until': 50.0, 'dt': 1.0, 'solver': ETDRK4Solver(dt=0.1)},
)
def complex_ginzburg_landau(
    shape: Sequence[int] = (64, 64),
    *,
    length: float = 64.0,
    b: float = 1.0,
    c: float = -1.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the complex Ginzburg-Landau equation `A_t = A + (1 + i b) laplacian(A) - (1 + i c) |A|^2 A`.

    The amplitude `A = u + i v` is held as its real and imaginary parts, and the linear part couples them per Fourier
    mode. With `1 + b c < 0` the uniform oscillation is unstable and spiral defects turn into turbulence.

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

    Returns:
        system:
            The system, with the real part `medium.u` and the imaginary part `medium.v`.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import complex_ginzburg_landau
        >>> complex_ginzburg_landau((8, 8)).elements[0].values['c']
        -1.5
    """
    shape = tuple(shape)
    rng = np.random.default_rng(random_state)
    values = {'u': _noise(shape, 0.01, rng), 'v': _noise(shape, 0.01, rng), 'b': b, 'c': c}
    space = LatticeSpace(shape, spacing=length / shape[0])
    return System(
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


@register_system(
    field='physics/pattern_formation',
    tags=('field', 'rolls', 'convection', 'spectral', 'pde'),
    references=('Swift and Hohenberg 1977', 'Cross and Hohenberg 1993', 'Datseris and Parlitz 2022'),
    defaults={'until': 200.0, 'dt': 5.0, 'solver': ETDRK4Solver(dt=0.5)},
)
def swift_hohenberg(
    shape: Sequence[int] = (64, 64),
    *,
    length: float = 16 * math.pi,
    r: float = 0.3,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Swift-Hohenberg equation `u_t = r u - (1 + laplacian)^2 u - u^3`, which forms rolls of length 2 pi.

    Above onset, `r > 0`, small noise grows into rolls whose wavenumber is near 1, the fastest growing one.

    Args:
        shape:
            The number of sites along every axis of the periodic lattice.

        length:
            The side of the domain.

        r:
            The distance above onset.

        random_state:
            The seed or generator of the noise.

    Returns:
        system:
            The system, with the field `cells.u`.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import swift_hohenberg
        >>> swift_hohenberg((8, 8), r=0.1).interactions[0].r
        0.1
    """
    shape = tuple(shape)
    values = {'u': _noise(shape, 0.01, random_state)}
    space = LatticeSpace(shape, spacing=length / shape[0])
    return System(
        elements=[
            FieldElement('cells', element=PatternFieldElement, space=space, values=values, discretization='spectral'),
        ],
        interactions=[SwiftHohenbergEnergyInteraction('energy', target='cells', r=r)],
    )
