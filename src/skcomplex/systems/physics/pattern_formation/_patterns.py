"""Pattern formation systems.

The module covers amplitude equations and rolls.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, ETDRK4Solver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ...general.gradients import GradientInteraction


class GinzburgLandauElement(Element):
    """Sites of a complex amplitude `A = u + i v`, held as its real part `u` and its imaginary part `v`.

    Args:
        label:
            The label of the field.

        u:
            The initial real part of the amplitude, one value per site or one for all.

        v:
            The initial imaginary part of the amplitude, one value per site or one for all.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import GinzburgLandauElement
        >>> GinzburgLandauElement('site', u=0.5).u
        0.5
    """

    def __init__(self, label: str, *, u: object = 0.0, v: object = 0.0, space: BaseSpace | None = None) -> None:
        super().__init__(label, space=space)
        self.u = u
        self.v = v


class GinzburgLandauInteraction(Interaction):
    """The complex Ginzburg-Landau dynamics `A_t = A + (1 + i b) laplacian(A) - (1 - i c) |A|^2 A` of an amplitude.

    The amplitude `A = u + i v` of the target is held as its real and imaginary parts. Without the Laplacian every site
    is a Stuart-Landau oscillator, and the dispersion `b` and the nonlinear frequency shift `c` set whether plane waves
    are stable or break into spiral defect chaos.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the field, with the real part `u` and the imaginary part `v`.

        b:
            The linear dispersion.

        c:
            The nonlinear frequency shift.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import GinzburgLandauInteraction
        >>> GinzburgLandauInteraction('dynamics', target='medium').c
        -1.5
    """

    law_classes = (RateLaw,)

    def __init__(self, label: str, *, target: Role, b: float = 1.0, c: float = -1.5) -> None:
        super().__init__(label)
        self.target = target
        self.b = b
        self.c = c

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rates of the real and imaginary parts of the amplitude of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        s, t = context, context.t
        u, v = s.target.u, s.target.v
        square = u**2 + v**2
        self.laws_ = [
            RateLaw(u.diff(t), u + laplacian(u) - s.b * laplacian(v) - square * (u - s.c * v)),
            RateLaw(v.diff(t), v + s.b * laplacian(u) + laplacian(v) - square * (s.c * u + v)),
        ]


class PatternFieldElement(Element):
    """Sites of a real field `u` whose patterns a free energy selects.

    Args:
        label:
            The label of the field.

        u:
            The initial value of the field, one value per site or one for all.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import PatternFieldElement
        >>> PatternFieldElement('site', u=0.1).u
        0.1
    """

    def __init__(self, label: str, *, u: object = 0.0, space: BaseSpace | None = None) -> None:
        super().__init__(label, space=space)
        self.u = u


class SwiftHohenbergEnergyInteraction(GradientInteraction):
    """Gradient flow of a field in the Swift-Hohenberg free energy.

    The flow is `u_t = r u - (1 + laplacian)^2 u + g u^2 - u^3`.

    The free energy is `F = integral -r u^2 / 2 + ((1 + laplacian) u)^2 / 2 - g u^3 / 3 + u^4 / 4`. Above onset,
    `r > 0`, the modes of wavenumber near 1 grow into rolls, or into hexagons for a large enough quadratic term `g`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

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
        >>> energy = SwiftHohenbergEnergyInteraction('energy', target='cells')
        >>> namespace = SimpleNamespace(target=SimpleNamespace(u=var('u')), r=var('r'), g=0)
        >>> [law.value for law in energy.prepare(namespace).laws_]
        [-u^3 + r*u - u - 2*laplacian(u) - laplacian(laplacian(u))]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str = 'u',
        r: float = 0.3,
        g: float = 0.0,
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

        solver:
            The solver of the problem, or `'auto'` for exponential steps of 0.1.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import ComplexGinzburgLandauSystem
        >>> ComplexGinzburgLandauSystem((8, 8)).prepare().system_.interactions[0].c
        -1.5
    """

    field = 'physics/pattern_formation'
    topics = ('field', 'chaos', 'spirals', 'spectral', 'pde')
    references = ('Aranson and Kramer 2002', 'Gros 2015')
    until = 50.0
    dt = 1.0

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.length = length
        self.b = b
        self.c = c
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the complex Ginzburg-Landau equation from a small random amplitude.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = tuple(self.shape)
        rng = np.random.default_rng(self.random_state)
        u = ComplexGinzburgLandauSystem._get_noise(shape, 0.01, rng)
        v = ComplexGinzburgLandauSystem._get_noise(shape, 0.01, rng)
        space = LatticeSpace(shape, spacing=self.length / shape[0], discretization='spectral')
        medium = GinzburgLandauElement('medium', u=u.ravel(), v=v.ravel(), space=space)
        dynamics = GinzburgLandauInteraction('dynamics', target='medium', b=self.b, c=self.c)
        self.system_ = System(self.label, elements=[medium], interactions=[dynamics], solver=ETDRK4Solver(dt=0.1))


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

        solver:
            The solver of the problem, or `'auto'` for exponential steps of 0.5.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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

    def __init__(
        self,
        shape: Sequence[int] = (64, 64),
        *,
        length: float = 16 * math.pi,
        r: float = 0.3,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.shape = shape
        self.length = length
        self.r = r
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the Swift-Hohenberg equation from small noise.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        shape = tuple(self.shape)
        u = ComplexGinzburgLandauSystem._get_noise(shape, 0.01, self.random_state)
        space = LatticeSpace(shape, spacing=self.length / shape[0], discretization='spectral')
        self.system_ = System(
            self.label,
            elements=[PatternFieldElement('cells', u=u.ravel(), space=space)],
            interactions=[SwiftHohenbergEnergyInteraction('energy', target='cells', r=self.r)],
            solver=ETDRK4Solver(dt=0.5),
        )
