"""String systems.

The module covers chains of masses on springs, guitar strings, and standing waves on a continuous string.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, oo, sin, sqrt, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import BaseSpace, ContinuousSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, VelocityVerletSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from ...general.gradients import GradientInteraction
from ..common.particles import MassPointElement


class WaveStringElement(Element):
    """Sites of a string, with the transverse displacement `u`.

    Args:
        label:
            The label of the string.

        u:
            The initial displacement, one value or one per site.

        space:
            The lattice of the sites, or `None` for one site.

    Examples:
        >>> from skcomplex.systems.physics.waves import WaveStringElement
        >>> WaveStringElement('site', u=0.5).u
        0.5
    """

    def __init__(self, label: str, *, u: object = 0.0, space: BaseSpace | None = None) -> None:
        super().__init__(label, space=space)
        self.u = u


class WaveTensionInteraction(Interaction):
    """Tension of a string, which accelerates its displacement as the wave equation `u'' = c^2 laplacian(u)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the string.

        c:
            The wave speed.

    Examples:
        >>> from skcomplex.systems.physics.waves import WaveTensionInteraction
        >>> WaveTensionInteraction('tension', target='string', c=2.0).c
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        c: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.c = c

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the acceleration of the displacement of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            u = context.target.u
            self.laws_ = [RateLaw(u.diff(context.t, 2), context.c**2 * laplacian(u))]


class PotentialEnergyInteraction(GradientInteraction):
    """Potential energy of the positions of particles.

    Its force on every particle changes its velocity.

    It is the gradient of the energy with the mass as divisor: the force `-dU/dx` on every particle gives the second
    derivative of its position of order 2, the force divided by its mass. A subclass binds one role per particle and
    returns the energy from `_get_potential`, or a user passes the energy as `potential`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or particle group.

        potential:
            A function of the namespace that returns the energy, or `None` for a subclass that returns it.

        states:
            The name of the vector state that holds the position of every particle.

        divisor:
            The name of the state or parameter that holds the mass of every particle.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var, vector
        >>> from skcomplex.systems.physics.waves import PotentialEnergyInteraction
        >>> ball = SimpleNamespace(x=vector([var('x')]), m=2)
        >>> trap = PotentialEnergyInteraction('trap', target='ball', potential=lambda s: s.target.x[0] ** 2)
        >>> [law.value for law in trap.prepare(SimpleNamespace(target=ball)).laws_]
        [(-x)]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        potential: Callable[[SimpleNamespace], object] | None = None,
        states: str | Sequence[str] = 'x',
        divisor: str = 'm',
    ) -> None:
        super().__init__(label, target=target, states=states, potential=potential, divisor=divisor, order=2)


class HarmonicPotentialInteraction(PotentialEnergyInteraction):
    """Harmonic potential that pulls a particle towards a center.

    The energy is `k * |x - center|**2 / 2`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        k:
            The stiffness.

        center:
            The center. A number applies to every component.

        states:
            The name of the vector state that holds the position of the particle.

        divisor:
            The name of the state or parameter that holds the mass of the particle.

    Examples:
        >>> from skcomplex.systems.physics.waves import HarmonicPotentialInteraction
        >>> HarmonicPotentialInteraction('trap', target='ball', k=2.0).k
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        k: Annotated[float, RealSet.open(0, oo)] = 1.0,
        center: npt.ArrayLike = 0.0,
        states: str = 'x',
        divisor: str = 'm',
    ) -> None:
        super().__init__(label, target=target, states=states, divisor=divisor)
        self.k = k
        self.center = center

    def _get_center(self, s: SimpleNamespace) -> object:
        """Return the center, the hook a subclass overrides to move it."""
        return s.center

    def _get_potential(self, s: SimpleNamespace) -> Expression:
        """Return `k * |x - center|**2 / 2`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        position, center = getattr(s.target, self._get_names()[0]), self._get_center(s)
        if not isinstance(center, FreeModuleElement):
            center = vector(SR, [center] * len(position))
        displacement = position - center
        return s.k * displacement.dot_product(displacement) / 2


class SpringInteraction(PotentialEnergyInteraction):
    """Spring between two particles.

    The energy is `k * (|x_first - x_second| - length)**2 / 2`, and its gradient moves both particles.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the first particle.

        second:
            The path of the second particle.

        k:
            The stiffness.

        length:
            The rest length.

        states:
            The name of the vector state that holds the positions of the particles.

        divisor:
            The name of the state or parameter that holds the masses of the particles.

    Examples:
        >>> from skcomplex.systems.physics.waves import SpringInteraction
        >>> SpringInteraction('spring', first='a', second='b', k=3.0, length=1.0).length
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        first: Role,
        second: Role,
        k: Annotated[float, RealSet.open(0, oo)] = 1.0,
        length: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        states: str = 'x',
        divisor: str = 'm',
    ) -> None:
        super().__init__(label, target=first, states=states, divisor=divisor)
        self.first = first
        self.second = second
        self.k = k
        self.length = length

    def _get_potential(self, s: SimpleNamespace) -> Expression:
        """Return `k * (|x_first - x_second| - length)**2 / 2`.

        Args:
            s:
                The namespace of the symbols the energy can use.

        Returns:
            energy:
                The potential energy as a symbolic expression.
        """
        gap = getattr(s.first, self._get_names()[0]) - getattr(s.second, self._get_names()[0])
        return s.k * (sqrt(gap.dot_product(gap)) - s.length) ** 2 / 2


class DrivenAnchorInteraction(HarmonicPotentialInteraction):
    """Spring that ties a particle to an anchor point.

    The anchor oscillates transversally as `amplitude sin(omega t)`.

    The energy is `k |x - center - (0, amplitude sin(omega t))|² / 2`, and a zero amplitude gives a fixed anchor.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle.

        k:
            The stiffness of the spring.

        center:
            The rest position of the anchor.

        amplitude:
            The amplitude of the transverse motion of the anchor.

        omega:
            The angular frequency of the anchor.

    Examples:
        >>> from skcomplex.systems.physics.waves import DrivenAnchorInteraction
        >>> DrivenAnchorInteraction('end', target='mass_0', center=[0.0, 0.0], amplitude=0.1).amplitude
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        k: Annotated[float, RealSet.open(0, oo)] = 1.0,
        center: npt.ArrayLike = 0.0,
        amplitude: float = 0.0,
        omega: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label, target=target, k=k, center=center)
        self.amplitude = amplitude
        self.omega = omega

    def _get_center(self, s: SimpleNamespace) -> Expression:
        """Return the position `center + (0, amplitude sin(omega t))` of the anchor."""
        return s.center + vector(SR, [0, s.amplitude * sin(s.omega * s.t)])


class SpringChainSystem(BaseSystem):
    """Masses joined in a row by springs of zero rest length between two anchors.

    The chain is a discrete string.

    The masses start evenly spaced between the anchors at `0` and `length`, stretched, so the chain carries tension.
    With `pluck`, the masses start on a triangle of that height peaking at the fraction `pluck_at` of the length, the
    pluck of a guitar string. With `amplitude`, the left anchor moves transversally as `amplitude sin(omega t)` and
    sends a transverse wave along the chain. Springs of zero rest length are linear, so the transverse modes are
    independent, with frequencies `2 sqrt(k / m) sin(n pi / (2 (N + 1)))`. The system has the masses `mass_<i>` and
    their positions `mass_<i>.x` in two dimensions.

    Args:
        n_masses:
            The number of masses between the anchors.

        length:
            The distance between the anchors.

        k:
            The stiffness of every spring.

        mass:
            The mass of every mass.

        pluck:
            The initial transverse displacement of the plucked point.

        pluck_at:
            The plucked point, as a fraction of the length.

        amplitude:
            The amplitude of the transverse motion of the left anchor.

        omega:
            The angular frequency of the left anchor.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for velocity Verlet with the step 0.01.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.waves import SpringChainSystem
        >>> len(SpringChainSystem(4).prepare().system_.interactions)
        5
    """

    field = 'physics/waves'
    topics = ('waves', 'ode', 'springs', 'normal modes', 'mechanics')
    references = ('Argun et al. 2021',)
    until = 20.0
    dt = 0.05

    def __init__(
        self,
        n_masses: int = 15,
        *,
        length: float = 1.0,
        k: float = 1.0,
        mass: float = 0.01,
        pluck: float = 0.05,
        pluck_at: float = 0.5,
        amplitude: float = 0.0,
        omega: float = 2.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_masses = n_masses
        self.length = length
        self.k = k
        self.mass = mass
        self.pluck = pluck
        self.pluck_at = pluck_at
        self.amplitude = amplitude
        self.omega = omega

    @staticmethod
    def _pluck(positions: np.ndarray, length: float, height: float, at: float) -> np.ndarray:
        """Return the triangular shape of a plucked string.

        The string is pulled to a height at a fraction of its length.
        """
        peak = at * length
        return np.where(positions <= peak, height * positions / peak, height * (length - positions) / (length - peak))

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to masses joined in a row by springs of zero rest length between two anchors.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        spacing = self.length / (self.n_masses + 1)
        positions = spacing * np.arange(1, self.n_masses + 1)
        heights = (
            self._pluck(positions, self.length, self.pluck, self.pluck_at) if self.pluck else np.zeros(self.n_masses)
        )
        masses = [
            MassPointElement(
                f'mass_{index}',
                x=[float(positions[index]), float(heights[index])],
                m=self.mass,
            )
            for index in range(self.n_masses)
        ]
        springs = [
            SpringInteraction(f'spring_{index}', first=f'mass_{index}', second=f'mass_{index + 1}', k=self.k)
            for index in range(self.n_masses - 1)
        ]
        anchors = [
            DrivenAnchorInteraction(
                'left_end',
                target='mass_0',
                k=self.k,
                center=[0.0, 0.0],
                amplitude=self.amplitude,
                omega=self.omega,
            ),
            DrivenAnchorInteraction(
                'right_end',
                target=f'mass_{self.n_masses - 1}',
                k=self.k,
                center=[self.length, 0.0],
            ),
        ]
        self.system_ = System(
            self.label,
            elements=masses,
            interactions=[*anchors, *springs],
            space=ContinuousSpace(n_dim=2),
            solver=VelocityVerletSolver(dt=0.01),
        )


class StandingWavesSystem(BaseSystem):
    """A string with fixed ends on `[-length / 2, length / 2]`.

    The string vibrates as a standing wave.

    The displacement follows the wave equation `u'' = c² laplacian(u)` on a lattice whose edges hold the string at
    zero. The string starts at rest in the mode `amplitude sin(mode pi (x +
    length / 2) / length)`, which oscillates in place with the period `2 length / (mode c)`, up to the discretization.
    The system has the displacement `string.u` and the velocity `string.u'`.

    Args:
        n_sites:
            The number of sites inside the string.

        length:
            The length of the string.

        c:
            The speed of the waves.

        mode:
            The number of half wavelengths along the string.

        amplitude:
            The initial amplitude.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.waves import StandingWavesSystem
        >>> trajectory = StandingWavesSystem(9).prepare().simulate(until=0.1)
        >>> trajectory.get_lattice('string.u').shape
        (3, 9)
    """

    field = 'physics/waves'
    topics = ('waves', 'field', 'pde', 'normal modes')
    references = ('Starkovich 2021',)
    until = 4.0
    dt = 0.05

    def __init__(
        self,
        n_sites: int = 99,
        *,
        length: float = 2.0,
        c: float = 1.0,
        mode: int = 1,
        amplitude: float = 1.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_sites = n_sites
        self.length = length
        self.c = c
        self.mode = mode
        self.amplitude = amplitude

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a string with fixed ends that vibrates as a standing wave.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        spacing = self.length / (self.n_sites + 1)
        positions = spacing * np.arange(1, self.n_sites + 1)
        shape = self.amplitude * np.sin(self.mode * math.pi * positions / self.length)
        space = LatticeSpace(self.n_sites, spacing=spacing, boundary='fixed', fill=0.0)
        self.system_ = System(
            self.label,
            elements=[WaveStringElement('string', u=shape, space=space)],
            interactions=[WaveTensionInteraction('tension', target='string', c=self.c)],
        )
