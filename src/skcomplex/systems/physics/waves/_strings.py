"""String systems.

The module covers chains of masses on springs, guitar strings, and standing waves on a continuous string.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, Expression, oo, sin, vector
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import ContinuousSpace, LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import VelocityVerletSolver
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.particles import HarmonicPotentialInteraction, MassPointElement, SpringInteraction


class WaveStringElement(Element):
    """Site of a string, with the transverse displacement `u` and the wave speed `c`.

    The displacement has order 2.

    Args:
        label:
            The label of the site.

        visual:
            How the site is drawn, or `None` for none.

        u:
            The initial displacement.

        u_d1:
            The initial transverse velocity.

        c:
            The wave speed.

    Examples:
        >>> from skcomplex.systems.physics.waves import WaveStringElement
        >>> WaveStringElement('site', c=2.0).c
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec(order=2)] = 0.0,
        u_d1: float = 0.0,
        c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.u_d1 = u_d1
        self.c = c


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
        target: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        center: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
        amplitude: Annotated[float, ParameterSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
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

    Examples:
        >>> from skcomplex.systems.physics.waves import SpringChainSystem
        >>> len(SpringChainSystem(4).prepare().system_.interactions)
        5
    """

    field = "physics/waves"
    topics = ("waves", "ode", "springs", "normal modes", "mechanics")
    references = ("Argun et al. 2021",)
    until = 20.0
    dt = 0.05
    solver = VelocityVerletSolver(dt=0.01)

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
        label: str = "system",
    ) -> None:
        super().__init__(label)
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

    def _build_system(self) -> System:
        """Return masses joined in a row by zero rest length springs between two anchors.

        The chain is a discrete string.
        """
        spacing = self.length / (self.n_masses + 1)
        positions = spacing * np.arange(1, self.n_masses + 1)
        heights = (
            self._pluck(positions, self.length, self.pluck, self.pluck_at) if self.pluck else np.zeros(self.n_masses)
        )
        masses = [
            MassPointElement(
                f"mass_{index}",
                x=[float(positions[index]), float(heights[index])],
                x_d1=[0.0, 0.0],
                m=self.mass,
            )
            for index in range(self.n_masses)
        ]
        springs = [
            SpringInteraction(f"spring_{index}", first=f"mass_{index}", second=f"mass_{index + 1}", k=self.k)
            for index in range(self.n_masses - 1)
        ]
        anchors = [
            DrivenAnchorInteraction(
                "left_end",
                target="mass_0",
                k=self.k,
                center=[0.0, 0.0],
                amplitude=self.amplitude,
                omega=self.omega,
            ),
            DrivenAnchorInteraction(
                "right_end",
                target=f"mass_{self.n_masses - 1}",
                k=self.k,
                center=[self.length, 0.0],
            ),
        ]
        return System(self.label, elements=masses, interactions=[*anchors, *springs], space=ContinuousSpace(n_dim=2))


class StandingWavesSystem(BaseSystem):
    """A string with fixed ends on `[-length / 2, length / 2]`.

    The string vibrates as a standing wave.

    The displacement follows the wave equation `u'' = c² u_xx`, written as `u' = v` and `v' = c² laplacian(u)` on a
    lattice whose edges hold the string at zero. The string starts at rest in the mode `amplitude sin(mode pi (x +
    length / 2) / length)`, which oscillates in place with the period `2 length / (mode c)`, up to the discretization.
    The system has the displacement `string.u` and the velocity `string.v`.

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

    Examples:
        >>> from skcomplex.systems.physics.waves import StandingWavesSystem
        >>> StandingWavesSystem(9).prepare().system_.elements[0].prepare().size_
        9
    """

    field = "physics/waves"
    topics = ("waves", "field", "pde", "normal modes")
    references = ("Starkovich 2021",)
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
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.n_sites = n_sites
        self.length = length
        self.c = c
        self.mode = mode
        self.amplitude = amplitude

    def _build_system(self) -> System:
        """Return a string with fixed ends on `[-length / 2, length / 2]`.

        The string vibrates as a standing wave.
        """
        spacing = self.length / (self.n_sites + 1)
        positions = spacing * np.arange(1, self.n_sites + 1)
        shape = self.amplitude * np.sin(self.mode * math.pi * positions / self.length)
        space = LatticeSpace(self.n_sites, spacing=spacing)
        edges = ("fixed", 0.0)
        string = FieldElement(
            "string",
            element=WaveStringElement,
            space=space,
            boundary={"u": edges},
            values={"u": shape, "c": self.c},
        )
        tension = RateInteraction(
            "tension",
            target="string",
            state="u",
            rate=lambda s: s.target.c**2 * laplacian(s.target.u),
        )
        return System(self.label, elements=[string], interactions=[tension])
