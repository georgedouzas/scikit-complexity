"""Hydrodynamically coupled spheres.

Two Brownian spheres in blinking optical traps share one fluid.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MobilityInteraction
from ....core.components.laws.utils import pulse_train
from ....core.components.spaces import ContinuousSpace, DiscreteTime
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

SPHERES = VisualSpec("points", {"position": "x", "radius": 1.0})


class TrappedSphereElement(Element):
    """Colloidal sphere at the position `x`, held by an optical trap.

    The trap is centred at `center`.

    The sphere has no friction of its own. A mobility interaction moves it through the fluid it shares with the other
    spheres, and its trap is one of the forces the mobility turns into velocities.

    Args:
        label:
            The label of the sphere or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        center:
            The centre of the trap of the sphere.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import TrappedSphereElement
        >>> TrappedSphereElement('bead', center=[3.0, 0.0]).center
        [3.0, 0.0]
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = SPHERES,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role="position")] = 0.0,
        center: Annotated[npt.ArrayLike, ParameterSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.center = center


class BlinkingTrapsInteraction(MobilityInteraction):
    """Blinking optical traps on spheres that move through a shared fluid.

    The spheres move in steps of Brownian dynamics.

    Every sphere feels the harmonic force `-k (x - center)` of its own trap while the traps are on, for `on_time` of
    every period `on_time + off_time`, and no force while they are off. The mobility turns the forces on all spheres
    into their velocities, and the thermal energy `temperature` adds Brownian displacements whose correlations between
    spheres follow the same mobility, so spheres released from their traps diffuse together (Batchelor 1976). Every
    step lasts `dt`, and the time of the system counts steps.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of spheres.

        k:
            The stiffness of every trap.

        on_time:
            The time the traps stay on in every period.

        off_time:
            The time the traps stay off in every period.

        tensor:
            `'free-draining'`, `'oseen'`, or `'rotne-prager-yamakawa'`.

        root:
            `'cholesky'` or `'lanczos'`.

        radius:
            The radius of the spheres.

        viscosity:
            The viscosity of the fluid.

        temperature:
            The thermal energy `kT`.

        dt:
            The length of a step.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import BlinkingTrapsInteraction
        >>> BlinkingTrapsInteraction('tweezers', target='beads', off_time=0.2).form
        'map'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        on_time: float = 1.0,
        off_time: float = 0.0,
        tensor: str = "rotne-prager-yamakawa",
        root: str = "cholesky",
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        viscosity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        dt: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(
            label,
            target=target,
            tensor=tensor,
            form="map",
            force=self._get_trap_force,
            root=root,
            radius=radius,
            viscosity=viscosity,
            temperature=temperature,
            dt=dt,
        )
        self.k = k
        self.on_time = on_time
        self.off_time = off_time

    def _get_trap_force(self, namespace: SimpleNamespace) -> FreeModuleElement:
        """Return the force of the trap of a sphere.

        The trap is switched off for `off_time` of every period.
        """
        s = namespace
        on = 1 if self.off_time == 0 else pulse_train(s.t * s.dt, self.on_time + self.off_time, self.on_time)
        return -s.k * on * (s.target.x - s.target.center)


class HydrodynamicallyCoupledTrapsSystem(BaseSystem):
    """Two Brownian spheres in blinking optical traps.

    The free motion of the spheres is correlated through the fluid.

    The traps hold the spheres at a distance `separation` along `x`, and switch off for `off_time` after every
    `on_time`. While they are off the spheres diffuse freely, and the force on one moves the other through the mobility,
    so their displacements correlate as `M_12 / M_11`. That is about `3 a / (2 r)` along the line of centres and `3 a /
    (4 r)` across it. The units are micrometres, seconds, and piconewtons, so the defaults are spheres of radius 1 um in
    water at room temperature, and each step of the system lasts `step` seconds. The system has the positions
    `spheres.x`.

    Args:
        separation:
            The distance between the centres of the traps.

        tensor:
            `'free-draining'`, `'oseen'`, or `'rotne-prager-yamakawa'`.

        root:
            `'cholesky'` or `'lanczos'`.

        radius:
            The radius of the spheres.

        viscosity:
            The viscosity of the fluid.

        kT:
            The thermal energy.

        k:
            The stiffness of the traps.

        on_time:
            The time the traps stay on in every period.

        off_time:
            The time the traps stay off in every period.

        step:
            The length of a step, in seconds.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import HydrodynamicallyCoupledTrapsSystem
        >>> HydrodynamicallyCoupledTrapsSystem(4.0).prepare().system_.elements[0].values['center']
        [[-2.0, 0.0], [2.0, 0.0]]
    """

    field = "physics/soft_matter"
    topics = ("brownian motion", "hydrodynamics", "optical tweezers", "blinking traps", "correlated noise")
    references = ("Argun et al. 2021", "Batchelor 1976", "Rotne and Prager 1969", "Magazzù et al. 2019")
    until = 20000
    dt = 1

    def __init__(
        self,
        separation: float = 3.0,
        *,
        tensor: str = "rotne-prager-yamakawa",
        root: str = "cholesky",
        radius: float = 1.0,
        viscosity: float = 1e-3,
        kT: float = 4.1e-3,
        k: float = 0.2,
        on_time: float = 0.3,
        off_time: float = 0.1,
        step: float = 5e-3,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.separation = separation
        self.tensor = tensor
        self.root = root
        self.radius = radius
        self.viscosity = viscosity
        self.kT = kT
        self.k = k
        self.on_time = on_time
        self.off_time = off_time
        self.step = step

    def _build_system(self) -> System:
        """Return two Brownian spheres in blinking optical traps.

        The free motion of the spheres is correlated through the fluid.
        """
        centers = [[-self.separation / 2, 0.0], [self.separation / 2, 0.0]]
        spheres = GroupElement(
            "spheres",
            element=TrappedSphereElement,
            size=2,
            values={"x": centers, "center": centers},
        )
        traps = BlinkingTrapsInteraction(
            "tweezers",
            target="spheres",
            k=self.k,
            on_time=self.on_time,
            off_time=self.off_time,
            tensor=self.tensor,
            root=self.root,
            radius=self.radius,
            viscosity=self.viscosity,
            temperature=self.kT,
            dt=self.step,
        )
        return System(
            self.label,
            elements=[spheres],
            interactions=[traps],
            space=ContinuousSpace(n_dim=2),
            time=DiscreteTime(),
        )
