"""Squirmers, spherical microswimmers driven by surface slip.

The squirmers are coupled through the flows they stir in a Stokes fluid.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, identity_matrix, matrix, oo
from sage.matrix.matrix0 import Matrix
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MobilityInteraction
from ....core.components.spaces import ContinuousSpace
from ....core.components.systems import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

POINTS = VisualSpec("points", {"position": "x", "radius": 1.0})


class SquirmerElement(Element):
    """Spherical squirmer.

    The microswimmer at the position `x` swims along the unit orientation `p`.

    The slip of its surface sets how it swims and stirs the fluid, and `SquirmerSlipInteraction` turns the slip of
    every squirmer into the motion of all of them.

    Args:
        label:
            The label of the squirmer or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        p:
            The initial orientation, a unit vector.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import SquirmerElement
        >>> SquirmerElement('swimmer', p=[0.0, 0.0, 1.0]).p
        [0.0, 0.0, 1.0]
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = POINTS,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role="position")] = 0.0,
        p: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.p = p


class SquirmerSlipInteraction(MobilityInteraction):
    """Slip of spherical squirmers.

    The B1 and B2 modes make them swim and stir each other through the mobility.

    A squirmer of radius `a` with the tangential surface slip `B1 sin(theta) + B2 sin(theta) cos(theta)` about its
    orientation `p` (Lighthill 1952, Blake 1971) swims at `2 B1 / 3` along `p`. Far from it, the slip stirs the fluid
    as the stresslet `4 pi eta a^2 B2 (p p - I / 3)` and the potential dipole `B1 a^3 p / 3` (Ishikawa, Simmonds, and
    Pedley 2006). So `beta = B2 / B1 < 0` makes a pusher, which pushes fluid out along its axis, and `beta > 0` a
    puller, which draws it in. The flows of the other squirmers move every squirmer and turn it by half their vorticity,
    through `MobilityInteraction`. A Weeks-Chandler-Andersen repulsion of strength `epsilon` keeps the spheres apart,
    within `2^(1/6)` diameters of the current radius.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of squirmers.

        B1:
            The first slip mode, which sets the swimming speed `2 B1 / 3`.

        B2:
            The second slip mode, negative for pushers and positive for pullers.

        epsilon:
            The strength of the repulsion between spheres closer than `2^(1/6)` diameters.

        radius:
            The radius of the squirmers.

        viscosity:
            The viscosity of the fluid.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import SquirmerSlipInteraction
        >>> SquirmerSlipInteraction('slip', target='squirmers', B2=-5.0).B2
        -5.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        B1: Annotated[float, ParameterSpec()] = 1.0,
        B2: Annotated[float, ParameterSpec()] = 0.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        viscosity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            orientation="p",
            velocity=self._get_swimming,
            stresslet=self._get_stresslet_of,
            dipole=self._get_dipole_of,
            potential=self._get_repulsion,
            radius=radius,
            viscosity=viscosity,
        )
        self.B1 = B1
        self.B2 = B2
        self.epsilon = epsilon

    def _get_swimming(self, namespace: SimpleNamespace) -> FreeModuleElement:
        """Return the swimming velocity `2 B1 p / 3`."""
        return 2 * namespace.B1 * namespace.target.p / 3

    def _get_stresslet_of(self, namespace: SimpleNamespace) -> Matrix:
        """Return the stresslet `4 pi eta a^2 B2 (p p - I / 3)`."""
        s = namespace
        heading = matrix(s.target.p).transpose()
        strength = 4 * math.pi * s.viscosity * s.radius**2 * s.B2
        return strength * (heading * heading.transpose() - identity_matrix(len(s.target.p)) / 3)

    def _get_dipole_of(self, namespace: SimpleNamespace) -> FreeModuleElement:
        """Return the potential dipole `B1 a^3 p / 3`."""
        return namespace.B1 * namespace.radius**3 * namespace.target.p / 3

    def _get_cutoff(self, namespace: SimpleNamespace) -> Expression:
        """Return the range `2^(1/6) 2 a` of the repulsion, from the current radius."""
        return 2 ** (7 / 6) * namespace.radius

    def _get_repulsion(self, namespace: SimpleNamespace, r: Expression) -> Expression:
        """Return the Weeks-Chandler-Andersen repulsion of spheres of diameter `2 a`.

        The repulsion is valid within the cutoff.
        """
        s = namespace
        ratio = 2 * s.radius / r
        return 4 * s.epsilon * (ratio**12 - ratio**6) + s.epsilon


class SquirmersSystem(BaseSystem):
    """Spherical squirmers that swim at `2 B1 / 3`.

    The squirmers stir each other as pushers or pullers.

    The squirmers start on a jittered cubic lattice of the spacing in an unbounded fluid, with random orientations.
    `beta = B2 / B1` sets the stresslet: pushers (`beta < 0`) draw their neighbours in from the sides and pullers (`beta
    > 0`) push them away. The system has the positions `squirmers.x` and orientations `squirmers.p`.

    Args:
        n_squirmers:
            The number of squirmers.

        B1:
            The first slip mode.

        beta:
            The ratio `B2 / B1` of the slip modes.

        radius:
            The radius of the squirmers.

        viscosity:
            The viscosity of the fluid.

        spacing:
            The spacing of the starting lattice.

        random_state:
            The seed or generator of the starting jitter and orientations.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import SquirmersSystem
        >>> SquirmersSystem(8, beta=2.0).prepare().system_.interactions[0].B2
        3.0
    """

    field = "physics/active_matter"
    topics = ("active matter", "microswimmers", "squirmers", "hydrodynamics", "stokes flow", "pushers", "pullers")
    references = (
        "Lighthill 1952",
        "Blake 1971",
        "Ishikawa, Simmonds, and Pedley 2006",
        "te Vrugt and Wittkowski 2024",
    )
    until = 40.0
    dt = 0.5

    def __init__(
        self,
        n_squirmers: int = 27,
        *,
        B1: float = 1.5,
        beta: float = -3.0,
        radius: float = 1.0,
        viscosity: float = 1.0,
        spacing: float = 5.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.n_squirmers = n_squirmers
        self.B1 = B1
        self.beta = beta
        self.radius = radius
        self.viscosity = viscosity
        self.spacing = spacing
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return spherical squirmers that swim at `2 B1 / 3`.

        The squirmers stir each other as pushers or pullers.
        """
        rng = np.random.default_rng(self.random_state)
        side = math.ceil(self.n_squirmers ** (1 / 3) - 1e-9)
        grid = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)[: self.n_squirmers]
        positions = self.spacing * grid + rng.uniform(-0.1, 0.1, size=(self.n_squirmers, 3)) * self.spacing
        orientations = rng.standard_normal((self.n_squirmers, 3))
        orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)
        group = GroupElement(
            "squirmers",
            element=SquirmerElement,
            size=self.n_squirmers,
            values={"x": positions, "p": orientations},
        )
        slip = SquirmerSlipInteraction(
            "slip",
            target="squirmers",
            B1=self.B1,
            B2=self.beta * self.B1,
            radius=self.radius,
            viscosity=self.viscosity,
        )
        return System(self.label, elements=[group], interactions=[slip], space=ContinuousSpace(n_dim=3))
