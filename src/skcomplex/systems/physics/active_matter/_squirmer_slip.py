"""Slip of spherical squirmers, the B1 and B2 modes that make them swim and stir each other through the mobility."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, identity_matrix, matrix, oo
from sage.matrix.matrix0 import Matrix
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ....core.components.interactions import MobilityInteraction
from ....core.specs import ParameterSpec, RoleSpec


class SquirmerSlipInteraction(MobilityInteraction):
    """Slip of spherical squirmers, the B1 and B2 modes that make them swim and stir each other through the mobility.

    A squirmer of radius `a` with the tangential surface slip `B1 sin(theta) + B2 sin(theta) cos(theta)` about its
    orientation `p` (Lighthill 1952, Blake 1971) swims at `2 B1 / 3` along `p`. Far from it, the slip stirs the fluid
    as the stresslet `4 pi eta a^2 B2 (p p - I / 3)` and the potential dipole `B1 a^3 p / 3` (Ishikawa, Simmonds, and
    Pedley 2006). So `beta = B2 / B1 < 0` makes a pusher, which pushes fluid out along its axis, and `beta > 0` a
    puller, which draws it in. The flows of the other squirmers move every squirmer and turn it by half their vorticity,
    through `MobilityInteraction`. A Weeks-Chandler-Andersen repulsion of strength `epsilon` keeps the spheres apart.

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
        >>> SquirmerSlipInteraction('slip', target='squirmers', B2=-5.0).cutoff
        2.244924...
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
            orientation='p',
            velocity=self._get_swimming,
            stresslet=self._get_stresslet_of,
            dipole=self._get_dipole_of,
            potential=self._get_repulsion,
            cutoff=2 ** (7 / 6) * radius,
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

    def _get_repulsion(self, namespace: SimpleNamespace, r: Expression) -> Expression:
        """Return the Weeks-Chandler-Andersen repulsion of spheres of diameter `2 a`, valid within the cutoff."""
        s = namespace
        ratio = 2 * s.radius / r
        return 4 * s.epsilon * (ratio**12 - ratio**6) + s.epsilon
