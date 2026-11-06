"""Blinking optical traps on spheres that move through a shared fluid, in steps of Brownian dynamics."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ....core.components.interactions import MobilityInteraction
from ....core.components.laws.utils import pulse_train
from ....core.specs import ParameterSpec, RoleSpec


class BlinkingTrapsInteraction(MobilityInteraction):
    """Blinking optical traps on spheres that move through a shared fluid, in steps of Brownian dynamics.

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
        tensor: str = 'rotne-prager-yamakawa',
        root: str = 'cholesky',
        radius: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        viscosity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        temperature: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        dt: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(
            label,
            target=target,
            tensor=tensor,
            form='map',
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
        """Return the force of the trap of a sphere, switched off for `off_time` of every period."""
        s = namespace
        on = 1 if self.off_time == 0 else pulse_train(s.t * s.dt, self.on_time + self.off_time, self.on_time)
        return -s.k * on * (s.target.x - s.target.center)
