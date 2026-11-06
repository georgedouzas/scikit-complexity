"""Hydrodynamically coupled spheres: two Brownian spheres in blinking optical traps that share one fluid."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.components.elements import GroupElement
from ....core.components.spaces import ContinuousSpace, DiscreteTime
from ....core.components.system import System
from ..._catalog import register_system
from ._blinking_traps import BlinkingTrapsInteraction
from ._trapped_sphere import TrappedSphereElement


@register_system(
    field='physics/soft_matter',
    tags=('brownian motion', 'hydrodynamics', 'optical tweezers', 'blinking traps', 'correlated noise'),
    references=('Argun et al. 2021', 'Batchelor 1976', 'Rotne and Prager 1969', 'Magazzù et al. 2019'),
    defaults={'until': 20000, 'dt': 1},
)
def hydrodynamically_coupled_traps(
    separation: float = 3.0,
    *,
    tensor: str = 'rotne-prager-yamakawa',
    root: str = 'cholesky',
    radius: float = 1.0,
    viscosity: float = 1e-3,
    kT: float = 4.1e-3,
    k: float = 0.2,
    on_time: float = 0.3,
    off_time: float = 0.1,
    dt: float = 5e-3,
) -> System:
    """Return two Brownian spheres in blinking optical traps, whose free motion is correlated through the fluid.

    The traps hold the spheres at a distance `separation` along `x`, and switch off for `off_time` after every
    `on_time`. While they are off the spheres diffuse freely, and the force on one moves the other through the
    mobility, so their displacements correlate as `M_12 / M_11`. That is about `3 a / (2 r)` along the line of centres
    and `3 a / (4 r)` across it. The units are micrometres, seconds, and piconewtons, so the defaults are spheres of
    radius 1 um in water at room temperature, and each step of the system lasts `dt` seconds.

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

        dt:
            The length of a step.

    Returns:
        system:
            The system, with the positions `spheres.x`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import hydrodynamically_coupled_traps
        >>> hydrodynamically_coupled_traps(4.0).elements[0].values['center']
        [[-2.0, 0.0], [2.0, 0.0]]
    """
    centers = [[-separation / 2, 0.0], [separation / 2, 0.0]]
    spheres = GroupElement(
        'spheres',
        element=TrappedSphereElement,
        size=2,
        values={'x': centers, 'center': centers},
    )
    traps = BlinkingTrapsInteraction(
        'tweezers',
        target='spheres',
        k=k,
        on_time=on_time,
        off_time=off_time,
        tensor=tensor,
        root=root,
        radius=radius,
        viscosity=viscosity,
        temperature=kT,
        dt=dt,
    )
    return System(elements=[spheres], interactions=[traps], space=ContinuousSpace(n_dim=2), time=DiscreteTime())
