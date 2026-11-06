"""Squirmers: spherical microswimmers driven by surface slip, coupled through the flows they stir in a Stokes fluid."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ..._catalog import register_system
from ._squirmer import SquirmerElement
from ._squirmer_slip import SquirmerSlipInteraction


@register_system(
    field='physics/active_matter',
    tags=('active matter', 'microswimmers', 'squirmers', 'hydrodynamics', 'stokes flow', 'pushers', 'pullers'),
    references=(
        'Lighthill 1952',
        'Blake 1971',
        'Ishikawa, Simmonds, and Pedley 2006',
        'te Vrugt and Wittkowski 2024',
    ),
    defaults={'until': 40.0, 'dt': 0.5},
)
def squirmers(
    n_squirmers: int = 27,
    *,
    B1: float = 1.5,
    beta: float = -3.0,
    radius: float = 1.0,
    viscosity: float = 1.0,
    spacing: float = 5.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return spherical squirmers that swim at `2 B1 / 3` and stir each other as pushers or pullers.

    The squirmers start on a jittered cubic lattice of the spacing in an unbounded fluid, with random orientations.
    `beta = B2 / B1` sets the stresslet: pushers (`beta < 0`) draw their neighbours in from the sides and pullers
    (`beta > 0`) push them away.

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

    Returns:
        system:
            The system, with the positions `squirmers.x` and orientations `squirmers.p`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import squirmers
        >>> squirmers(8, beta=2.0).interactions[0].B2
        3.0
    """
    rng = np.random.default_rng(random_state)
    side = math.ceil(n_squirmers ** (1 / 3) - 1e-9)
    grid = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing='ij'), axis=-1).reshape(-1, 3)[:n_squirmers]
    positions = spacing * grid + rng.uniform(-0.1, 0.1, size=(n_squirmers, 3)) * spacing
    orientations = rng.standard_normal((n_squirmers, 3))
    orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)
    group = GroupElement(
        'squirmers',
        element=SquirmerElement,
        size=n_squirmers,
        values={'x': positions, 'p': orientations},
    )
    slip = SquirmerSlipInteraction(
        'slip',
        target='squirmers',
        B1=B1,
        B2=beta * B1,
        radius=radius,
        viscosity=viscosity,
    )
    return System(elements=[group], interactions=[slip], space=ContinuousSpace(n_dim=3))
