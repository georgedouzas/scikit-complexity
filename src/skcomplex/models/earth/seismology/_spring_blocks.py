"""Spring-block model of a fault driven by a moving plate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.base import Domain, state
from ....core.model import Element, Field, LatticeSpace, Law, Rate, Symbols, System, both, laplacian, where
from ....core.runtime import register_model


def _spring_block_laws(s: Symbols) -> list[Law]:
    """Return the sliding of a block under its springs and the stick-slip friction that weakens with speed."""
    sliding = where(s.v > 0, s.v, 0)
    force = s.k_c * laplacian(s.x) + s.k_p * (s.velocity * s.t - s.x)
    friction = s.F0 / (1 + sliding / s.v_c)
    stuck = both(s.v <= 0, force < s.F0)
    return [Rate(s.x, sliding), Rate(s.v, where(stuck, 0, (force - friction) / s.m))]


SpringBlock = Element.define(
    'SpringBlock',
    states={'x': state(default=0.0, domain=Domain.REAL), 'v': state(default=0.0, domain=Domain.REAL)},
    parameters={'m': 1.0, 'k_c': 2.5, 'k_p': 1.0, 'velocity': 0.01, 'F0': 1.0, 'v_c': 0.5},
    laws=_spring_block_laws,
)


@register_model(
    field='earth/seismology',
    tags=('field', 'stick-slip', 'earthquakes', 'friction'),
    references=('Burridge and Knopoff 1967', 'Carlson and Langer 1989', 'Charbonneau 2017'),
    defaults={'until': 200.0, 'dt': 1.0},
)
def burridge_knopoff(
    n_blocks: int = 20,
    *,
    k_c: float = 2.5,
    k_p: float = 1.0,
    velocity: float = 0.01,
    F0: float = 1.0,
    v_c: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Burridge-Knopoff fault, a chain of blocks pulled by a moving plate that sticks and slips.

    Every block is coupled to its neighbours by springs of stiffness `k_c` and to a plate moving at `velocity` by a
    leaf spring of stiffness `k_p`. A block at rest stays stuck until its spring force exceeds the static friction `F0`,
    and a sliding block feels the friction `F0 / (1 + v / v_c)`, which weakens with its speed, so slips spread as
    earthquakes of many sizes. The ends of the chain are free, and the blocks start at small random offsets.

    Args:
        n_blocks:
            The number of blocks.

        k_c:
            The stiffness of the springs between neighbouring blocks.

        k_p:
            The stiffness of the leaf springs to the plate.

        velocity:
            The velocity of the plate.

        F0:
            The static friction force.

        v_c:
            The speed at which the friction halves.

        random_state:
            The seed or generator of the initial offsets.

    Returns:
        system:
            The model, with the positions `fault.x` and velocities `fault.v` of the blocks.

    Examples:
        >>> from skcomplex.models.earth.seismology import burridge_knopoff
        >>> burridge_knopoff(5).elements[0].size
        5
    """
    offsets = np.random.default_rng(random_state).uniform(-0.5, 0.5, n_blocks)
    fault = Field(
        'fault',
        SpringBlock,
        LatticeSpace(n_blocks, boundary='reflecting'),
        boundary={'x': 'zero-flux', 'v': 'zero-flux'},
        values={'x': offsets, 'k_c': k_c, 'k_p': k_p, 'velocity': velocity, 'F0': F0, 'v_c': v_c},
    )
    return System(elements=[fault])
