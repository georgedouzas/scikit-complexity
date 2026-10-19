"""Dry friction: shaken chains of frictional blocks that relax to blocked configurations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.base import Domain, parameter
from ....core.model import (
    Element,
    Field,
    LatticeSpace,
    Law,
    Noise,
    Rate,
    Symbols,
    System,
    both,
    laplacian,
    pulse_train,
    where,
)
from ....core.runtime import register_model
from ....core.solvers import EulerMaruyama


def _frictional_block_laws(s: Symbols) -> list[Law]:
    """Return the springs, the Coulomb friction with sticking, and the shaking of a block of the chain."""
    force = s.k * laplacian(s.x) - s.anchor * s.x
    moving = abs(s.v) > s.v_stick
    direction = where(moving, where(s.v > 0, 1, -1), where(force > 0, 1, -1))
    stuck = both(abs(s.v) <= s.v_stick, abs(force) < s.mu_s)
    shaking = pulse_train(s.t, s.shake_time + s.relax_time, s.shake_time)
    return [
        Rate(s.x, where(moving, s.v, 0)),
        Rate(s.v, where(stuck, 0, (force - s.mu_d * direction) / s.m)),
        Noise(s.v, s.amplitude * shaking / s.m),
    ]


FrictionalBlock = Element.define(
    'FrictionalBlock',
    states={'x': 0.0, 'v': 0.0},
    parameters={
        'm': parameter(default=1.0, domain=Domain.POSITIVE),
        'k': parameter(default=1.0, domain=Domain.NONNEGATIVE),
        'anchor': parameter(default=0.0, domain=Domain.NONNEGATIVE),
        'mu_s': parameter(default=1.0, domain=Domain.NONNEGATIVE),
        'mu_d': parameter(default=0.8, domain=Domain.NONNEGATIVE),
        'amplitude': parameter(default=2.0, domain=Domain.NONNEGATIVE),
        'shake_time': parameter(default=2.0, domain=Domain.POSITIVE),
        'relax_time': parameter(default=8.0, domain=Domain.POSITIVE),
        'v_stick': parameter(default=0.02, domain=Domain.POSITIVE),
    },
    laws=_frictional_block_laws,
)


@register_model(
    field='physics/soft_matter',
    tags=('field', 'sde', 'friction', 'granular matter', 'blocked states'),
    references=('Bertin 2021', 'Edwards and Oakeshott 1989', 'Bertin, Dauchot, and Droz 2004'),
    defaults={'until': 200.0, 'dt': 10.0, 'solver': EulerMaruyama(dt=0.01)},
)
def shaken_spring_block(
    n_blocks: int = 50,
    *,
    k: float = 1.0,
    anchor: float = 0.0,
    mu_s: float = 1.0,
    mu_d: float = 0.8,
    amplitude: float = 2.0,
    shake_time: float = 2.0,
    relax_time: float = 8.0,
    v_stick: float = 0.02,
) -> System:
    """Return a chain of blocks joined by springs on a frictional substrate, shaken and relaxed in turn.

    Every block of unit weight feels the springs to its neighbours, `k (x_{i+1} + x_{i-1} - 2 x_i)`, an optional
    spring of stiffness `anchor` to its rest position, and Coulomb friction. A sliding block feels the dynamic friction
    `-mu_d sign(v)`, and a block at rest stays at rest while the spring force on it is below the static friction
    `mu_s`. In every cycle a random force of intensity `amplitude^2` shakes the blocks for `shake_time`, and then the
    chain relaxes for `relax_time` to a blocked configuration, where the elongations `xi_i = x_{i+1} - x_i` satisfy
    `k |xi_{i+1} - xi_i| < mu_s`. Repeated cycles sample blocked configurations, whose energy per spring grows with
    the shaking, and whose elongations are correlated over a length that grows with that energy. Without springs
    between blocks and with an anchor, the blocks are independent, and their blocked positions satisfy
    `|x_i| < mu_s / anchor`. A block counts as at rest below the speed `v_stick`, which must exceed the change of
    speed `(mu_s + mu_d) h` of a solver step `h`. The ends of the chain are free, and the blocks start at rest.

    Args:
        n_blocks:
            The number of blocks.

        k:
            The stiffness of the springs between neighbouring blocks.

        anchor:
            The stiffness of the springs that hold every block to its rest position.

        mu_s:
            The static friction coefficient.

        mu_d:
            The dynamic friction coefficient.

        amplitude:
            The amplitude of the random shaking force.

        shake_time:
            The duration of the shaking in every cycle.

        relax_time:
            The duration of the relaxation in every cycle.

        v_stick:
            The speed below which a block counts as at rest.

    Returns:
        system:
            The model, with the positions `chain.x` and velocities `chain.v` of the blocks.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import shaken_spring_block
        >>> shaken_spring_block(5).elements[0].size
        5
    """
    chain = Field(
        'chain',
        FrictionalBlock,
        LatticeSpace(n_blocks, boundary='reflecting'),
        boundary={'x': 'zero-flux', 'v': 'zero-flux'},
        values={
            'x': np.zeros(n_blocks),
            'k': k,
            'anchor': anchor,
            'mu_s': mu_s,
            'mu_d': mu_d,
            'amplitude': amplitude,
            'shake_time': shake_time,
            'relax_time': relax_time,
            'v_stick': v_stick,
        },
    )
    return System(elements=[chain])
