"""Dry friction: shaken chains of frictional blocks that relax to blocked configurations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, FieldElement
from ....core.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.laws.utils import both, laplacian, pulse_train, where
from ....core.spaces import LatticeSpace
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.solvers import EulerMaruyamaSolver
from ..._catalog import register_model


def _frictional_block_laws(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the springs, the Coulomb friction with sticking, and the shaking of a block of the chain."""
    force = s.k * laplacian(s.x) - s.anchor * s.x
    moving = abs(s.v) > s.v_stick
    direction = where(moving, where(s.v > 0, 1, -1), where(force > 0, 1, -1))
    stuck = both(abs(s.v) <= s.v_stick, abs(force) < s.mu_s)
    shaking = pulse_train(s.t, s.shake_time + s.relax_time, s.shake_time)
    return [
        RateLaw(s.x, where(moving, s.v, 0)),
        RateLaw(s.v, where(stuck, 0, (force - s.mu_d * direction) / s.m)),
        NoiseLaw(s.v, s.amplitude * shaking / s.m),
    ]


class FrictionalBlockElement(Element):
    """Frictional block element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        v: Annotated[object, StateSpec()] = 0.0,
        m: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        k: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        anchor: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        mu_s: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        mu_d: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        amplitude: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        shake_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        relax_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 8.0,
        v_stick: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 0.02,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.m = m
        self.k = k
        self.anchor = anchor
        self.mu_s = mu_s
        self.mu_d = mu_d
        self.amplitude = amplitude
        self.shake_time = shake_time
        self.relax_time = relax_time
        self.v_stick = v_stick

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the frictional block element."""
        return _frictional_block_laws(s)


@register_model(
    field='physics/soft_matter',
    tags=('field', 'sde', 'friction', 'granular matter', 'blocked states'),
    references=('Bertin 2021', 'Edwards and Oakeshott 1989', 'Bertin, Dauchot, and Droz 2004'),
    defaults={'until': 200.0, 'dt': 10.0, 'solver': EulerMaruyamaSolver(dt=0.01)},
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
        >>> shaken_spring_block(5).elements[0].check().prepare().size_
        5
    """
    chain = FieldElement(
        'chain',
        element=FrictionalBlockElement,
        space=LatticeSpace(n_blocks, boundary='reflecting'),
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
