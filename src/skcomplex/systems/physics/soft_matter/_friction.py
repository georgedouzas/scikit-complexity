"""Dry friction.

Shaken chains of frictional blocks relax to blocked configurations.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw
from ....core.components.laws.utils import both, laplacian, pulse_train, where
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class FrictionalBlockElement(Element):
    """Blocks of a chain on a rough floor, at the positions `x` with the velocities `v`.

    Args:
        label:
            The label of the blocks.

        x:
            The initial position, one per block or shared.

        v:
            The initial velocity, one per block or shared.

        m:
            The mass.

        space:
            The lattice of the chain, or `None` for one block.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import FrictionalBlockElement
        >>> FrictionalBlockElement('block', m=2.0).m
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        x: object = 0.0,
        v: object = 0.0,
        m: Annotated[float, RealSet.open(0, oo)] = 1.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.x = x
        self.v = v
        self.m = m


class ShakenFrictionInteraction(Interaction):
    """Springs, Coulomb friction, and periodic shaking of a chain of blocks.

    The springs of stiffness `k` between neighbours and an anchor spring pull every block, Coulomb friction holds it
    while the force stays below `mu_s` and slows it by `mu_d` while it slides, and a random force of amplitude
    `amplitude` shakes it during the first `shake_time` of every `shake_time + relax_time`. It is a granular pile that
    is tapped, or a fault whose blocks stick and slip.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the blocks, whose positions are `x`, velocities `v`, and mass `m`.

        k:
            The stiffness of the springs between neighbours.

        anchor:
            The stiffness of the spring to the rest position.

        mu_s:
            The static friction force.

        mu_d:
            The dynamic friction force.

        amplitude:
            The amplitude of the shaking force.

        shake_time:
            The duration of every shaking.

        relax_time:
            The time between shakings.

        v_stick:
            The speed below which a block can stick.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ShakenFrictionInteraction
        >>> ShakenFrictionInteraction('friction', target='chain', mu_s=2.0).mu_s
        2.0
    """

    law_classes = (RateLaw, NoiseLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        k: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        anchor: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        mu_s: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        mu_d: Annotated[float, RealSet.closed_open(0, oo)] = 0.8,
        amplitude: Annotated[float, RealSet.closed_open(0, oo)] = 2.0,
        shake_time: Annotated[float, RealSet.open(0, oo)] = 2.0,
        relax_time: Annotated[float, RealSet.open(0, oo)] = 8.0,
        v_stick: Annotated[float, RealSet.open(0, oo)] = 0.02,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.k = k
        self.anchor = anchor
        self.mu_s = mu_s
        self.mu_d = mu_d
        self.amplitude = amplitude
        self.shake_time = shake_time
        self.relax_time = relax_time
        self.v_stick = v_stick

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the positions and the velocities of the blocks, and the shaking of the velocities.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            x, v, m = s.target.x, s.target.v, s.target.m
            force = s.k * laplacian(x) - s.anchor * x
            moving = abs(v) > s.v_stick
            direction = where(moving, where(v > 0, 1, -1), where(force > 0, 1, -1))
            stuck = both(abs(v) <= s.v_stick, abs(force) < s.mu_s)
            shaking = pulse_train(t, s.shake_time + s.relax_time, s.shake_time)
            self.laws_ = [
                RateLaw(x.diff(t), where(moving, v, 0)),
                RateLaw(v.diff(t), where(stuck, 0, (force - s.mu_d * direction) / m)),
                NoiseLaw(v, s.amplitude * shaking / m),
            ]


class ShakenSpringBlockSystem(BaseSystem):
    """A chain of blocks joined by springs on a frictional substrate.

    The chain is shaken and relaxed in turn.

    Every block of unit weight feels the springs to its neighbours, `k (x_{i+1} + x_{i-1} - 2 x_i)`, an optional spring
    of stiffness `anchor` to its rest position, and Coulomb friction. A sliding block feels the dynamic friction `-mu_d
    sign(v)`, and a block at rest stays at rest while the spring force on it is below the static friction `mu_s`. In
    every cycle a random force of intensity `amplitude^2` shakes the blocks for `shake_time`, and then the chain relaxes
    for `relax_time` to a blocked configuration, where the elongations `xi_i = x_{i+1} - x_i` satisfy `k |xi_{i+1} -
    xi_i| < mu_s`. Repeated cycles sample blocked configurations, whose energy per spring grows with the shaking, and
    whose elongations are correlated over a length that grows with that energy. Without springs between blocks and with
    an anchor, the blocks are independent, and their blocked positions satisfy `|x_i| < mu_s / anchor`. A block counts
    as at rest below the speed `v_stick`, which must exceed the change of speed `(mu_s + mu_d) h` of a solver step `h`.
    The ends of the chain are free, and the blocks start at rest. The system has the positions `chain.x` and velocities
    `chain.v` of the blocks.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ShakenSpringBlockSystem
        >>> ShakenSpringBlockSystem(5).prepare().simulate(until=10.0).get_values('chain.x').shape
        (2, 5)
    """

    field = 'physics/soft_matter'
    topics = ('field', 'sde', 'friction', 'granular matter', 'blocked states')
    references = ('Bertin 2021', 'Edwards and Oakeshott 1989', 'Bertin, Dauchot, and Droz 2004')
    until = 200.0
    dt = 10.0

    def __init__(
        self,
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
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_blocks = n_blocks
        self.k = k
        self.anchor = anchor
        self.mu_s = mu_s
        self.mu_d = mu_d
        self.amplitude = amplitude
        self.shake_time = shake_time
        self.relax_time = relax_time
        self.v_stick = v_stick

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a chain of blocks joined by springs on a frictional substrate.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        space = LatticeSpace(self.n_blocks, boundary='reflecting')
        friction = ShakenFrictionInteraction(
            'friction',
            target='chain',
            k=self.k,
            anchor=self.anchor,
            mu_s=self.mu_s,
            mu_d=self.mu_d,
            amplitude=self.amplitude,
            shake_time=self.shake_time,
            relax_time=self.relax_time,
            v_stick=self.v_stick,
        )
        self.system_ = System(
            self.label,
            elements=[FrictionalBlockElement('chain', x=np.zeros(self.n_blocks), space=space)],
            interactions=[friction],
            solver=EulerMaruyamaSolver(dt=0.01),
        )
