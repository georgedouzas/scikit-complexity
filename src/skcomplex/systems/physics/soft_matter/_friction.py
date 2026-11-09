"""Dry friction.

Shaken chains of frictional blocks relax to blocked configurations.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.components.laws.utils import both, laplacian, pulse_train, where
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class FrictionalBlockElement(Element):
    """Block of a chain on a rough floor, joined to its neighbours.

    The block is at the position `x` with the velocity `v`.

    The springs of stiffness `k` and an anchor spring pull the block, Coulomb friction holds it while the force stays
    below `mu_s` and slows it by `mu_d` while it slides, and periodic shaking of amplitude `amplitude` kicks it during
    the first `shake_time` of every `shake_time + relax_time`.

    Args:
        label:
            The label of the block.

        visual:
            How the block is drawn, or `None` for none.

        x:
            The initial position.

        v:
            The initial velocity.

        m:
            The mass.

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
            The speed below which the block can stick.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import FrictionalBlockElement
        >>> FrictionalBlockElement('block', mu_s=2.0).mu_s
        2.0
    """

    law_classes = (
        NoiseLaw,
        RateLaw,
    )

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        v: Annotated[float, StateSpec()] = 0.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        anchor: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        mu_s: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        mu_d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        amplitude: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        shake_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        relax_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 8.0,
        v_stick: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.02,
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
        """Return the springs, the sticking Coulomb friction, and the shaking of the block.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the position and the velocity, and the noise law of the shaking.
        """
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

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ShakenSpringBlockSystem
        >>> ShakenSpringBlockSystem(5).prepare().system_.elements[0].prepare().size_
        5
    """

    field = "physics/soft_matter"
    topics = ("field", "sde", "friction", "granular matter", "blocked states")
    references = ("Bertin 2021", "Edwards and Oakeshott 1989", "Bertin, Dauchot, and Droz 2004")
    until = 200.0
    dt = 10.0
    solver = EulerMaruyamaSolver(dt=0.01)

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
        label: str = "system",
    ) -> None:
        super().__init__(label)
        self.n_blocks = n_blocks
        self.k = k
        self.anchor = anchor
        self.mu_s = mu_s
        self.mu_d = mu_d
        self.amplitude = amplitude
        self.shake_time = shake_time
        self.relax_time = relax_time
        self.v_stick = v_stick

    def _build_system(self) -> System:
        """Return a chain of blocks joined by springs on a frictional substrate.

        The chain is shaken and relaxed in turn.
        """
        chain = FieldElement(
            "chain",
            element=FrictionalBlockElement,
            space=LatticeSpace(self.n_blocks, boundary="reflecting"),
            boundary={"x": "zero-flux", "v": "zero-flux"},
            values={
                "x": np.zeros(self.n_blocks),
                "k": self.k,
                "anchor": self.anchor,
                "mu_s": self.mu_s,
                "mu_d": self.mu_d,
                "amplitude": self.amplitude,
                "shake_time": self.shake_time,
                "relax_time": self.relax_time,
                "v_stick": self.v_stick,
            },
        )
        return System(self.label, elements=[chain])
