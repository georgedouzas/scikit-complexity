"""Spring-block system of a fault driven by a moving plate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import both, laplacian, where
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, RK4Solver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class SpringBlockElement(Element):
    """Blocks of the Burridge-Knopoff fault, their displacements and velocities.

    Args:
        label:
            The label of the blocks.

        x:
            The initial displacement, one value or one per block.

        v:
            The initial velocity, one value or one per block.

        space:
            The lattice of the blocks, or `None` for one block.

    Examples:
        >>> from skcomplex.systems.physics.seismology import SpringBlockElement
        >>> SpringBlockElement('block', x=0.5).x
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        x: object = 0.0,
        v: object = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.x = x
        self.v = v


class StickSlipInteraction(Interaction):
    """Stick-slip friction of blocks pulled by springs and a moving plate.

    The force on a block is the pull `k_c` of the springs to its neighbours plus the pull `k_p` of the plate moving at
    `velocity`. A block at rest sticks until the force exceeds the static friction `F0`, and a sliding block feels the
    friction `F0 / (1 + v / v_c)`, so `x' = max(v, 0)` and `m v' = force - friction` while the block slides.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the blocks.

        m:
            The mass of a block.

        k_c:
            The stiffness of the springs between neighbouring blocks.

        k_p:
            The stiffness of the springs to the plate.

        velocity:
            The speed of the plate.

        F0:
            The static friction.

        v_c:
            The characteristic speed of the weakening of the friction.

    Examples:
        >>> from skcomplex.systems.physics.seismology import StickSlipInteraction
        >>> StickSlipInteraction('friction', target='fault').F0
        1.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        m: float = 1.0,
        k_c: float = 2.5,
        k_p: float = 1.0,
        velocity: float = 0.01,
        F0: float = 1.0,
        v_c: float = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.m = m
        self.k_c = k_c
        self.k_p = k_p
        self.velocity = velocity
        self.F0 = F0
        self.v_c = v_c

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the sliding of the blocks under their springs and the stick-slip friction.

        Args:
            context:
                The namespace of the blocks, the time `t`, and the parameters.
        """
        super()._prepare(context)
        x, v, t = context.target.x, context.target.v, context.t
        sliding = where(v > 0, v, 0)
        force = context.k_c * laplacian(x) + context.k_p * (context.velocity * t - x)
        friction = context.F0 / (1 + sliding / context.v_c)
        stuck = both(v <= 0, force < context.F0)
        self.laws_ = [
            RateLaw(x.diff(t), sliding),
            RateLaw(v.diff(t), where(stuck, 0, (force - friction) / context.m)),
        ]


class BurridgeKnopoffSystem(BaseSystem):
    """The Burridge-Knopoff fault.

    A chain of blocks pulled by a moving plate sticks and slips.

    Every block is coupled to its neighbours by springs of stiffness `k_c` and to a plate moving at `velocity` by a
    leaf spring of stiffness `k_p`. A block at rest stays stuck until its spring force exceeds the static friction `F0`,
    and a sliding block feels the friction `F0 / (1 + v / v_c)`, which weakens with its speed, so slips spread as
    earthquakes of many sizes. The ends of the chain are free, and the blocks start at small random offsets.

    The drift switches whenever a block sticks or starts to slide, and these switches of single blocks give no events,
    so the adaptive default solver stalls on them: LSODA gives way to RK45 after its stall guard fires, and 2000 time
    units take about 24 s. The system runs with `RK4Solver(dt=0.05)`, well inside the stability limit set by the
    highest frequency `sqrt(k_p + 4 k_c)`, which runs the same span in about 3.5 s with the same statistics of slips.
    The system has the positions `fault.x` and velocities `fault.v` of the blocks.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.seismology import BurridgeKnopoffSystem
        >>> trajectory = BurridgeKnopoffSystem(5).prepare().simulate(until=1.0)
        >>> trajectory.get_values('fault.x').shape
        (2, 5)
    """

    field = 'physics/seismology'
    topics = ('field', 'stick-slip', 'earthquakes', 'friction')
    references = ('Burridge and Knopoff 1967', 'Carlson and Langer 1989', 'Charbonneau 2017')
    until = 200.0
    dt = 1.0

    def __init__(
        self,
        n_blocks: int = 20,
        *,
        k_c: float = 2.5,
        k_p: float = 1.0,
        velocity: float = 0.01,
        F0: float = 1.0,
        v_c: float = 0.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_blocks = n_blocks
        self.k_c = k_c
        self.k_p = k_p
        self.velocity = velocity
        self.F0 = F0
        self.v_c = v_c
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a chain of blocks pulled by a moving plate.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        offsets = np.random.default_rng(self.random_state).uniform(-0.5, 0.5, self.n_blocks)
        fault = SpringBlockElement('fault', x=offsets, space=LatticeSpace(self.n_blocks, boundary='reflecting'))
        friction = StickSlipInteraction(
            'friction',
            target='fault',
            k_c=self.k_c,
            k_p=self.k_p,
            velocity=self.velocity,
            F0=self.F0,
            v_c=self.v_c,
        )
        self.system_ = System(self.label, elements=[fault], interactions=[friction], solver=RK4Solver(dt=0.05))
