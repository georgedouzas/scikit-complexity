"""Spring-block system of a fault driven by a moving plate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.components.elements import Element, FieldElement
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import both, laplacian, where
from ....core.components.spaces import LatticeSpace
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class SpringBlockElement(Element):
    """Block of the Burridge-Knopoff fault, pulled by springs and a moving plate against a friction that weakens.

    The force on a block is the pull `k_c` of the springs to its neighbours plus the pull `k_p` of the plate moving at
    `velocity`. A block at rest sticks until the force exceeds the static friction `F0`, and a sliding block feels the
    friction `F0 / (1 + v / v_c)`. The stick condition reads the whole force, the coupling included, so the block keeps
    its own rate laws.

    Args:
        label:
            The label of the block or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial displacement.

        v:
            The initial velocity.

        m:
            The mass.

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
        >>> from skcomplex.systems.physics.seismology import SpringBlockElement
        >>> SpringBlockElement('block').F0
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        v: Annotated[float, StateSpec()] = 0.0,
        m: Annotated[float, ParameterSpec()] = 1.0,
        k_c: Annotated[float, ParameterSpec()] = 2.5,
        k_p: Annotated[float, ParameterSpec()] = 1.0,
        velocity: Annotated[float, ParameterSpec()] = 0.01,
        F0: Annotated[float, ParameterSpec()] = 1.0,
        v_c: Annotated[float, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.v = v
        self.m = m
        self.k_c = k_c
        self.k_p = k_p
        self.velocity = velocity
        self.F0 = F0
        self.v_c = v_c

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the sliding of the block under its springs and the stick-slip friction that weakens with speed.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the displacement and the velocity.
        """
        sliding = where(s.v > 0, s.v, 0)
        force = s.k_c * laplacian(s.x) + s.k_p * (s.velocity * s.t - s.x)
        friction = s.F0 / (1 + sliding / s.v_c)
        stuck = both(s.v <= 0, force < s.F0)
        return [RateLaw(s.x, sliding), RateLaw(s.v, where(stuck, 0, (force - friction) / s.m))]


@register_system(
    field='physics/seismology',
    tags=('field', 'stick-slip', 'earthquakes', 'friction'),
    references=('Burridge and Knopoff 1967', 'Carlson and Langer 1989', 'Charbonneau 2017'),
    defaults={'until': 200.0, 'dt': 1.0, 'solver': RK4Solver(dt=0.05)},
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

    The drift switches whenever a block sticks or starts to slide, and these switches of single blocks give no events,
    so the adaptive default solver stalls on them: LSODA gives way to RK45 after its stall guard fires, and 2000 time
    units take about 24 s. The registered solver is `RK4Solver(dt=0.05)`, well inside the stability limit set by the
    highest frequency `sqrt(k_p + 4 k_c)`, which runs the same span in about 3.5 s with the same statistics of slips.

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
            The system, with the positions `fault.x` and velocities `fault.v` of the blocks.

    Examples:
        >>> from skcomplex.systems.physics.seismology import burridge_knopoff
        >>> burridge_knopoff(5).elements[0].prepare().size_
        5
    """
    offsets = np.random.default_rng(random_state).uniform(-0.5, 0.5, n_blocks)
    fault = FieldElement(
        'fault',
        element=SpringBlockElement,
        space=LatticeSpace(n_blocks, boundary='reflecting'),
        boundary={'x': 'zero-flux', 'v': 'zero-flux'},
        values={'x': offsets, 'k_c': k_c, 'k_p': k_p, 'velocity': velocity, 'F0': F0, 'v_c': v_c},
    )
    return System(elements=[fault])
