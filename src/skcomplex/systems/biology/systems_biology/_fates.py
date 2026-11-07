"""Systems biology systems: cell fate decisions as landscapes and as lateral inhibition between cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import total
from ....core.components.system import System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class FateLandscapeElement(Element):
    """Cell state in the compactified elliptic umbilic, the minimal landscape of a decision among three fates.

    The state `(x, y)` descends the potential `f = x^4 + y^4 + x^3 - 2 x y^2 + a (x^2 + y^2) + theta1 x + theta2 y` of
    Rand et al. (2021, figure 5D), `dx/dt = -df/dx` and `dy/dt = -df/dy`. The signals `theta1` and `theta2` change
    in time at the rates `v1` and `v2`, and the sign of `a` makes the central rest point an attractor or a repellor.

    Args:
        label:
            The label of the cell or of the group of cells.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        a:
            The curvature of the potential at the centre.

        theta1:
            The first signal at the start.

        theta2:
            The second signal at the start.

        v1:
            The rate of change of the first signal.

        v2:
            The rate of change of the second signal.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import FateLandscapeElement
        >>> FateLandscapeElement('cell', a=-0.2).a
        -0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
        a: Annotated[float, ParameterSpec()] = -0.2,
        theta1: Annotated[float, ParameterSpec()] = 0.0,
        theta2: Annotated[float, ParameterSpec()] = 0.0,
        v1: Annotated[float, ParameterSpec()] = 0.0,
        v2: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.a = a
        self.theta1 = theta1
        self.theta2 = theta2
        self.v1 = v1
        self.v2 = v2

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the gradient descent of the potential under the drifting signals.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        theta1, theta2 = s.theta1 + s.v1 * s.t, s.theta2 + s.v2 * s.t
        return [
            RateLaw(s.x, -(4 * s.x**3 + 3 * s.x**2 - 2 * s.y**2 + 2 * s.a * s.x + theta1)),
            RateLaw(s.y, -(4 * s.y**3 - 4 * s.x * s.y + 2 * s.a * s.y + theta2)),
        ]


class ProneuralCellElement(Element):
    """Cell of an epithelium that adopts the neural fate when the inhibition from its neighbours is weak.

    The state `a` runs from 0, the epidermal fate, to 1, the neural fate of a sensory bristle. The sigmoids of the
    lateral inhibition share the width `width`.

    Args:
        label:
            The label of the cell or of the group of cells.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        a:
            The initial neural state.

        width:
            The width of the sigmoids.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import ProneuralCellElement
        >>> ProneuralCellElement('cell', a=0.1).a
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.127,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.width = width


class LateralInhibitionInteraction(Interaction):
    """Lateral inhibition of the neural fate of cells by their neighbours, through a kernel of the distance.

    Following Eq. (2) of Rand et al. (2021), `da_i/dt = sigma(a_i - h_i) - a_i` with the inhibition `h_i = sum_{j != i}
    K_ij d(a_j)`, where `sigma(u) = 1 / (1 + exp(-u / width))` and `d(a) = sigma(a - 1 / 2)`. A cell under an
    inhibition below 0.37 can only turn neural, above 0.63 only epidermal, and in between it is bistable.

    Args:
        label:
            The label of the interaction.

        cells:
            The path of the group of cells.

        kernel:
            The inhibition `K_ij` of every cell by every other, with a zero diagonal.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import LateralInhibitionInteraction
        >>> LateralInhibitionInteraction('inhibition', cells='cells', kernel=((0.0, 1.0), (1.0, 0.0))).kernel
        ((0.0, 1.0), (1.0, 0.0))
    """

    def __init__(
        self,
        label: str,
        *,
        cells: Annotated[str, RoleSpec()],
        kernel: Sequence[Sequence[float]] = ((0.0,),),
    ) -> None:
        super().__init__(label)
        self.cells = cells
        self.kernel = kernel

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the relaxation of every cell towards the fate its inhibition allows.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        cell = s.cells
        inhibition = total(self._sigmoid(cell.a - 1 / 2, cell.width), weights=np.asarray(self.kernel, dtype=float))
        return [RateLaw(cell.a, self._sigmoid(cell.a - inhibition, cell.width) - cell.a)]

    def _sigmoid(self, u: Expression, width: Expression) -> Expression:
        """Return the logistic sigmoid `1 / (1 + exp(-u / width))`."""
        return 1 / (1 + exp(-u / width))


@register_system(
    field='biology/systems_biology',
    tags=('cell fate', 'morse-smale', 'elliptic umbilic', 'landscape', 'ode'),
    references=('Rand et al. 2021',),
    defaults={'until': 40.0, 'dt': 0.2},
)
def three_way_fate_decision(
    cells: int = 30,
    *,
    a: float = -0.2,
    theta1: float = 0.0,
    theta2: float = 0.0,
    v1: float = 0.0,
    v2: float = 0.0,
    spread: float = 0.6,
    seed: int | None = 0,
) -> System:
    """Return cells that choose among three fates on the compactified elliptic umbilic landscape of Rand et al.

    Every cell descends the potential `x^4 + y^4 + x^3 - 2 x y^2 + a (x^2 + y^2) + theta1 x + theta2 y`. With `a < 0`
    and no signal the centre is a repellor and three attractors, the fates, lie around it, separated by three saddles,
    so cells that start near the centre split among the three fates. Signals that drift at the rates `v1` and `v2`
    move the system across the fold curves of its parameter space, where a fate disappears and its cells switch to
    another.

    Args:
        cells:
            The number of cells.

        a:
            The curvature of the potential at the centre.

        theta1:
            The first signal at the start.

        theta2:
            The second signal at the start.

        v1:
            The rate of change of the first signal.

        v2:
            The rate of change of the second signal.

        spread:
            The half-width of the uniform initial states around the centre.

        seed:
            The seed of the initial states.

    Returns:
        system:
            The system, with the states `cells.x` and `cells.y`.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import three_way_fate_decision
        >>> three_way_fate_decision(5).elements[0].prepare().size_
        5
    """
    rng = np.random.default_rng(seed)
    start = rng.uniform(-spread, spread, (2, cells))
    values = {'x': start[0], 'y': start[1], 'a': a, 'theta1': theta1, 'theta2': theta2, 'v1': v1, 'v2': v2}
    return System(elements=[GroupElement('cells', element=FateLandscapeElement, size=cells, values=values)])


@register_system(
    field='biology/systems_biology',
    tags=('lateral inhibition', 'pattern formation', 'bristles', 'notch-delta', 'ode'),
    references=('Corson et al. 2017', 'Rand et al. 2021'),
    defaults={'until': 60.0, 'dt': 0.5},
)
def bristle_lateral_inhibition(
    cells: int = 40,
    *,
    strength: float = 1.0,
    length: float = 2.5,
    width: float = 0.127,
    noise: float = 0.05,
    seed: int | None = 0,
) -> System:
    """Return a ring of cells that inhibit the neural fate of their neighbours, leaving sparse, evenly spaced bristles.

    Following Eq. (2) of Rand et al. (2021), every cell relaxes to `sigma(a_i - h_i)` under the inhibition `h_i = sum_j
    K_ij d(a_j)`, with the Gaussian kernel `K_ij = strength exp(-d_ij^2 / (2 length^2))` of the distance `d_ij` along
    the ring. All cells start near the epidermal state and rise together, until small differences break the symmetry
    at a saddle on the diagonal: the leading cells turn neural and silence their neighbours, so the neural cells end
    several cell diameters apart.

    Args:
        cells:
            The number of cells on the ring.

        strength:
            The inhibition between neighbouring cells at zero distance.

        length:
            The range of the inhibition, in cell diameters.

        width:
            The width of the sigmoids.

        noise:
            The largest initial neural state, drawn uniformly from zero.

        seed:
            The seed of the initial states.

    Returns:
        system:
            The system, with the neural states `cells.a`.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import bristle_lateral_inhibition
        >>> bristle_lateral_inhibition(10).elements[0].prepare().size_
        10
    """
    index = np.arange(cells)
    distance = np.abs(index[:, None] - index[None, :])
    distance = np.minimum(distance, cells - distance)
    kernel = strength * np.exp(-(distance**2) / (2 * length**2))
    np.fill_diagonal(kernel, 0.0)
    states = np.random.default_rng(seed).uniform(0, noise, cells)
    group = GroupElement('cells', element=ProneuralCellElement, size=cells, values={'a': states, 'width': width})
    inhibition = LateralInhibitionInteraction('inhibition', cells='cells', kernel=tuple(map(tuple, kernel.tolist())))
    return System(elements=[group], interactions=[inhibition])
