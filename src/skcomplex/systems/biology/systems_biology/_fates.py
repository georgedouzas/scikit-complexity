"""Systems biology systems: cell fate decisions as landscapes and as lateral inhibition between cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.system import System
from ..._catalog import register_system
from ._fate_landscape import FateLandscapeElement
from ._lateral_inhibition import LateralInhibitionInteraction
from ._proneural_cell import ProneuralCellElement


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
