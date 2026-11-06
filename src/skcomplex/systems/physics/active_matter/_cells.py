"""Cells of Szabó et al. (2006): self-propelled discs whose headings turn towards the velocities the forces give."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, I, arcsin, cos, exp, oo, pi, sin, sqrt, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import clip, neighbors, where
from ....core.components.spaces import ContinuousSpace, RadiusNeighborhood
from ....core.components.system import System
from ....core.engine.backends import NumbaBackend
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system

SOFTENING = 1e-12


class MotileCellElement(Element):
    """Motile cell of a tissue, with its position, the heading of its motility, and its force and motility parameters.

    The position follows `x' = v0 n + mu F`, where `n` is the unit vector of the heading and `F` the sum of the pair
    forces. A pair force is `F_rep (d - R_eq) / R_eq` towards the neighbour below the distance `R_eq`, and
    `F_adh (d - R_eq) / (R_0 - R_eq)` towards it up to the distance `R_0`, the cell diameter.

    Args:
        label:
            The label of the cell or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial position.

        theta:
            The initial heading.

        v0:
            The speed of the motility.

        mu:
            The mobility.

        tau:
            The relaxation time of the heading.

        eta:
            The strength of the angular noise, whose variance per unit time is `eta^2 / 12`.

        R_eq:
            The equilibrium distance of a pair.

        R_0:
            The distance beyond which a pair exerts no force.

        F_rep:
            The largest repulsive force.

        F_adh:
            The largest adhesive force.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import MotileCellElement
        >>> MotileCellElement('cell').F_rep
        30.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi, role='direction')] = 0.0,
        v0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        mu: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        R_eq: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0 / 6.0,
        R_0: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        F_rep: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 30.0,
        F_adh: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.75,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.theta = theta
        self.v0 = v0
        self.mu = mu
        self.tau = tau
        self.eta = eta
        self.R_eq = R_eq
        self.R_0 = R_0
        self.F_rep = F_rep
        self.F_adh = F_adh


def _velocity(s: SimpleNamespace, *, reach: float) -> FreeModuleElement:
    """Return the velocity `v0 n + mu F` of a cell, with the sum `F` of the forces of the neighbours within a reach."""
    cell = s.target
    dx, dy = cell.other.x[0] - cell.x[0], cell.other.x[1] - cell.x[1]
    distance = sqrt(dx**2 + dy**2)
    stretch = distance - cell.R_eq
    strength = where(
        stretch < 0,
        cell.F_rep * stretch / cell.R_eq,
        cell.F_adh * stretch / (cell.R_0 - cell.R_eq),
    )
    force = neighbors(vector([strength * dx / distance, strength * dy / distance]), over=RadiusNeighborhood(reach))
    return vector([cos(cell.theta), sin(cell.theta)]) * cell.v0 + force * cell.mu


def _turning(s: SimpleNamespace, *, reach: float) -> Expression:
    """Return the rate `arcsin((n x v / |v|) . e_z) / tau` that turns a heading towards the velocity."""
    cell = s.target
    velocity = _velocity(s, reach=reach)
    speed = sqrt(velocity[0] ** 2 + velocity[1] ** 2 + SOFTENING)
    cross = (cos(cell.theta) * velocity[1] - sin(cell.theta) * velocity[0]) / speed
    return arcsin(clip(cross, -1, 1)) / cell.tau


def _angular_noise(s: SimpleNamespace) -> Expression:
    """Return the amplitude `eta / sqrt(12)` of the angular noise."""
    return s.target.eta / sqrt(12)


def _heading(member: SimpleNamespace) -> Expression:
    """Return the unit vector `exp(i theta)` of a member's heading in the complex plane."""
    return exp(I * member.theta)


def _polarization(s: SimpleNamespace) -> Expression:
    """Return the modulus of the mean heading."""
    return abs(s.mean_heading)


@register_system(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'flocking', 'tissue', 'neighbours'),
    references=('Szabó et al. 2006', 'Vicsek and Zafeiris 2012'),
    defaults={'until': 60.0, 'dt': 0.5, 'solver': EulerMaruyamaSolver(dt=0.01), 'backend': NumbaBackend()},
    observables=lambda system: [
        ObservableSpec('mean_heading', _heading, group='cells'),
        ObservableSpec('polarization', _polarization),
    ],
)
def szabo_cells(
    n_cells: int = 200,
    *,
    density: float = 1.0,
    eta: float = 0.6,
    v0: float = 1.0,
    mu: float = 1.0,
    tau: float = 1.0,
    R_eq: float = 5.0 / 6.0,
    R_0: float = 1.0,
    F_rep: float = 30.0,
    F_adh: float = 0.75,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the cells of Szabó et al. (2006) in a periodic box, which move together above a critical density.

    A cell moves at `x' = v0 n + mu F`, where `F` sums the pair forces of the neighbours within `R_0`, the diameter of a
    cell, and its heading turns towards its velocity, `theta' = arcsin((n x v / |v|) . e_z) / tau + xi`. No cell reads
    the headings of the others, and adhesion alone orders them. With `eta = 0.6` the transition lies at the number
    density `rho_c = 0.18 / R_0^2`.

    Args:
        n_cells:
            The number of cells.

        density:
            The number of cells per unit area.

        eta:
            The strength of the angular noise.

        v0:
            The speed of the motility.

        mu:
            The mobility.

        tau:
            The relaxation time of the heading.

        R_eq:
            The equilibrium distance of a pair.

        R_0:
            The distance beyond which a pair exerts no force.

        F_rep:
            The largest repulsive force.

        F_adh:
            The largest adhesive force.

        random_state:
            The seed or generator of the initial positions and headings.

    Returns:
        system:
            The system, with the positions `cells.x`, the headings `cells.theta`, and the observables `mean_heading` and
            `polarization`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import szabo_cells
        >>> szabo_cells(16, density=0.5).space.extent
        (5.656854249492381, 5.656854249492381)
    """
    size = math.sqrt(n_cells / density)
    rng = np.random.default_rng(random_state)
    cells = GroupElement(
        'cells',
        element=MotileCellElement,
        size=n_cells,
        values={
            'x': rng.uniform(0.0, size, (n_cells, 2)),
            'theta': rng.uniform(0.0, 2 * math.pi, n_cells),
            'v0': v0,
            'mu': mu,
            'tau': tau,
            'eta': eta,
            'R_eq': R_eq,
            'R_0': R_0,
            'F_rep': F_rep,
            'F_adh': F_adh,
        },
        visual=VisualSpec('points', {'position': 'x', 'radius': R_0 / 2, 'color_by': 'theta'}),
    )
    motion = RateInteraction('motion', target='cells', state='x', rate=functools.partial(_velocity, reach=R_0))
    turning = RateInteraction(
        'turning',
        target='cells',
        state='theta',
        rate=functools.partial(_turning, reach=R_0),
        noise=_angular_noise,
    )
    return System(
        elements=[cells],
        interactions=[motion, turning],
        space=ContinuousSpace(extent=(size, size), boundary='periodic'),
    )
