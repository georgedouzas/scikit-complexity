"""Cells of the Szabó (2006) system.

The cells are self-propelled discs whose headings turn towards the velocities the forces give.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, I, arcsin, cos, exp, oo, sin, sqrt, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, Period
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw
from ....core.components.laws.utils import clip, neighbors, where
from ....core.components.spaces import BaseSpace, ContinuousSpace, RadiusNeighborhood
from ....core.engine.backends import BaseBackend, NumbaBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.specs import Role, VisualSpec
from ....core.systems import BaseSystem, System

SOFTENING = 1e-12


class MotileCellElement(Element):
    """Motile cells of a tissue with the positions `x` and the headings `theta` of their motility.

    Args:
        label:
            The label of the cells.

        x:
            The initial positions, one row per cell or shared.

        theta:
            The initial headings, one per cell or shared.

        size:
            The number of cells, or `None` for one.

        space:
            The continuous space of the cells, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import MotileCellElement
        >>> MotileCellElement('cell', theta=1.0).theta
        1.0
    """

    visual = VisualSpec('points', {'position': 'x', 'radius': 0.5, 'color_by': 'theta'})

    def __init__(
        self,
        label: str,
        *,
        x: object = (0.0, 0.0),
        theta: Annotated[object, Period(2 * math.pi)] = 0.0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.theta = theta


class CellMotilityInteraction(Interaction):
    """Motility of cells that push and pull their neighbours and turn towards their velocity, as Szabó et al. (2006).

    The position follows `x' = v0 n + mu F`, where `n` is the unit vector of the heading and `F` the sum of the pair
    forces of the neighbours within `R_0`. A pair force is `F_rep (d - R_eq) / R_eq` towards the neighbour below the
    distance `R_eq`, and `F_adh (d - R_eq) / (R_0 - R_eq)` towards it up to the distance `R_0`, the cell diameter. The
    heading turns towards the velocity, `theta' = arcsin((n x v / |v|) . e_z) / tau + xi`, with angular noise of
    variance `eta^2 / 12` per unit time. It is a keratocyte in a wound, or a cell of an epithelium that migrates.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the cells, with the positions `x` and the headings `theta`.

        v0:
            The speed of the motility.

        mu:
            The mobility.

        tau:
            The relaxation time of the heading.

        eta:
            The strength of the angular noise.

        R_eq:
            The equilibrium distance of a pair.

        R_0:
            The distance beyond which a pair exerts no force.

        F_rep:
            The largest repulsive force.

        F_adh:
            The largest adhesive force.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import CellMotilityInteraction
        >>> CellMotilityInteraction('motility', target='cells').F_rep
        30.0
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        v0: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        mu: Annotated[float, RealSet.open(0, oo)] = 1.0,
        tau: Annotated[float, RealSet.open(0, oo)] = 1.0,
        eta: Annotated[float, RealSet.closed_open(0, oo)] = 0.6,
        R_eq: Annotated[float, RealSet.open(0, oo)] = 5.0 / 6.0,
        R_0: Annotated[float, RealSet.open(0, oo)] = 1.0,
        F_rep: Annotated[float, RealSet.closed_open(0, oo)] = 30.0,
        F_adh: Annotated[float, RealSet.closed_open(0, oo)] = 0.75,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.v0 = v0
        self.mu = mu
        self.tau = tau
        self.eta = eta
        self.R_eq = R_eq
        self.R_0 = R_0
        self.F_rep = F_rep
        self.F_adh = F_adh

    def _get_velocity(self, s: SimpleNamespace) -> FreeModuleElement:
        """Return the velocity `v0 n + mu F` of a cell, with the forces of the neighbours within `R_0`."""
        cell = s.target
        dx, dy = cell.other.x[0] - cell.x[0], cell.other.x[1] - cell.x[1]
        distance = sqrt(dx**2 + dy**2)
        stretch = distance - s.R_eq
        strength = where(stretch < 0, s.F_rep * stretch / s.R_eq, s.F_adh * stretch / (s.R_0 - s.R_eq))
        force = neighbors(
            vector([strength * dx / distance, strength * dy / distance]),
            over=RadiusNeighborhood(self.R_0),
        )
        return vector([cos(cell.theta), sin(cell.theta)]) * s.v0 + force * s.mu

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the motion of the cells of the target, and the turning and the noise of their headings.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            theta = s.target.theta
            velocity = self._get_velocity(s)
            speed = sqrt(velocity[0] ** 2 + velocity[1] ** 2 + SOFTENING)
            cross = (cos(theta) * velocity[1] - sin(theta) * velocity[0]) / speed
            self.laws_ = [
                RateLaw(s.target.x, velocity),
                RateLaw(theta.diff(s.t), arcsin(clip(cross, -1, 1)) / s.tau),
                NoiseLaw(theta, s.eta / sqrt(12)),
            ]


def _get_heading(member: SimpleNamespace) -> Expression:
    """Return the unit vector `exp(i theta)` of a member's heading.

    The vector lies in the complex plane.
    """
    return exp(I * member.theta)


def _get_polarization(s: SimpleNamespace) -> Expression:
    """Return the modulus of the mean heading."""
    return abs(s.mean_heading)


class SzaboCellsSystem(BaseSystem):
    """The cells of Szabó et al. (2006) in a periodic box.

    The cells move together above a critical density.

    A cell moves at `x' = v0 n + mu F`, where `F` sums the pair forces of the neighbours within `R_0`, the diameter of a
    cell, and its heading turns towards its velocity, `theta' = arcsin((n x v / |v|) . e_z) / tau + xi`. No cell reads
    the headings of the others, and adhesion alone orders them. With `eta = 0.6` the transition lies at the number
    density `rho_c = 0.18 / R_0^2`. The system has the positions `cells.x`, the headings `cells.theta`, and the
    observables `mean_heading` and `polarization`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import SzaboCellsSystem
        >>> SzaboCellsSystem(16, density=0.5).prepare().system_.elements[0].space.extent
        (5.656854249492381, 5.656854249492381)
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'flocking', 'tissue', 'neighbours')
    references = ('Szabó et al. 2006', 'Vicsek and Zafeiris 2012')
    until = 60.0
    dt = 0.5
    observables = (
        Observable('mean_heading', _get_heading, group='cells'),
        Observable('polarization', _get_polarization),
    )

    def __init__(
        self,
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
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_cells = n_cells
        self.density = density
        self.eta = eta
        self.v0 = v0
        self.mu = mu
        self.tau = tau
        self.R_eq = R_eq
        self.R_0 = R_0
        self.F_rep = F_rep
        self.F_adh = F_adh
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the cells of Szabó et al. (2006) in a periodic box.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        size = math.sqrt(self.n_cells / self.density)
        rng = np.random.default_rng(self.random_state)
        cells = MotileCellElement(
            'cells',
            x=rng.uniform(0.0, size, (self.n_cells, 2)),
            theta=rng.uniform(0.0, 2 * math.pi, self.n_cells),
            size=self.n_cells,
            space=ContinuousSpace(extent=(size, size), boundary='periodic'),
        )
        motility = CellMotilityInteraction(
            'motility',
            target='cells',
            v0=self.v0,
            mu=self.mu,
            tau=self.tau,
            eta=self.eta,
            R_eq=self.R_eq,
            R_0=self.R_0,
            F_rep=self.F_rep,
            F_adh=self.F_adh,
        )
        self.system_ = System(
            self.label,
            elements=[cells],
            interactions=[motility],
            solver=EulerMaruyamaSolver(dt=0.01),
            backend=NumbaBackend(),
        )
