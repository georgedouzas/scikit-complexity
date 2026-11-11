"""Kinetic crowd systems: Bellomo's walkers in a venue with walls.

They choose a direction and then a speed.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
import scipy.ndimage
from sage.all__sagemath_symbolics import oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array, IntArray

TINY = 1e-12


class KineticWalkerView(AgentView):
    """View of a walker of the kinetic crowd system, with its states as attributes."""

    x: float
    y: float
    speed: float
    direction: float
    target: float


class KineticWalkerElement(Element):
    """Walker of the kinetic crowd system.

    It holds its position, its speed, its direction, and the side it walks to.

    Args:
        label:
            The label of the walker or the group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        x:
            The position along the street.

        y:
            The position across the street, between the walls at 0 and the width.

        speed:
            The dimensionless speed `v`, between 0 and the largest speed 1.

        direction:
            The direction `theta` of the velocity.

        target:
            The direction of the target, 0 for the exit at the right end and 1 for the exit at the left end.

    Examples:
        >>> from skcomplex.systems.social.crowds import KineticWalkerElement
        >>> KineticWalkerElement('walker', target=1.0).target
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'target', 'color_range': (0.0, 1.0)}),
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
        speed: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.5,
        direction: Annotated[float, StateSpec()] = 0.0,
        target: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = direction
        self.target = target

    @classmethod
    def _get_view(cls) -> type[KineticWalkerView]:
        """Return the class of the views of the agents, whose attributes are the states."""
        return KineticWalkerView


class KineticCrowdInteraction(Interaction):
    """Decision process of Bellomo's kinetic crowd model.

    A direct simulation Monte Carlo of the walkers runs it. The density `rho` and the mean velocity `xi` are counted on
    square cells of side `cell` of a street of length `length`, periodic along it, and width `width`, between walls,
    with `rho = 1` at `packing` walkers per unit area. Every walker interacts with the crowd with the probability
    `eta_A dt` in a step of length `dt`. It first turns to the preferred direction
    `nu_p ~ rho nu_v + (1 - rho) (beta nu_s + (1 - beta) nu_t) / |.|`, where `nu_v` points down the gradient of the
    density, `nu_s` along the mean velocity, and `nu_t` to its target, and then adjusts its speed to the perceived
    density `rho_theta = rho + d / sqrt(1 + d^2) ((1 - rho) H(d) + rho H(-d))` of the derivative `d` of the density
    along its new direction: a walker no faster than the mean speed accelerates to
    `xi + alpha beta (1 - rho_theta) (alpha beta - xi)` with the probability `alpha beta (1 - rho_theta)`, and a faster
    one decelerates to `xi - rho_theta xi` with the probability `(1 - c alpha beta) rho_theta`. A walker at a distance
    `d < d_w` from a wall and heading to it then reduces the normal component of its velocity to `d / d_w` of it at
    constant speed, and every walker moves along its velocity. From the time `panic_time` on, the stress parameter is
    `panic_stress` in place of `stress`. The interaction reports the band index, the mean of
    `|rho_1 - rho_2| / (rho_1 + rho_2)` over the cells of the street where walkers are, of the densities smoothed by a
    Gaussian of width `band_width`, and the mean speed.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of walkers.

        length:
            The length of the street, periodic along it.

        width:
            The width of the street, between the walls.

        cell:
            The side of the square cells of the density and the mean velocity.

        packing:
            The number of walkers per unit area at the largest density 1.

        quality:
            The quality `alpha` of the venue.

        stress:
            The stress `beta`, the weight of the stream against the target.

        deceleration:
            The constant `c < 1` of the probability of deceleration.

        rate:
            The interaction rate `eta_A` between walkers.

        wall_distance:
            The distance `d_w` from a wall below which walkers turn away from it.

        dt:
            The length of a step.

        panic_time:
            The time `t_c` from which the stress is `panic_stress`, or `None` for no panic.

        panic_stress:
            The stress `beta_2` after the panic time.

        band_width:
            The width of the Gaussian smoothing of the densities of the band index.

    Examples:
        >>> from skcomplex.systems.social.crowds import KineticCrowdInteraction
        >>> KineticCrowdInteraction('decisions', target='walkers').stress
        0.5
    """

    draws = True

    law_classes = (RuleLaw,)

    reports: ClassVar[tuple[str, ...]] = ('band_index', 'mean_speed')

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        length: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        cell: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.5,
        packing: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.0,
        quality: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 1.0,
        stress: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.5,
        deceleration: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, 1))] = 0.95,
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        wall_distance: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        dt: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        panic_time: float | None = None,
        panic_stress: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.9,
        band_width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.length = length
        self.width = width
        self.cell = cell
        self.packing = packing
        self.quality = quality
        self.stress = stress
        self.deceleration = deceleration
        self.rate = rate
        self.wall_distance = wall_distance
        self.dt = dt
        self.panic_time = panic_time
        self.panic_stress = panic_stress
        self.band_width = band_width

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the decisions and the motion of the walkers.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.walk)]

    def _get_cells(self, x: Array, y: Array) -> tuple[IntArray, IntArray, tuple[int, int]]:
        """Return the cell indices of positions along and across the street.

        It also returns the numbers of cells.
        """
        shape = (max(round(self.length / self.cell), 1), max(round(self.width / self.cell), 1))
        column = np.clip((x / self.length * shape[0]).astype(int), 0, shape[0] - 1)
        row = np.clip((y / self.width * shape[1]).astype(int), 0, shape[1] - 1)
        return column, row, shape

    def _get_fields(self, x: Array, y: Array, velocity: Array, target: Array) -> dict[str, Array]:
        """Return the counts of each side on the cells, and the fields at every walker.

        The fields are the density, its gradient, and the means.
        """
        column, row, shape = self._get_cells(x, y)
        area = (self.length / shape[0]) * (self.width / shape[1])
        counts = [np.zeros(shape) for _ in range(2)]
        for side in range(2):
            np.add.at(counts[side], (column[target == side], row[target == side]), 1.0)
        total = counts[0] + counts[1]
        momentum = np.zeros((2, *shape))
        for axis in range(2):
            np.add.at(momentum[axis], (column, row), velocity[:, axis])
        speeds = np.zeros(shape)
        np.add.at(speeds, (column, row), np.linalg.norm(velocity, axis=1))
        mean = momentum / np.maximum(total, 1.0)
        density = np.minimum(total / (self.packing * area), 1.0)
        padded = np.pad(density, ((1, 1), (0, 0)), mode='wrap')
        gradient_x = np.gradient(padded, self.length / shape[0], axis=0)[1:-1]
        gradient_y = np.gradient(density, self.width / shape[1], axis=1) if shape[1] > 1 else np.zeros(shape)
        return {
            'first': counts[0],
            'second': counts[1],
            'density': density[column, row],
            'gradient': np.stack([gradient_x[column, row], gradient_y[column, row]], axis=1),
            'mean': mean[:, column, row].T,
            'speed': (speeds / np.maximum(total, 1.0))[column, row],
        }

    @staticmethod
    def _get_unit(vectors: Array, fallback: Array) -> Array:
        """Return the unit vectors of the rows.

        Where a vector vanishes, it returns the fallback rows.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.where(norms > TINY, vectors / np.maximum(norms, TINY), fallback)

    def _interact(
        self,
        walkers: dict[str, Array],
        fields: dict[str, Array],
        stress: float,
        rng: np.random.Generator,
    ) -> None:
        """Turn the interacting walkers to their preferred direction.

        Their speeds are adjusted in probability.
        """
        rho, gradient, mean = fields['density'], fields['gradient'], fields['mean']
        goal = np.stack([np.where(walkers['target'] == 0, 1.0, -1.0), np.zeros(rho.size)], axis=1)
        vacuum = self._get_unit(-gradient, np.zeros_like(gradient))
        stream = self._get_unit(mean, goal)
        mixed = self._get_unit(stress * stream + (1 - stress) * goal, goal)
        preferred = self._get_unit(rho[:, None] * vacuum + (1 - rho[:, None]) * mixed, goal)
        slope = np.einsum('ij,ij->i', gradient, preferred)
        perceived = rho + slope / np.sqrt(1 + slope**2) * np.where(slope >= 0, 1 - rho, rho)
        mean_speed = fields['speed']
        gamma = self.quality * stress
        accelerate = (walkers['speed'] <= mean_speed) & (rng.random(rho.size) < gamma * (1 - perceived))
        decelerate = (walkers['speed'] > mean_speed) & (
            rng.random(rho.size) < (1 - self.deceleration * gamma) * perceived
        )
        speed = np.where(accelerate, mean_speed + gamma * (1 - perceived) * (gamma - mean_speed), walkers['speed'])
        speed = np.where(decelerate, mean_speed - perceived * mean_speed, speed)
        active = rng.random(rho.size) < self.rate * self.dt
        walkers['speed'] = np.where(active, np.clip(speed, 0.0, 1.0), walkers['speed'])
        angle = np.arctan2(preferred[:, 1], preferred[:, 0])
        walkers['direction'] = np.where(active, angle, walkers['direction'])

    def _avoid_walls(self, walkers: dict[str, Array]) -> None:
        """Reduce the normal component of the velocity of walkers near a wall.

        Only walkers heading to the wall turn, at constant speed.
        """
        speed, direction = walkers['speed'], walkers['direction']
        vx, vy = speed * np.cos(direction), speed * np.sin(direction)
        for distance, normal in ((walkers['y'], 1.0), (self.width - walkers['y'], -1.0)):
            close = (distance < self.wall_distance) & (vy * normal < 0)
            ratio = np.clip(distance / self.wall_distance, 0.0, 1.0)
            normal_part = ratio * vy
            tangent = np.where(vx >= 0, 1.0, -1.0) * np.sqrt(np.maximum(speed**2 - normal_part**2, 0.0))
            vx, vy = np.where(close, tangent, vx), np.where(close, normal_part, vy)
        walkers['direction'] = np.arctan2(vy, vx)

    def _get_band_index(self, first: Array, second: Array) -> float:
        """Return the band index of the densities of both sides where walkers are.

        The densities are smoothed over `band_width`.
        """
        sigma = self.band_width / self.cell
        smooth = [scipy.ndimage.gaussian_filter(counts, sigma, mode=('wrap', 'nearest')) for counts in (first, second)]
        total = smooth[0] + smooth[1]
        occupied = total > 1e-3 * total.max() if total.max() > 0 else total > 0
        separation = np.abs(smooth[0] - smooth[1]) / np.maximum(total, TINY)
        return float(separation[occupied].mean()) if occupied.any() else 0.0

    def walk(self, system: SystemView) -> None:
        """Run the decisions, the turns at the walls, and the motion of one step.

        Args:
            system:
                The system.
        """
        views = system.get_agents(self.target, KineticWalkerView)
        if not views:
            return
        walkers = {
            name: np.array([getattr(view, name) for view in views])
            for name in ('x', 'y', 'speed', 'direction', 'target')
        }
        stress = self.stress
        if self.panic_time is not None and system.time >= self.panic_time:
            stress = self.panic_stress
        velocity = np.stack([np.cos(walkers['direction']), np.sin(walkers['direction'])], axis=1)
        fields = self._get_fields(walkers['x'], walkers['y'], walkers['speed'][:, None] * velocity, walkers['target'])
        self._interact(walkers, fields, stress, system.rng)
        self._avoid_walls(walkers)
        walkers['x'] = np.mod(walkers['x'] + self.dt * walkers['speed'] * np.cos(walkers['direction']), self.length)
        walkers['y'] = np.clip(
            walkers['y'] + self.dt * walkers['speed'] * np.sin(walkers['direction']),
            0.0,
            self.width,
        )
        for position, view in enumerate(views):
            for name in ('x', 'y', 'speed', 'direction'):
                setattr(view, name, float(walkers[name][position]))
        system.report('band_index', self._get_band_index(fields['first'], fields['second']))
        system.report('mean_speed', float(walkers['speed'].mean()))


class KineticCrowdSystem(BaseSystem):
    """Bellomo's kinetic crowd of two groups walking in opposite directions.

    They walk along a street with walls. Walkers in a continuous venue choose first a direction and then a speed
    (Bellomo et al. 2017 sections 6.4 and 6.5). The preferred direction weighs the search of less crowded areas, the
    attraction of the mean stream, and the target, by the density `rho` and the stress `beta` (Eq. 6.19). The speed then
    adjusts to the density perceived along that direction, with the venue quality `alpha` and the stress (Eqs. 6.22 to
    6.24), and walkers near a wall turn parallel to it at constant speed (Eq. 6.25). Panic raises the stress from
    `beta_1` to `beta_2` at a critical time (Eqs. 6.27 and 6.28). The kinetic equation is solved, as in the book, by a
    particle method close to direct simulation Monte Carlo, here with one particle per walker. In space-homogeneous
    conditions the mean speed follows the fundamental diagram
    `alpha^3 beta^3 (1 - rho)^2 / (alpha^2 beta^2 (1 - rho)^2 + (1 - c alpha beta) rho^2)` (Eq. 6.30), which a single
    cell spanning the street reproduces. The default is the counterflow of section 6.5.2, 50 walkers uniformly
    distributed in a street periodic along its length, for which the book reports segregation into lanes, measured by
    the band index of Eq. 6.31, most pronounced at `beta = 0.5`. The densities are counted on cells of side `cell`, an
    assumption of this particle method. The venue is a distinct system from `EvacuationSystem`, a network of rooms. One
    step is a tenth of a time unit.

    The system has the group `walkers` and the reporters `decisions.band_index` and `decisions.mean_speed`.

    Args:
        n_walkers:
            The number of walkers, half walking to each end.

        length:
            The length of the street, periodic along it.

        width:
            The width of the street, between the walls.

        cell:
            The side of the square cells of the density and the mean velocity.

        packing:
            The number of walkers per unit area at the largest density 1.

        quality:
            The quality `alpha` of the venue.

        stress:
            The stress `beta`, the weight of the stream against the target.

        deceleration:
            The constant `c < 1` of the probability of deceleration.

        rate:
            The interaction rate `eta_A` between walkers.

        wall_distance:
            The distance `d_w` from a wall below which walkers turn away from it.

        panic_time:
            The time `t_c`, in steps, from which the stress is `panic_stress`, or `None` for no panic.

        panic_stress:
            The stress `beta_2` after the panic time.

        random_state:
            The seed or generator of the initial positions and speeds.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.crowds import KineticCrowdSystem
        >>> KineticCrowdSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'social/crowds'
    topics = ('pedestrians', 'kinetic theory', 'agents', 'lanes', 'panic', 'monte carlo')
    references = ('Bellomo, Knopoff, and Soler 2017', 'Bellomo, Bellouquid, and Knopoff 2013')
    until = 300

    def __init__(
        self,
        n_walkers: int = 50,
        *,
        length: float = 20.0,
        width: float = 5.0,
        cell: float = 2.5,
        packing: float = 4.0,
        quality: float = 1.0,
        stress: float = 0.5,
        deceleration: float = 0.95,
        rate: float = 1.0,
        wall_distance: float = 0.5,
        panic_time: float | None = None,
        panic_stress: float = 0.9,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_walkers = n_walkers
        self.length = length
        self.width = width
        self.cell = cell
        self.packing = packing
        self.quality = quality
        self.stress = stress
        self.deceleration = deceleration
        self.rate = rate
        self.wall_distance = wall_distance
        self.panic_time = panic_time
        self.panic_stress = panic_stress
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        target = (np.arange(self.n_walkers) % 2).astype(float)
        walkers = GroupElement(
            'walkers',
            element=KineticWalkerElement,
            size=self.n_walkers,
            values={
                'x': rng.random(self.n_walkers) * self.length,
                'y': rng.random(self.n_walkers) * self.width,
                'speed': 0.5 * rng.random(self.n_walkers),
                'direction': np.pi * target,
                'target': target,
            },
        )
        decisions = KineticCrowdInteraction(
            'decisions',
            target='walkers',
            length=self.length,
            width=self.width,
            cell=self.cell,
            packing=self.packing,
            quality=self.quality,
            stress=self.stress,
            deceleration=self.deceleration,
            rate=self.rate,
            wall_distance=self.wall_distance,
            panic_time=self.panic_time,
            panic_stress=self.panic_stress,
        )
        return CompositeSystem(self.label, elements=[walkers], interactions=[decisions])
