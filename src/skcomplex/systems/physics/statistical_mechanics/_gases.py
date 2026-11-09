"""Statistical mechanics systems.

The module covers gases, fluids, and spin systems.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.interactions import BaseInteraction
from ....core.components.spaces import ContinuousSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import VelocityVerletSolver
from ....core.specs import VisualSpec
from ..common.particles import LangevinThermostatInteraction, LennardJonesInteraction, MassPointElement


class IdealGasSystem(BaseSystem):
    """An ideal gas of non-interacting molecules in a reflecting box.

    Positions are uniform in the box and velocity components are normal with variance `kT / mass`, with the mean
    velocity removed. The molecules are drawn as points colored by speed. NOTE-RETURNS: The gas.

    Args:
        n_molecules:
            The number of molecules.

        kT:
            The temperature in energy units.

        size:
            The side of the box.

        n_dim:
            The number of dimensions.

        mass:
            The mass of every molecule.

        random_state:
            The seed or generator of the initial positions and velocities.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import IdealGasSystem
        >>> group = IdealGasSystem(1000, random_state=0).prepare().system_.elements[0]
        >>> group.label, group.element.__name__, group.size
        ('gas', 'MassPointElement', 1000)
    """

    field = 'physics/statistical_mechanics'
    topics = ('gas', 'ode', 'boundary')
    references = ('Maxwell 1860',)
    until = 2.0
    dt = 0.01

    def __init__(
        self,
        n_molecules: int = 500,
        *,
        kT: float = 1.0,
        size: float = 1.0,
        n_dim: int = 2,
        mass: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_molecules = n_molecules
        self.kT = kT
        self.size = size
        self.n_dim = n_dim
        self.mass = mass
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return an ideal gas of non-interacting molecules in a reflecting box."""
        rng = np.random.default_rng(self.random_state)
        velocities = rng.normal(0.0, np.sqrt(self.kT / self.mass), size=(self.n_molecules, self.n_dim))
        velocities -= velocities.mean(axis=0)
        visual = VisualSpec('points', {'position': 'x', 'radius': self.size / 200, 'color_by': 'x_d1', 'states': ()})
        molecules = GroupElement(
            'gas',
            element=MassPointElement,
            size=self.n_molecules,
            values={
                'x': rng.uniform(0.0, self.size, size=(self.n_molecules, self.n_dim)),
                'x_d1': velocities,
                'm': self.mass,
            },
            visual=visual,
        )
        return CompositeSystem(
            self.label,
            elements=[molecules],
            space=ContinuousSpace(extent=[self.size] * self.n_dim, boundary='reflecting'),
        )


class LennardJonesFluidSystem(BaseSystem):
    """Lennard-Jones particles in a periodic square box.

    A thermostat optionally holds the particles at a temperature.

    The system has the particle group `fluid` in two dimensions.

    Args:
        n_particles:
            The number of particles.

        density:
            The number of particles per unit area.

        temperature:
            The temperature of the Langevin thermostat, or `None` for constant energy. A thermostat makes the system
            stochastic, so it runs with the Euler-Maruyama solver instead of the default velocity Verlet.

        cutoff:
            The distance beyond which pairs do not interact.

        random_state:
            The seed or generator of the initial jitter and velocities.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import LennardJonesFluidSystem
        >>> LennardJonesFluidSystem(100).prepare().system_.elements[0].prepare().size_
        100
    """

    field = 'physics/statistical_mechanics'
    topics = ('molecular dynamics', 'pair potential', 'ode')
    references = ('Lennard-Jones 1924', 'Verlet 1967')
    until = 0.5
    dt = 0.005
    solver = VelocityVerletSolver()

    def __init__(
        self,
        n_particles: int = 400,
        *,
        density: float = 0.5,
        temperature: float | None = None,
        cutoff: float = 2.5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.density = density
        self.temperature = temperature
        self.cutoff = cutoff
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return Lennard-Jones particles in a periodic square box.

        A thermostat optionally holds the particles at a temperature.
        """
        rng = np.random.default_rng(self.random_state)
        size = float(np.sqrt(self.n_particles / self.density))
        side = int(np.ceil(np.sqrt(self.n_particles)))
        spacing = size / side
        grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[: self.n_particles]
        positions = np.mod((grid + 0.5) * spacing + rng.normal(0, 0.02 * spacing, (self.n_particles, 2)), size)
        velocities = rng.normal(0, np.sqrt(self.temperature or 1.0), (self.n_particles, 2))
        velocities -= velocities.mean(axis=0)
        interactions: list[BaseInteraction] = [
            LennardJonesInteraction('lj', first='fluid', second='fluid', cutoff=self.cutoff, divisor='m', period=size),
        ]
        if self.temperature is not None:
            interactions.append(
                LangevinThermostatInteraction('thermostat', target='fluid', temperature=self.temperature),
            )
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'fluid',
                    element=MassPointElement,
                    size=self.n_particles,
                    values={'x': positions, 'x_d1': velocities},
                ),
            ],
            interactions=interactions,
            space=ContinuousSpace(extent=(size, size), boundary='periodic'),
        )
