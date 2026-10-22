"""Statistical mechanics models: gases, fluids, and spin models."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.base import VisualSpec
from ....core.components import Particle
from ....core.interactions import Interaction
from ....core.runtime import register_model
from ....core.space import ContinuousSpace
from ....core.system import System
from ..realizations import LangevinThermostat, LennardJones


@register_model(
    field='physics/statistical_mechanics',
    tags=('gas', 'ode', 'boundary'),
    references=('Maxwell 1860',),
    defaults={'until': 2.0, 'dt': 0.01},
)
def ideal_gas(
    n_molecules: int = 500,
    *,
    kT: float = 1.0,
    size: float = 1.0,
    n_dim: int = 2,
    mass: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
    label: str = 'gas',
) -> System:
    """Return an ideal gas of non-interacting molecules in a reflecting box.

    Positions are uniform in the box and velocity components are normal with variance `kT / mass`, with the mean
    velocity removed. The molecules are drawn as points colored by speed.

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
            The label of the molecule group. The box is labelled `box`.

    Returns:
        system:
            The gas.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ideal_gas
        >>> ideal_gas(1000, random_state=0).elements[0]
        Group('gas', element=Particle, size=1000)
    """
    rng = np.random.default_rng(random_state)
    velocities = rng.normal(0.0, np.sqrt(kT / mass), size=(n_molecules, n_dim))
    velocities -= velocities.mean(axis=0)
    visual = VisualSpec('points', {'position': 'x', 'radius': size / 200, 'color_by': 'v', 'states': ()})
    molecules = Particle.group(
        label,
        n_molecules,
        x=rng.uniform(0.0, size, size=(n_molecules, n_dim)),
        v=velocities,
        m=mass,
        visual=visual,
    )
    return System(
        components=[molecules],
        space=ContinuousSpace(*[size] * n_dim, boundary='reflecting'),
    )


@register_model(
    field='physics/statistical_mechanics',
    tags=('molecular dynamics', 'pair potential', 'ode'),
    references=('Lennard-Jones 1924', 'Verlet 1967'),
    defaults={'until': 0.5, 'dt': 0.005, 'solver': 'velocity-verlet'},
)
def lennard_jones_fluid(
    n_particles: int = 400,
    *,
    density: float = 0.5,
    temperature: float | None = None,
    cutoff: float = 2.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Lennard-Jones particles in a periodic square box, optionally held at a temperature by a thermostat.

    Args:
        n_particles:
            The number of particles.

        density:
            The number of particles per unit area.

        temperature:
            The temperature of the Langevin thermostat, or `None` for constant energy. A thermostat makes the model
            stochastic, so it runs with the Euler-Maruyama solver instead of the default velocity Verlet.

        cutoff:
            The distance beyond which pairs do not interact.

        random_state:
            The seed or generator of the initial jitter and velocities.

    Returns:
        system:
            The model, with the particle group `fluid` in two dimensions.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import lennard_jones_fluid
        >>> lennard_jones_fluid(100).elements[0].size
        100
    """
    rng = np.random.default_rng(random_state)
    size = float(np.sqrt(n_particles / density))
    side = int(np.ceil(np.sqrt(n_particles)))
    spacing = size / side
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:n_particles]
    positions = np.mod((grid + 0.5) * spacing + rng.normal(0, 0.02 * spacing, (n_particles, 2)), size)
    velocities = rng.normal(0, np.sqrt(temperature or 1.0), (n_particles, 2))
    velocities -= velocities.mean(axis=0)
    interactions: list[Interaction] = [
        LennardJones('lj', first='fluid', second='fluid', cutoff=cutoff, period=size),
    ]
    if temperature is not None:
        interactions.append(LangevinThermostat('thermostat', target='fluid', temperature=temperature))
    return System(
        components=[Particle.group('fluid', n_particles, x=positions, v=velocities)],
        interactions=interactions,
        space=ContinuousSpace(size, size, boundary='periodic'),
    )
