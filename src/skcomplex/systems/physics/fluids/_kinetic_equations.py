"""Kinetic equations of gases and plasmas over position and velocity, on the phase spaces of the core."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ....core.components.elements import DensityElement, FieldElement
from ....core.components.interactions import AdvectionInteraction, EncounterInteraction
from ....core.components.laws.utils import coordinate, gradient, inverse_laplacian, moment
from ....core.components.spaces import LatticeSpace, PhaseSpace
from ....core.components.system import System
from ....core.engine.solvers import RK4Solver, SemiLagrangianSolver, SSPRKSolver, StrangSplittingSolver
from ....core.errors import ValidationError
from ....core.specs import AxisSpec, VisualSpec
from ..._catalog import register_system
from ._bhatnagar_gross_krook import BhatnagarGrossKrookInteraction

SETUPS = ('landau', 'two-stream')
SHOCK_TUBE = 'sod'
VELOCITY_AXES = ('vx', 'vy')
VLASOV_VISUAL = VisualSpec('image', {'moments': (0,)})


def _get_maxwellian(
    density: npt.NDArray[np.float64],
    mean: npt.NDArray[np.float64],
    temperature: npt.NDArray[np.float64],
    speeds: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """Return the Maxwellian of densities, mean velocities, and temperatures over one velocity axis, one row each."""
    rows = (speeds[None, :] - mean[:, None]) ** 2 / (2 * temperature[:, None])
    return density[:, None] * np.exp(-rows) / np.sqrt(2 * np.pi * temperature[:, None])


def _get_streaming(_: SimpleNamespace) -> list[Expression]:
    """Return the drift of free streaming over one position and one velocity axis."""
    return [coordinate('v'), 0]


def _get_plane_streaming(_: SimpleNamespace) -> list[Expression]:
    """Return the drift of free streaming over one position and two velocity axes."""
    return [coordinate('vx'), 0, 0]


def _get_poisson_drift(s: SimpleNamespace) -> list[Expression]:
    """Return the velocity along position and the force of the potential of the density along velocity."""
    density = moment(s.target.f, 0, axes=('v',))
    return [coordinate('v'), -gradient(inverse_laplacian(1 - density), 1)[0]]


def _get_uniform_gas(positions: int, velocities: int, knudsen: float, half_width: float) -> tuple[FieldElement, float]:
    """Return a gas over one position and two velocity axes, with a density wave, and its collision frequency."""
    length = 2 * math.pi
    axes = [AxisSpec(name, velocities, -half_width, half_width) for name in VELOCITY_AXES]
    space = PhaseSpace(LatticeSpace((positions,), spacing=length / positions), axes).prepare()
    vx, vy = np.meshgrid(space.coordinates_['vx'], space.coordinates_['vy'], indexing='ij')
    x = np.arange(positions) * length / positions
    profile = np.exp(-(vx**2 + vy**2) / 2) / (2 * np.pi)
    values = np.multiply.outer(1.0 + 0.2 * np.sin(x), profile)
    gas = FieldElement('gas', element=DensityElement, space=space, values={'f': values}, discretization='finite-volume')
    return gas, 1.0 / knudsen


def _get_maxwell_molecules(_speed: object, _cosine: object) -> float:
    """Return the cross-section of Maxwell molecules in two velocity dimensions."""
    return 1 / (2 * math.pi)


@register_system(
    field='physics/fluids',
    tags=('field', 'kinetic', 'phase space', 'collisions', 'pde'),
    references=('Bellomo et al. 2017', 'Mouhot and Pareschi 2006'),
    defaults={
        'until': 0.5,
        'dt': 0.1,
        'solver': StrangSplittingSolver(SemiLagrangianSolver(0.1), RK4Solver(0.05), 0.1),
    },
)
def boltzmann_equation(
    positions: int = 8,
    velocities: int = 12,
    *,
    knudsen: float = 1.0,
    half_width: float = 6.0,
) -> System:
    """Return the Boltzmann equation of Maxwell molecules over one position and two velocity axes.

    Free streaming moves a density wave along the periodic position axis, and binary collisions redistribute the density
    over the velocities at every position, by the fast spectral method.

    Args:
        positions:
            The number of positions.

        velocities:
            The number of points of each velocity axis.

        knudsen:
            The Knudsen number, the inverse of the collision frequency.

        half_width:
            The half-width of each velocity axis.

    Returns:
        system:
            The system, with the density `gas.f`.

    Examples:
        >>> from skcomplex.systems.physics.fluids import boltzmann_equation
        >>> boltzmann_equation(4, 8).elements[0].prepare().size_
        256
    """
    gas, frequency = _get_uniform_gas(positions, velocities, knudsen, half_width)
    stream = AdvectionInteraction('stream', target='gas', state='f', drift=_get_plane_streaming)
    collide = EncounterInteraction(
        'collide',
        candidate='gas',
        cross_section=_get_maxwell_molecules,
        factor=lambda _: frequency,
    )
    return System(elements=[gas], interactions=[stream, collide])


@register_system(
    field='physics/fluids',
    tags=('field', 'kinetic', 'phase space', 'collisions', 'pde'),
    references=('Bellomo et al. 2017', 'Enskog 1922'),
    defaults={
        'until': 0.5,
        'dt': 0.1,
        'solver': StrangSplittingSolver(SemiLagrangianSolver(0.1), RK4Solver(0.05), 0.1),
    },
)
def enskog(
    positions: int = 8,
    velocities: int = 12,
    *,
    knudsen: float = 1.0,
    diameter: float = 0.2,
    half_width: float = 6.0,
) -> System:
    """Return the local Enskog equation, the Boltzmann equation with the contact value of the pair correlation of disks.

    The collision frequency is multiplied at every position by Henderson's contact value `(1 - 7 c / 16) / (1 - c)^2` of
    the area fraction `c = pi n d^2 / 4`, with `n` the density moment and `d` the diameter of the disks. The collision
    partners sit at one position, so the shift of one diameter of the full Enskog collision is left out.

    Args:
        positions:
            The number of positions.

        velocities:
            The number of points of each velocity axis.

        knudsen:
            The Knudsen number, the inverse of the collision frequency of the dilute gas.

        diameter:
            The diameter of the disks.

        half_width:
            The half-width of each velocity axis.

    Returns:
        system:
            The system, with the density `gas.f`.

    Examples:
        >>> from skcomplex.systems.physics.fluids import enskog
        >>> enskog(4, 8).elements[0].prepare().size_
        256
    """
    gas, frequency = _get_uniform_gas(positions, velocities, knudsen, half_width)
    area = math.pi * diameter**2 / 4

    def contact(s: SimpleNamespace) -> Expression:
        fraction = area * moment(s.candidate.f, 0, axes=VELOCITY_AXES)
        return frequency * (1 - 7 * fraction / 16) / (1 - fraction) ** 2

    stream = AdvectionInteraction('stream', target='gas', state='f', drift=_get_plane_streaming)
    collide = EncounterInteraction('collide', candidate='gas', cross_section=_get_maxwell_molecules, factor=contact)
    return System(elements=[gas], interactions=[stream, collide])


@register_system(
    field='physics/fluids',
    tags=('field', 'kinetic', 'phase space', 'shocks', 'pde'),
    references=('Bhatnagar, Gross, and Krook 1954', 'Mieussens 2000', 'Sod 1978'),
    defaults={'until': 0.2, 'dt': 0.05, 'solver': StrangSplittingSolver(SSPRKSolver(), RK4Solver(0.005), 0.005)},
)
def bgk_equation(
    positions: int = 100,
    velocities: int = 32,
    *,
    knudsen: float = 1e-2,
    half_width: float = 6.0,
) -> System:
    """Return the BGK equation of the Sod shock tube over one position and one velocity axis.

    The density relaxes at every position towards the maximum-entropy density with its density, momentum, and energy,
    and the gas starts from the two states of the shock tube of Sod on either side of the middle of a unit interval. As
    the Knudsen number falls the density moment tends to the Euler solution with the adiabatic index 3 of one velocity
    dimension.

    Args:
        positions:
            The number of positions.

        velocities:
            The number of points of the velocity axis.

        knudsen:
            The Knudsen number, the relaxation time.

        half_width:
            The half-width of the velocity axis.

    Returns:
        system:
            The system, with the density `gas.f`.

    Examples:
        >>> from skcomplex.systems.physics.fluids import bgk_equation
        >>> bgk_equation(8, 16).elements[0].prepare().size_
        128
    """
    position = LatticeSpace((positions,), spacing=1.0 / positions, boundary='reflecting')
    space = PhaseSpace(position, [AxisSpec('v', velocities, -half_width, half_width)]).prepare()
    left = np.arange(positions) < positions // 2
    density = np.where(left, 1.0, 0.125)
    temperature = np.where(left, 1.0, 0.8)
    values = _get_maxwellian(density, np.zeros(positions), temperature, space.coordinates_['v'])
    gas = FieldElement(
        'gas',
        element=DensityElement,
        space=space,
        values={'f': values},
        discretization='finite-volume',
        visual=VLASOV_VISUAL,
    )
    stream = AdvectionInteraction('stream', target='gas', state='f', drift=_get_streaming)
    relax = BhatnagarGrossKrookInteraction('relax', target='gas', axes=('v',), time=knudsen)
    return System(elements=[gas], interactions=[stream, relax])


@register_system(
    field='physics/fluids',
    tags=('field', 'kinetic', 'phase space', 'plasma', 'pde'),
    references=('Bellomo et al. 2017', 'Cheng and Knorr 1976'),
    defaults={'until': 30.0, 'dt': 0.25, 'solver': SemiLagrangianSolver(0.25)},
)
def vlasov_equation(
    positions: int = 32,
    velocities: int = 64,
    *,
    setup: str = 'landau',
    amplitude: float = 0.01,
    half_width: float = 6.0,
) -> System:
    """Return the Vlasov-Poisson equation of electrons over a neutralizing background, on a periodic position axis.

    The electric field is solved from the density moment at every step. The set-up `'landau'` starts from a Maxwellian
    with a density wave of wave number 0.5, whose electric energy decays at twice the Landau damping rate 0.1533. The
    set-up `'two-stream'` starts from two beams of temperature 0.25 at the velocities -1.5 and 1.5, which are unstable
    at the wave number 0.5.

    Args:
        positions:
            The number of positions.

        velocities:
            The number of points of the velocity axis.

        setup:
            `'landau'` or `'two-stream'`.

        amplitude:
            The relative amplitude of the initial density wave.

        half_width:
            The half-width of the velocity axis.

    Returns:
        system:
            The system, with the density `gas.f`.

    Raises:
        ValidationError:
            The set-up is unknown.

    Examples:
        >>> from skcomplex.systems.physics.fluids import vlasov_equation
        >>> vlasov_equation(8, 16).elements[0].prepare().size_
        128
    """
    if setup not in SETUPS:
        error_msg = f'Unknown set-up {setup!r} of the Vlasov equation. Use one of {", ".join(SETUPS)}.'
        raise ValidationError(error_msg)
    length = 4 * math.pi
    space = PhaseSpace(
        LatticeSpace((positions,), spacing=length / positions),
        [AxisSpec('v', velocities, -half_width, half_width)],
    )
    space.prepare()
    x = np.arange(positions) * length / positions
    v = space.coordinates_['v']
    profile = np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi)
    if setup == 'two-stream':
        profile = (np.exp(-2 * (v - 1.5) ** 2) + np.exp(-2 * (v + 1.5) ** 2)) * np.sqrt(2 / np.pi) / 2
    gas = FieldElement(
        'gas',
        element=DensityElement,
        space=space,
        values={'f': np.outer(1.0 + amplitude * np.cos(0.5 * x), profile)},
        discretization='finite-volume',
        visual=VLASOV_VISUAL,
    )
    return System(
        elements=[gas],
        interactions=[AdvectionInteraction('vlasov', target='gas', state='f', drift=_get_poisson_drift)],
    )
