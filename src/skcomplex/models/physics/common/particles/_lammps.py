"""Translate the named particle forces, pair potentials, and thermostats to native LAMMPS pair styles and fixes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

from .....core.interactions import BaseInteraction, ConstantForceInteraction, DampingInteraction
from .....engine.types import PairArguments
from .....specs import LammpsFixSpec
from .....types import Array
from ._langevin_thermostat import LangevinThermostatInteraction
from ._lennard_jones import LennardJonesInteraction


def _lennard_jones(value: Callable[[str], Array]) -> str:
    """Return the coefficients of `lj/cut`."""
    return f'{float(value("epsilon")[0])} {float(value("sigma")[0])}'


def _addforce(fix: LammpsFixSpec) -> str:
    """Return the `fix addforce` command of a constant force."""
    force = [*map(float, fix.value('F')), 0.0, 0.0, 0.0]
    return f'fix {fix.name} {fix.group} addforce {force[0]} {force[1]} {force[2]}'


def _viscous(fix: LammpsFixSpec) -> str:
    """Return the `fix viscous` command of a linear damping."""
    return f'fix {fix.name} {fix.group} viscous {float(fix.value("gamma")[0])}'


def _langevin(fix: LammpsFixSpec) -> str:
    """Return the `fix langevin` command of a Langevin thermostat."""
    temperature, gamma = float(fix.value('temperature')[0]), float(fix.value('gamma')[0])
    return f'fix {fix.name} {fix.group} langevin {temperature} {temperature} {fix.mass() / gamma} {fix.seed()}'


LAMMPS_PAIRS: dict[type[BaseInteraction], tuple[str, PairArguments]] = {
    LennardJonesInteraction: ('lj/cut', _lennard_jones),
}
LAMMPS_FIXES: dict[type[BaseInteraction], Callable[[LammpsFixSpec], str]] = {
    ConstantForceInteraction: _addforce,
    DampingInteraction: _viscous,
    LangevinThermostatInteraction: _langevin,
}
