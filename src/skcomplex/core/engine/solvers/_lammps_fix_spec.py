"""Define the lammps fix spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ...base import BaseSpec
from ...types import Array


@dataclass(frozen=True)
class LammpsFixSpec(BaseSpec):
    """A spec of the values that the native fix of an interaction reads in LAMMPS.

    It names the fix, such as `'langevin'` for a heat bath, the group of atoms it acts on, and functions that give a
    value by name, the mass, and the seed, which the LAMMPS solver reads when it writes the command of the fix. It sets
    the heat bath of a liquid, the drag of a colloid, and the push of active particles.

    Read more in the user guide, under Solvers.

    Args:
        name:
            The unique name of the fix.

        group:
            The LAMMPS group of the atoms the fix acts on.

        value:
            The function that returns the current value of a parameter of the interaction by name.

        mass:
            The function that returns the mass of the atoms of the interaction's target.

        seed:
            The function that draws a random seed.

    See Also:
        LammpsSolver: Writes the commands of the fixes.

    Examples:
        >>> from skcomplex.core.engine.solvers import LammpsFixSpec
        >>> fix = LammpsFixSpec('langevin', 'gas', lambda name: 1.0, lambda: 1.0, lambda: 7)
        >>> fix.name, fix.group
        ('langevin', 'gas')
    """

    name: str
    group: str
    value: Callable[[str], Array]
    mass: Callable[[], float]
    seed: Callable[[], int]


type FixCommand = Callable[[LammpsFixSpec], str]
