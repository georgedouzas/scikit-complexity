"""Values the native fix of an interaction reads when the LAMMPS backend writes its command."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ...core.types import Vector


@dataclass(frozen=True)
class LammpsFix:
    """Values the native fix of an interaction reads when the LAMMPS backend writes its command.

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
    """

    name: str
    group: str
    value: Callable[[str], Vector]
    mass: Callable[[], float]
    seed: Callable[[], int]
