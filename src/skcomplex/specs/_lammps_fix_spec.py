"""Define the lammps fix spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class LammpsFixSpec(BaseSpec):
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
    value: Callable[[str], Array]
    mass: Callable[[], float]
    seed: Callable[[], int]
