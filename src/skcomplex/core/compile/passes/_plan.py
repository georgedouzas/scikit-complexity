"""Plan the partitions of a system from the census of its laws and their classes once lowered."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from .._census import Census
from .._compilation import Compilation
from .._pipeline import Pass
from .._planner import plan_partitions


class PlanPass(Pass):
    """Pass that plans the partitions of the system with the registered forms and solvers."""

    reads = ('census', 'kinds')
    writes = ('plan',)

    def run(self, compilation: Compilation) -> None:
        """Plan the partitions of the system.

        Args:
            compilation:
                The compilation state.
        """
        compilation.plan = plan_partitions(
            cast('Census', compilation.census),
            cast('dict[str, set[type]]', compilation.kinds),
        )
