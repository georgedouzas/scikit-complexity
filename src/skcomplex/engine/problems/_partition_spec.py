"""Define the partition spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...specs._base_spec import BaseSpec
from ._base_problem_spec import BaseProblemSpec


@dataclass(frozen=True)
class PartitionSpec(BaseSpec):
    """Part of a co-simulation problem, compiled with the laws of its own elements.

    The partition's problem has the state layout of the whole system. The states of other partitions have no laws in
    it, so they hold their values while it advances and act as its inputs.

    Args:
        name:
            The name of the partition, the path of its subsystem.

        problem:
            The problem of the partition.

        owned:
            The flat state indices of the partition's own elements.

        owners:
            The paths of the partition's own elements and groups.
    """

    name: str
    problem: BaseProblemSpec
    owned: npt.NDArray[np.intp]
    owners: frozenset[str]
