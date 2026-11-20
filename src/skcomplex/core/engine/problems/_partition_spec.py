"""Define the partition spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...base import BaseConfig
from ...specs._base_spec import BaseSpec
from ._base_problem_spec import BaseProblemSpec


@dataclass(frozen=True)
class PartitionSpec(BaseSpec):
    """A spec of a part of a co-simulation problem, with the laws of its elements.

    The partition's problem has the state layout of the whole system. The states of other partitions have no laws in
    it, so they hold their values while it advances and act as its inputs. The problem of a nested system that another
    compiler compiles, such as a system of words, has a layout of its own instead, which `states` and `parameters`
    place in the vectors of the whole system.

    Args:
        name:
            The name of the partition, the path of its subsystem.

        problem:
            The problem of the partition.

        owned:
            The flat state indices of the partition's own elements.

        owners:
            The paths of the partition's own elements and groups.

        solver:
            The solver of the nested system the partition holds, which it keeps as it keeps its time, or `None` for the
            default solver of its form.

        states:
            The indices of the state vector of the whole system that hold the state vector of the problem, or `None`
            for a problem with the layout of the whole system.

        parameters:
            The indices of the parameter vector of the whole system that hold the parameter vector of the problem, or
            `None` for a problem with the layout of the whole system.
    """

    name: str
    problem: BaseProblemSpec
    owned: npt.NDArray[np.intp]
    owners: frozenset[str]
    solver: BaseConfig | None = None
    states: npt.NDArray[np.intp] | None = None
    parameters: npt.NDArray[np.intp] | None = None
