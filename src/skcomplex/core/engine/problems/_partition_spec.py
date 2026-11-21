"""Define the partition spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...base import BaseConfig, BaseSpec
from ._base_problem_spec import BaseProblemSpec


@dataclass(frozen=True)
class PartitionSpec(BaseSpec):
    """A spec of one subsystem of a system whose subsystems run with their own solvers.

    When the subsystems of a system need different solvers, such as a linear program of a cell inside a growing
    population, each runs as a partition that advances in turn. The partition holds the problem of its subsystem and the
    states it owns, while the states of the other partitions hold their values and act as its inputs. It couples the
    metabolism of a cell with its growth, a controller with its plant, and a market with its traders.

    Read more in the user guide, under Solvers.

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
            The indices of the state vector of the whole system that hold the state vector of the problem, or `None` for
            a problem with the layout of the whole system.

        parameters:
            The indices of the parameter vector of the whole system that hold the parameter vector of the problem, or
            `None` for a problem with the layout of the whole system.

    See Also:
        ProblemSpec: The problem of a partition.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import DynamicFBASystem
        >>> partition = DynamicFBASystem().prepare().simulate(until=0).problem.partitions[0]
        >>> partition.name, sorted(partition.owners)
        ('fba', ['fba/cell'])
    """

    name: str
    problem: BaseProblemSpec
    owned: npt.NDArray[np.intp]
    owners: frozenset[str]
    solver: BaseConfig | None = None
    states: npt.NDArray[np.intp] | None = None
    parameters: npt.NDArray[np.intp] | None = None
