"""Catalysed replication of the molecules of a group, with a common outflow that keeps the total at 1."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.laws.utils import total
from ....core.specs import ParameterSpec, RoleSpec
from ._replicator import ReplicatorInteraction


class CatalysisInteraction(ReplicatorInteraction):
    """Catalysed replication of the molecules of a group, with a common outflow that keeps the total at 1.

    Molecule `i` replicates at the rate `f_i = lambda_i + omega sum_j T_ij x_j`, with its own growth `lambda_i` and the
    0 or 1 topology `T` of the catalytic links read as the weights of a total over the molecules. The outflow is the
    total replication, so the concentrations follow the replicator equation `dx_i/dt = x_i (f_i - sum_j x_j f_j)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of molecules.

        omega:
            The catalytic rate of a link.

        topology:
            The catalytic links, 1 where molecule `j` catalyses molecule `i`.

    Examples:
        >>> from skcomplex.systems.biology.evolution import CatalysisInteraction
        >>> CatalysisInteraction('catalysis', target='soup', topology=((0, 1), (1, 0))).topology
        ((0, 1), (1, 0))
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        omega: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        topology: Sequence[Sequence[float]] = ((0.0, 1.0), (1.0, 0.0)),
    ) -> None:
        super().__init__(label, target=target)
        self.omega = omega
        self.topology = topology

    def _check_fitness(self) -> None:
        """Leave the fitness unchecked, since the catalysis derives it from its topology."""

    def _get_fitness(self, namespace: SimpleNamespace) -> Expression:
        """Return the catalysed growth `lambda_i + omega sum_j T_ij x_j` of a molecule.

        Args:
            namespace:
                The namespace of the symbols the fitness can use, with the namespace of the target.

        Returns:
            fitness:
                The fitness of the molecule.
        """
        molecule = namespace.target
        fitness: Expression = molecule.growth + namespace.omega * total(
            molecule.x,
            weights=np.asarray(self.topology, dtype=float),
        )
        return fitness
