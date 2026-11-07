"""Update of every node of a Boolean network to the sign of the sum of its inputs, flipped with a probability."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import scipy.sparse

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import total, uniform, where
from ....core.specs import RoleSpec


class NoisyMajorityInteraction(Interaction):
    """Update of every node of a Boolean network to the sign of the sum of its inputs, flipped with a probability.

    Node `i` takes `sign(sum_j sigma_ij)` over its inputs with the probability `1 - eta` and the opposite sign with
    the probability `eta`, all nodes at once, the network of Huepe and Aldana-González (2002).

    Args:
        label:
            The label of the interaction.

        nodes:
            The path of the group of nodes.

        inputs:
            The input matrix, dense or sparse, one row per node with 1 for every input, an odd number per row.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import NoisyMajorityInteraction
        >>> NoisyMajorityInteraction('majority', nodes='nodes', inputs=((0.0, 1.0), (1.0, 0.0))).nodes
        'nodes'
    """

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        inputs: Sequence[Sequence[float]] | scipy.sparse.sparray = ((0.0,),),
    ) -> None:
        super().__init__(label)
        self.nodes = nodes
        self.inputs = inputs

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the noisy majority update.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        node = s.nodes
        weights = self.inputs if scipy.sparse.issparse(self.inputs) else np.asarray(self.inputs, dtype=float)
        majority = where(total(node.sigma, weights=weights) > 0, 1, -1)
        return [UpdateLaw(node.sigma, where(uniform('flip') < node.eta, -majority, majority))]
