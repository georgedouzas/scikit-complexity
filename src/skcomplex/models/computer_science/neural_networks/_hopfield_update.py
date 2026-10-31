"""Define Hopfield updates, the alignment of every neuron in turn with the sign of its weighted input."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np

from ....core.interactions import MapInteraction
from ....core.types import ModelViewProtocol
from ....specs import RoleSpec
from ._neuron_view import NeuronView


class HopfieldUpdateInteraction(MapInteraction):
    """Asynchronous update of a Hopfield network, where every neuron in turn takes the sign of its weighted input.

    The neurons update one after the other in a new random order every step, each reading the states the neurons before
    it set, so the energy `-sum w_ij s_i s_j / 2` never increases. A neuron whose input is zero keeps its state.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of neurons.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import HopfieldUpdateInteraction
        >>> HopfieldUpdateInteraction('update', target='neurons').check().schedule
        'random-sequential'
    """

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()]) -> None:
        super().__init__(
            label,
            target=target,
            state='spin',
            function=self._get_spin,
            schedule='random-sequential',
            form='agent',
        )

    def _get_spin(self, agent: NeuronView, model: ModelViewProtocol) -> float:
        """Return the sign of the weighted input of a neuron from the other neurons, or its state for a zero input."""
        del model
        weights = np.asarray(agent.weights)
        neighbors = [other for other in agent.neighbors() if isinstance(other, NeuronView)]
        field = sum(weights[int(np.asarray(other.location))] * other.spin for other in neighbors)
        return float(np.sign(field)) if field != 0 else float(agent.spin)
