"""View of a neuron of a Hopfield network, typed for the rules that read and set it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.engine.solvers import AgentView
from ....core.types import Array


class NeuronView(AgentView):
    """View of a neuron of a Hopfield network, with its states as attributes."""

    spin: float
    weights: Array
