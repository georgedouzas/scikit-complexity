"""Synapse that adds the presynaptic trace from a transmission delay ago to the postsynaptic potential."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import RateInteraction
from ....core.components.laws.utils import delayed
from ....core.specs import ParameterSpec, RoleSpec


class DelayedSynapseInteraction(RateInteraction):
    """Synapse that adds the presynaptic trace from a transmission delay ago to the postsynaptic potential.

    The target gains `J s_source(t - D)`, so every presynaptic spike moves the postsynaptic potential by `J` in
    total, spread over the decay time of the trace and arriving after the delay `D`.

    Args:
        label:
            The label of the interaction.

        source:
            The path of the presynaptic neuron.

        target:
            The path of the postsynaptic neuron.

        J:
            The efficacy in mV, positive for excitation and negative for inhibition.

        D:
            The transmission delay in ms.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import DelayedSynapseInteraction
        >>> DelayedSynapseInteraction('synapse', source='a', target='b', J=-1.0).D
        1.5
    """

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        target: Annotated[str, RoleSpec()],
        J: Annotated[float, ParameterSpec()] = 0.5,
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.5,
    ) -> None:
        super().__init__(label, target=target, state='v')
        self.source = source
        self.J = J
        self.D = D

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return the delayed synaptic input `J s_source(t - D)`.

        Args:
            namespace:
                The namespace of the symbols the rate can use, with the namespaces of the source and the target.

        Returns:
            rate:
                The rate of the postsynaptic potential.
        """
        return namespace.J * delayed(namespace.source.s, namespace.D)
