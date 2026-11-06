"""Agent with many internal configurations, of which two show a characteristic with the spin `1` or `-1`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, ZZ

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._complex_agent_view import ComplexAgentView


class ComplexAgentElement(Element):
    """Agent with many internal configurations, of which two show a characteristic with the spin `1` or `-1`.

    Configuration 0 has the spin `S = 1`, configuration 1 the spin `S = -1`, and the configurations from 2 to `H + 1`
    the spin 0. In `standardized_agents`, a `MapInteraction` sets the spin after every change of configuration.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        configuration:
            The internal configuration, from 0 to `H + 1`.

        spin:
            The spin of the configuration.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.opinion_dynamics import ComplexAgentElement
        >>> GroupElement(
        ...     'agents', element=ComplexAgentElement, size=10, values={'configuration': 5.0, 'spin': 0.0}
        ... ).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        configuration: Annotated[float, StateSpec(domain=NN)] = 2.0,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.configuration = configuration
        self.spin = spin

    @classmethod
    def _get_view(cls) -> type[ComplexAgentView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return ComplexAgentView
