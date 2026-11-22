"""Find the elements and groups that run as agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import re

from ....components.elements import BaseElement
from ....components.interactions import BaseInteraction
from ....components.laws._rule import _RuleLaw
from ....enums import VariableKind
from ..._utils import binding_of, bound_labels, declarations_of, probe_laws
from ...problems._group_element import _GroupElement
from ...problems._node_spec import _NodeSpec
from .._base import BasePass
from .._compilation import Compilation

MEMBER_SUFFIX = re.compile(r'\[-?\d+\]$')


class AgentsPass(BasePass):
    """A pass that finds the elements and groups that run as agents.

    An agent is a member of a group that runs rules, such as a trader that gives a coin to another trader. Groups that
    the rules only read and change are agents too, as passive agents. A single element, or a group on the sites or edges
    of a space, is never a passive agent.

    Read more in the user guide, under Extend the library.

    See Also:
        AgentsProblemPass: Adds the agents to the problem.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.economics.econophysics import BoltzmannWealthSystem
        >>> compilation = PassCompiler().prepare(BoltzmannWealthSystem()).compilation_
        >>> [node.path for node in compilation.nodes if node.agent]
        ['traders']
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('nodes',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('nodes',)

    def _run(self, inputs: object = None) -> None:
        """Mark the agent nodes."""
        compilation = self.compilation_
        nodes = Compilation._get_required(compilation.nodes, 'nodes')
        agents = self._get_agent_paths(nodes)
        compilation.nodes = [dataclasses.replace(node, agent=node.path in agents) for node in nodes]

    @staticmethod
    def _is_passive(component: object) -> bool:
        """Check whether a component is a passive group off the parts of the space.

        Its element type has no laws of its own.
        """
        return isinstance(component, _GroupElement) and component.on is None and probe_laws(component) == []

    @staticmethod
    def _get_bound_paths(node: _NodeSpec) -> set[str]:
        """Return the paths of the components the roles of an interaction bind.

        The paths have no member indices.
        """
        paths = set()
        for name in declarations_of(node.component, VariableKind.ROLE):
            for item in bound_labels(binding_of(node.component, name)):
                paths.add(MEMBER_SUFFIX.sub('', f'{node.scope}{item}'))
        return paths

    @staticmethod
    def _get_agent_paths(nodes: list[_NodeSpec]) -> set[str]:
        """Return the paths of the rule-running components and the passive groups read.

        The components are elements and groups, and rules read the passive groups.
        """
        agents = {node.path for node in nodes if AgentsPass._runs_rules(node.component) and node.active is None}
        passive = [node for node in nodes if node.path not in agents and AgentsPass._is_passive(node.component)]
        if not passive:
            return agents
        interactions = [
            (node, probe_laws(node.component)) for node in nodes if isinstance(node.component, BaseInteraction)
        ]
        has_rules = bool(agents) or any(
            any(isinstance(law, _RuleLaw) for law in laws or ()) for _, laws in interactions
        )
        if not has_rules:
            return agents
        bound = {
            path
            for node, laws in interactions
            if laws is None or not all(isinstance(law, _RuleLaw) for law in laws)
            for path in AgentsPass._get_bound_paths(node)
        }
        return agents | {node.path for node in passive if node.path not in bound}

    @staticmethod
    def _runs_rules(component: object) -> bool:
        """Check whether a component runs rules of its own.

        It is a variable group, or an element or group with rule laws. Such a component is an agent. The compiler also
        makes agents of the groups whose element type has no laws of its own in a system with rules, unless an
        interaction whose laws are not rules binds them.

        Args:
            component:
                The element, group, interaction, or system.

        Returns:
            runs_rules:
                Whether the component is a variable group, or an element or group whose own laws include an agent rule.
        """
        if isinstance(component, _GroupElement) and component.variable:
            return True
        if not isinstance(component, BaseElement):
            return False
        return any(isinstance(law, _RuleLaw) for law in probe_laws(component) or [])
