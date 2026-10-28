"""Compile agent-based models: the rules of agents and matchings, which the agent stepper runs on Mesa."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import re
from typing import cast

from ....core.elements import BaseElement, GroupElement
from ....core.interactions import BaseInteraction
from ....core.laws import MatchLaw, RuleLaw
from ..._utils import declarations_of
from ...problems import VariableKind
from .._base import BasePass
from .._probe import probe_laws
from .._tree import Node

_MEMBER_SUFFIX = re.compile(r'\[-?\d+\]$')


class AgentsPass(BasePass):
    """Pass that marks the nodes of the elements and groups that run on the agent stepper.

    An agent is a member of a variable group, or of an element or group whose own laws include a `RuleLaw`. When the
    tree holds rules, of components or of interactions, the groups whose element type has no laws of its own are agents
    too, as passive agents that rules read and change, unless an interaction whose laws are not all rules or matchings
    binds them through a role. Single elements, such as the holders of reductions, and groups on the sites, edges, or
    hyperedges of a space are never passive agents.
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
        nodes = cast('list[Node]', compilation.nodes)
        agents = self._agent_paths(nodes)
        compilation.nodes = [dataclasses.replace(node, agent=node.path in agents) for node in nodes]

    @staticmethod
    def _is_passive(component: object) -> bool:
        """Check whether a component is a group off the parts of the space whose element type has no laws of its own."""
        return isinstance(component, GroupElement) and component.on is None and probe_laws(component) == []

    @staticmethod
    def _bound_paths(node: Node) -> set[str]:
        """Return the paths of the components the roles of an interaction bind, without member indices."""
        paths = set()
        for name in declarations_of(type(node.component), VariableKind.ROLE):
            value = getattr(node.component, name)
            for item in [value] if isinstance(value, str) else value if isinstance(value, list | tuple) else ():
                paths.add(_MEMBER_SUFFIX.sub('', f'{node.scope}{item}'))
        return paths

    @staticmethod
    def _agent_paths(nodes: list[Node]) -> set[str]:
        """Return the paths of the rule-running elements and groups, and of the passive groups that rules read."""
        agents = {node.path for node in nodes if AgentsPass._runs_rules(node.component) and node.active is None}
        passive = [node for node in nodes if node.path not in agents and AgentsPass._is_passive(node.component)]
        if not passive:
            return agents
        interactions = [
            (node, probe_laws(node.component)) for node in nodes if isinstance(node.component, BaseInteraction)
        ]
        has_rules = bool(agents) or any(any(isinstance(law, RuleLaw) for law in laws or ()) for _, laws in interactions)
        if not has_rules:
            return agents
        bound = {
            path
            for node, laws in interactions
            if laws is None or not all(isinstance(law, RuleLaw | MatchLaw) for law in laws)
            for path in AgentsPass._bound_paths(node)
        }
        return agents | {node.path for node in passive if node.path not in bound}

    @staticmethod
    def _runs_rules(component: object) -> bool:
        """Check whether a component runs rules of its own: a variable group, or an element or group with rule laws.

        Such a component is an agent. The compiler also makes agents of the groups whose element type has no laws of its
        own in a system with rules, unless an interaction whose laws are not rules binds them.

        Args:
            component:
                The element, group, interaction, or system.

        Returns:
            runs_rules:
                Whether the component is a variable group, or an element or group whose own laws include a `RuleLaw`.

        Examples:
            >>> from skcomplex.core.elements import Element
            >>> from skcomplex.core.laws import RuleLaw
            >>> from skcomplex.engine.compile.passes._agents import AgentsPass
            >>> class WalkerElement(Element):
            ...     def _get_laws(self, namespace):
            ...         return [RuleLaw(lambda agent, model: None)]
            >>> AgentsPass._runs_rules(WalkerElement('walker')), AgentsPass._runs_rules(Element('plain'))
            (True, False)
        """
        if isinstance(component, GroupElement) and component.variable:
            return True
        if not isinstance(component, BaseElement):
            return False
        return any(isinstance(law, RuleLaw) for law in probe_laws(component) or [])
