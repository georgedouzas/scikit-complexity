"""Compile agent-based models: the rules of agents and matchings, which the agent stepper runs on Mesa."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import dataclasses
import re
from dataclasses import replace
from typing import cast

from ...base import BackendError, Domain, LabelError, ValidationError, VariableKind, VisualSpec, declarations_of
from ...compile import (
    Block,
    BlockLaws,
    Compilation,
    CompiledAgents,
    Declarations,
    KernelSpec,
    Laws,
    Node,
    Pass,
    Problem,
    visual_of,
)
from ...components import Group, probe_laws
from ...interactions import Interaction
from ...laws import Invariant, Law
from ...space import Space
from ._laws import Match, Rule, runs_rules

MEMBER_SUFFIX = re.compile(r'\[-?\d+\]$')
AGENT_VIEW_NAMES = frozenset(
    {
        'id',
        'group',
        'group_label',
        'location',
        'alive',
        'neighbor_cells',
        'neighbors',
        'agents_here',
        'move_to',
        'remove',
        'create',
        'displacement_to',
        'distance_to',
    },
)
MESA_AGENT_NAMES = frozenset({'position', 'pos', 'cell', 'unique_id', 'model'})
AGENT_RESERVED_NAMES = AGENT_VIEW_NAMES | MESA_AGENT_NAMES


def _check_agent_names(node: Node) -> None:
    """Raise an error for a state or parameter of an agent named like an attribute of the agent view or of Mesa."""
    declared = [
        *declarations_of(node.component_type, VariableKind.STATE),
        *declarations_of(node.component_type, VariableKind.PARAMETER),
    ]
    clashes = sorted(name for name in declared if name in AGENT_RESERVED_NAMES)
    if clashes:
        error_msg = (
            f'Agents of {node.path} declare the variables {", ".join(clashes)}, whose names the agent view or Mesa '
            f'reserves. Rename them, since {", ".join(sorted(AGENT_RESERVED_NAMES))} are reserved names of agents.'
        )
        raise LabelError(error_msg)


def _agent_visual(node: Node) -> VisualSpec:
    """Return the visual of agents, their own or their element type's, and the agents' points without one."""
    visual = visual_of(node)
    return VisualSpec('agents') if visual == VisualSpec() else visual


def _agent_space(blocks: list[Block]) -> Space | None:
    """Return the one space every agent group lives in, or raise an error when groups live in different spaces."""
    spaces = {id(block.node.space): block.node.space for block in blocks if block.node.is_agent}
    if len(spaces) > 1:
        error_msg = 'Agent groups of one system must live in the same space.'
        raise ValidationError(error_msg)
    return next(iter(spaces.values()), None)


def _agent_owner(spec: KernelSpec, law: Law, agents: set[str]) -> str | None:
    """Return the group a rule or a matching acts for: the first agent group among its roles, or its first role."""
    if isinstance(law, Match):
        law = law.rule
    if not isinstance(law, Rule):
        return None
    paths = [path for value in spec.roles.values() for path in ([value] if isinstance(value, str) else value)]
    return next((path for path in paths if path in agents), next(iter(paths), spec.owner))


def _compiled_agents(
    blocks: list[Block],
    rules: list[tuple[KernelSpec, tuple[Rule, ...]]],
    specs: Laws,
) -> tuple[CompiledAgents, ...]:
    """Return the agent elements and groups with their states, initial values, parameters, and rules."""
    by_owner = {spec.owner: items for spec, items in rules}
    invariants = {
        spec.owner: tuple(law for law in laws if isinstance(law, Invariant) and law.is_member) for spec, laws in specs
    }
    agents = []
    for block in blocks:
        node = block.node
        if not node.is_agent:
            continue
        _check_agent_names(node)
        declared = declarations_of(node.component_type, VariableKind.STATE)
        complex_states = [name for name, declaration in declared.items() if declaration.domain is Domain.COMPLEX]
        if complex_states:
            error_msg = (
                f'Agent group {node.path} declares the complex state {complex_states[0]}, which the Mesa backend '
                'cannot run. Agent states take real domains, so keep the real and imaginary parts as two states.'
            )
            raise BackendError(error_msg)
        states = tuple(declared)
        agents.append(
            CompiledAgents(
                path=node.path,
                element=node.component,
                size=node.members or 1,
                states=states,
                values={name: node.value_of(name) for name in states},
                parameters={name: entry.name for name, entry in block.entries.items()},
                rules=by_owner.get(node.path, ()),
                visual=_agent_visual(node),
                invariants=invariants.get(node.path, ()),
            ),
        )
    return tuple(agents)


def _is_passive(component: object) -> bool:
    """Check whether a component is a group off the parts of the space whose element type has no laws of its own."""
    return isinstance(component, Group) and component.on is None and probe_laws(component) == []


def _bound_paths(node: Node) -> set[str]:
    """Return the paths of the components the roles of an interaction bind, without member indices."""
    paths = set()
    for name in declarations_of(type(node.component), VariableKind.ROLE):
        value = getattr(node.component, name)
        for item in [value] if isinstance(value, str) else value or ():
            paths.add(MEMBER_SUFFIX.sub('', f'{node.scope}{item}'))
    return paths


def _agent_paths(nodes: list[Node]) -> set[str]:
    """Return the paths of the rule-running elements and groups, and of the passive groups that rules read."""
    agents = {node.path for node in nodes if runs_rules(node.component) and node.active is None}
    passive = [node for node in nodes if node.path not in agents and _is_passive(node.component)]
    if not passive:
        return agents
    interactions = [(node, probe_laws(node.component)) for node in nodes if isinstance(node.component, Interaction)]
    has_rules = bool(agents) or any(any(isinstance(law, Rule) for law in laws or ()) for _, laws in interactions)
    if not has_rules:
        return agents
    bound = {
        path
        for node, laws in interactions
        if laws is None or not all(isinstance(law, Rule | Match) for law in laws)
        for path in _bound_paths(node)
    }
    return agents | {node.path for node in passive if node.path not in bound}


class AgentsPass(Pass):
    """Pass that marks the nodes of the elements and groups that run on the agent stepper.

    An agent is a member of a variable group, or of an element or group whose own laws include a `Rule`. When the tree
    holds rules, of components or of interactions, the groups whose element type has no laws of its own are agents
    too, as passive agents that rules read and change, unless an interaction whose laws are not all rules or matchings
    binds them through a role. Single elements, such as aggregates, and groups on the sites, edges, or hyperedges of a
    space are never passive agents.
    """

    reads = ('nodes',)
    writes = ('nodes',)

    def run(self, compilation: Compilation) -> None:
        """Mark the agent nodes.

        Args:
            compilation:
                The compilation state.
        """
        nodes = cast('list[Node]', compilation.nodes)
        agents = _agent_paths(nodes)
        compilation.nodes = [dataclasses.replace(node, agent=node.path in agents) for node in nodes]


class UnpackMatchesPass(Pass):
    """Pass that replaces every matching by the rule it runs."""

    reads = ('collected',)
    writes = ('collected',)

    def run(self, compilation: Compilation) -> None:
        """Replace the matchings by their rules.

        Args:
            compilation:
                The compilation state.
        """
        compilation.collected = [
            (spec, [law.rule if isinstance(law, Match) else law for law in laws], block)
            for spec, laws, block in cast('BlockLaws', compilation.collected)
        ]


class SetRulesApartPass(Pass):
    """Pass that takes the rules out of the laws lowered into kernels, since the agent stepper runs them."""

    reads = ('reduced',)
    writes = ('reduced', 'rules')

    def run(self, compilation: Compilation) -> None:
        """Take the rules out of the reduced laws.

        Args:
            compilation:
                The compilation state.
        """
        reduced = cast('Laws', compilation.reduced)
        compilation.facts['rules'] = [
            (spec, tuple(law for law in laws if isinstance(law, Rule))) for spec, laws in reduced
        ]
        compilation.reduced = [(spec, [law for law in laws if not isinstance(law, Rule)]) for spec, laws in reduced]


class AgentsProblemPass(Pass):
    """Pass that adds the agents with their rules, the rules of other components, and the agent space to the problem."""

    reads = ('problem', 'layout', 'reduced', 'rules')
    writes = ('problem',)

    def run(self, compilation: Compilation) -> None:
        """Add the agents and the rules to the problem.

        Args:
            compilation:
                The compilation state.
        """
        blocks = list(cast('Declarations', compilation.layout).blocks.values())
        rules = cast('list[tuple[KernelSpec, tuple[Rule, ...]]]', compilation.facts['rules'])
        reduced = cast('Laws', compilation.reduced)
        agents = {block.node.path for block in blocks if block.node.is_agent}
        compilation.problem = replace(
            cast('Problem', compilation.problem),
            agents=_compiled_agents(blocks, rules, reduced),
            model_rules=tuple((spec.owner, law) for spec, items in rules for law in items if spec.owner not in agents),
            model_invariants=tuple(
                (spec.owner, law)
                for spec, laws in reduced
                for law in laws
                if isinstance(law, Invariant) and law.is_member and spec.owner not in agents
            ),
            agent_space=_agent_space(blocks),
        )
