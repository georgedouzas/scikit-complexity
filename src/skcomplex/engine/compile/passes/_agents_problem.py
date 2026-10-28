"""Pass that adds the agents with their rules, the rules of other components, and the agent space to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from typing import cast

from sage.all__sagemath_symbolics import CC

from ....core.laws import RuleLaw
from ....core.spaces import BaseSpace
from ....errors import BackendError, LabelError, ValidationError
from ....specs import BlockSpec, CompiledAgentsSpec, DeclarationsSpec, KernelSpec, NodeSpec, ProblemSpec, VisualSpec
from ....types import VariableKind
from ..._utils import declarations_of
from .._base import BasePass
from ._layout import LayoutPass
from ._problem import ProblemPass

_AGENT_VIEW_NAMES = frozenset(
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
        'lags',
    },
)


_MESA_AGENT_NAMES = frozenset({'position', 'pos', 'cell', 'unique_id', 'model'})


AGENT_RESERVED_NAMES = _AGENT_VIEW_NAMES | _MESA_AGENT_NAMES


class AgentsProblemPass(BasePass):
    """Pass that adds the agents with their rules, the rules of other components, and the agent space to the problem."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'layout', 'reduced', 'rules')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the agents and the rules to the problem."""
        compilation = self.compilation_
        blocks = list(cast('DeclarationsSpec', compilation.layout).blocks.values())
        rules = cast('list[tuple[KernelSpec, tuple[RuleLaw, ...]]]', compilation.rules)
        agents = {block.node.path for block in blocks if block.node.is_agent}
        compilation.problem = replace(
            cast('ProblemSpec', compilation.problem),
            agents=self._compiled_agents(blocks, rules),
            model_rules=tuple((spec.owner, law) for spec, items in rules for law in items if spec.owner not in agents),
            agent_space=self._agent_space(blocks),
        )

    @staticmethod
    def _check_agent_names(node: NodeSpec) -> None:
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

    @staticmethod
    def _agent_visual(node: NodeSpec) -> VisualSpec:
        """Return the visual of agents, their own or their element type's, and the agents' points without one."""
        visual = ProblemPass._visual_of(node)
        return VisualSpec('agents') if visual == VisualSpec() else visual

    @staticmethod
    def _agent_space(blocks: list[BlockSpec]) -> BaseSpace | None:
        """Return the one space every agent group lives in, or raise an error when groups live in different spaces."""
        spaces = {id(block.node.space): block.node.space for block in blocks if block.node.is_agent}
        if len(spaces) > 1:
            error_msg = 'Agent groups of one system must live in the same space.'
            raise ValidationError(error_msg)
        return next(iter(spaces.values()), None)

    @staticmethod
    def _compiled_agents(
        blocks: list[BlockSpec],
        rules: list[tuple[KernelSpec, tuple[RuleLaw, ...]]],
    ) -> tuple[CompiledAgentsSpec, ...]:
        """Return the agent elements and groups with their states, initial values, parameters, and rules."""
        by_owner = {spec.owner: items for spec, items in rules}
        agents = []
        for block in blocks:
            node = block.node
            if not node.is_agent:
                continue
            AgentsProblemPass._check_agent_names(node)
            declared = declarations_of(node.component_type, VariableKind.STATE)
            complex_states = [name for name, declaration in declared.items() if declaration.domain is CC]
            if complex_states:
                error_msg = (
                    f'Agent group {node.path} declares the complex state {complex_states[0]}, which the Mesa backend '
                    'cannot run. Agent states take real domains, so keep the real and imaginary parts as two states.'
                )
                raise BackendError(error_msg)
            states = tuple(declared)
            agents.append(
                CompiledAgentsSpec(
                    path=node.path,
                    element=node.component,
                    size=node.members or 1,
                    states=states,
                    values={name: LayoutPass._value_of(node, name) for name in states},
                    parameters={name: entry.name for name, entry in block.entries.items()},
                    rules=by_owner.get(node.path, ()),
                    visual=AgentsProblemPass._agent_visual(node),
                ),
            )
        return tuple(agents)
