"""Add the agents, their rules, and the space they live in to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

from sage.all__sagemath_symbolics import CC

from ....components.elements import VisualSpec
from ....components.laws._rule import _RuleLaw
from ....components.spaces import BaseSpace
from ....enums import VariableKind
from ....errors import BackendError, LabelError, ValidationError
from ..._utils import declarations_of
from ...problems._base_problem_spec import _CompiledAgents
from ...problems._kernel_spec import KernelSpec
from .._base import BasePass
from .._compilation import Compilation, _Block, _Node
from ._layout import LayoutPass
from ._problem import ProblemPass

AGENT_VIEW_NAMES = frozenset(
    {
        'id',
        'group',
        'group_label',
        'location',
        'location_key',
        'alive',
        'get_neighbor_cells',
        'get_neighbors',
        'get_agents_here',
        'move_to',
        'get_regions_here',
        'remove',
        'create',
        'get_displacement_to',
        'get_distance_to',
        'get_lags',
        'get_referenced',
        'get_value',
        'set_value',
    },
)
MESA_AGENT_NAMES = frozenset({'position', 'pos', 'cell', 'unique_id', 'system'})
AGENT_RESERVED_NAMES = AGENT_VIEW_NAMES | MESA_AGENT_NAMES


class AgentsProblemPass(BasePass):
    """A pass that adds the agents, their rules, and their space to the problem.

    It gathers every agent group with its states and rules, and the rules of the other components, such as a market that
    clears after the traders act, and checks that every agent group lives in one space.

    Read more in the user guide, under Extend the library.

    See Also:
        AgentsPass: Finds the agents.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.economics.econophysics import BoltzmannWealthSystem
        >>> compilation = PassCompiler().prepare(BoltzmannWealthSystem()).run().compilation_
        >>> [agents.path for agents in compilation.problem.agents]
        ['traders']
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'layout', 'reduced', 'rules')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the agents and the rules to the problem."""
        compilation = self.compilation_
        blocks = list(Compilation._get_required(compilation.layout, 'layout').blocks.values())
        rules = Compilation._get_required(compilation.rules, 'rules')
        agents = {block.node.path for block in blocks if block.node.is_agent}
        compilation.problem = replace(
            Compilation._get_required(compilation.problem, 'problem'),
            agents=self._get_compiled_agents(blocks, rules),
            system_rules=tuple((spec.owner, law) for spec, items in rules for law in items if spec.owner not in agents),
            agent_space=self._get_agent_space(blocks),
        )

    @staticmethod
    def _check_agent_names(node: _Node) -> None:
        """Raise an error for an agent state or parameter named like an attribute.

        The attribute is one of the agent view or of Mesa.
        """
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
    def _get_agent_visual(node: _Node) -> VisualSpec:
        """Return the visual of agents and the points of the agents without one.

        The visual is their own or that of their element type.
        """
        visual = ProblemPass._get_visual_of(node)
        return VisualSpec('agents') if visual == VisualSpec() else visual

    @staticmethod
    def _get_agent_space(blocks: list[_Block]) -> BaseSpace | None:
        """Return the one space every agent group lives in.

        Groups that live in different spaces raise an error.
        """
        spaces = {id(block.node.space): block.node.space for block in blocks if block.node.is_agent}
        if len(spaces) > 1:
            error_msg = 'Agent groups of one system must live in the same space.'
            raise ValidationError(error_msg)
        return next(iter(spaces.values()), None)

    @staticmethod
    def _get_compiled_agents(
        blocks: list[_Block],
        rules: list[tuple[KernelSpec, tuple[_RuleLaw, ...]]],
    ) -> tuple[_CompiledAgents, ...]:
        """Return the agent elements and groups with their states, values, and rules.

        The values are the initial values and the parameters.
        """
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
                _CompiledAgents(
                    path=node.path,
                    element=node.as_element,
                    size=node.members or 1,
                    states=states,
                    values={name: LayoutPass._get_value_of(node, name) for name in states},
                    parameters={name: entry.name for name, entry in block.entries.items()},
                    rules=by_owner.get(node.path, ()),
                    visual=AgentsProblemPass._get_agent_visual(node),
                    objects=tuple(
                        name for name, item in declared.items() if item.domain is None or item.domain is object
                    ),
                    roles={name: item.role for name, item in declared.items() if item.role is not None},
                ),
            )
        return tuple(agents)
