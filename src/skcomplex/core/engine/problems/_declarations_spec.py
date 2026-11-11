"""Define the declarations spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property

from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ...specs._entry_spec import EntrySpec
from ...specs._wiener_process_spec import WienerProcessSpec
from ...types import Array
from ._block_spec import BlockSpec

ACTIVE = '_active'
ENDS = ('_source', '_target')
MEMBERS = '_members'


@dataclass(frozen=True)
class DeclarationsSpec(BaseSpec):
    """A spec of the entries, initial values, and blocks declared for the components.

    Args:
        time:
            The time symbol.

        states:
            The entries of the states, in the order of the state vector.

        parameters:
            The entries of the parameters, in the order of the parameter vector.

        initial_states:
            The flat initial value of every state entry.

        initial_parameters:
            The flat initial value of every parameter entry.

        blocks:
            The block of every element, group, and interaction by path.

        processes:
            The block of every Wiener process that the noise roles of interactions bind, by path.

        handles:
            The path of the Wiener process and the component of every handle symbol.

        agents:
            The path of the agent group and the variable of every symbol of agents.
    """

    time: Expression
    states: tuple[EntrySpec, ...] = ()
    parameters: tuple[EntrySpec, ...] = ()
    initial_states: tuple[Array, ...] = ()
    initial_parameters: tuple[Array, ...] = ()
    blocks: Mapping[str, BlockSpec] = field(default_factory=dict)
    processes: Mapping[str, BlockSpec] = field(default_factory=dict)
    handles: Mapping[Expression, tuple[str, int]] = field(default_factory=dict)
    agents: Mapping[Expression, tuple[str, str]] = field(default_factory=dict)

    @property
    def ends(self) -> dict[bool, int]:
        """Sizes of the state vector and of the parameter vector.

        The size of the state vector is under `True`, and that of the parameter vector under `False`.
        """
        return {True: sum(entry.size for entry in self.states), False: sum(entry.size for entry in self.parameters)}

    @property
    def wieners(self) -> dict[str, WienerProcessSpec]:
        """Wiener processes that the noise roles of interactions bind, by path."""
        return {
            path: block.node.component
            for path, block in self.processes.items()
            if isinstance(block.node.component, WienerProcessSpec)
        }

    @property
    def members(self) -> int:
        """Number of members of the groups, leaving out agents.

        The number chooses a default backend.
        """
        return sum(block.node.members or 0 for block in self.blocks.values() if not block.node.is_agent)

    @cached_property
    def capacity(self) -> dict[str, BlockSpec]:
        """Blocks of the variable groups on the sites or edges of networks, by path."""
        return {path: block for path, block in self.blocks.items() if block.node.active is not None}

    @cached_property
    def groups(self) -> frozenset[str]:
        """Paths of the declared groups."""
        return frozenset(path for path, block in self.blocks.items() if block.node.members is not None)

    def get_changing_edges(self, space: object) -> dict[str | None, BlockSpec]:
        """Return the variable edge group of every layer of a network whose edges change.

        Args:
            space:
                The network space.

        Returns:
            groups:
                The block of the variable edge group by layer.
        """
        return {
            getattr(block.node.component, 'layer', None): block
            for block in self.capacity.values()
            if getattr(block.node.component, 'on', None) == 'edges' and block.node.space is space
        }

    def get_changing_entries(self, space: object) -> dict[str | None, tuple[EntrySpec, EntrySpec, EntrySpec]]:
        """Return the source, target, and activity entries of every changing layer.

        The entries are those of the variable edge group of the layer.

        Args:
            space:
                The network space.

        Returns:
            entries:
                The entries by layer.
        """
        return {
            layer: (block.entries[ENDS[0]], block.entries[ENDS[1]], block.entries[ACTIVE])
            for layer, block in self.get_changing_edges(space).items()
        }
