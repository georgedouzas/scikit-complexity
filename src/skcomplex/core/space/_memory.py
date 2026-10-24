"""Define memory sets, the ordered group elements whose cells a cell of a cellular automaton on a group reads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import ClassVar, Protocol

from ..base import Requires, SpaceKind


class _Generated(Protocol):
    """Group given by the names of its generators."""

    generators: tuple[str, ...]


@dataclass(frozen=True)
class Memory:
    """Ordered memory set of a cellular automaton on a group space: the group elements whose values a rule reads.

    A cell at the element `g` reads the cell at `g s` for every element `s` of the memory set, in its order, so the
    neighbourhood keeps its orientation. Elements are words in the generators, such as `'a'`, `'a^-1'`, or `'a*b'`,
    and `'1'` is the identity, the cell itself.

    Args:
        elements:
            The words of the memory set.

    Examples:
        >>> from skcomplex.core.space import Memory
        >>> len(Memory(('1', 'a', 'a^-1')))
        3
    """

    requires: ClassVar[Requires] = Requires(space={SpaceKind.LATTICE})

    elements: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'elements', tuple(str(element) for element in self.elements))

    def __len__(self) -> int:
        """Return the number of elements."""
        return len(self.elements)

    @classmethod
    def generators(
        cls,
        group: _Generated,
        *,
        center: bool = True,
        inverses: bool = True,
    ) -> 'Memory':
        """Return the memory set of the generators of a group, after the identity and with their inverses if asked.

        Args:
            group:
                The group.

            center:
                Whether the identity comes first.

            inverses:
                Whether every generator is followed by its inverse.

        Returns:
            memory:
                The memory set.

        Examples:
            >>> from skcomplex.formalisms.groups import finitely_presented_group
            >>> from skcomplex.core.space import Memory
            >>> Memory.generators(finitely_presented_group(['a', 'b'])).elements
            ('1', 'a', 'a^-1', 'b', 'b^-1')
        """
        elements = [word for name in group.generators for word in ((name, f'{name}^-1') if inverses else (name,))]
        return cls(('1', *elements) if center else tuple(elements))
