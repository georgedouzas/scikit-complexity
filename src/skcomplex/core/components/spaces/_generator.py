"""Define generator neighbourhoods, the memory sets of the generators of a group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ._memory import MemoryNeighborhood


class GeneratorNeighborhood(MemoryNeighborhood):
    """Memory set of the generators of a group.

    The identity comes first, and the inverses follow when asked.

    Args:
        generators:
            The names of the generators of the group.

        center:
            Whether the identity comes first.

        inverses:
            Whether every generator is followed by its inverse.

        label:
            The label of the neighbourhood.

    Attributes:
        elements_:
            The words of the memory set.

    Examples:
        >>> from skcomplex.core.components.spaces import GeneratorNeighborhood
        >>> GeneratorNeighborhood(['a', 'b']).prepare().elements_
        ('1', 'a', 'a^-1', 'b', 'b^-1')
    """

    def __init__(
        self,
        generators: tuple[str, ...],
        *,
        center: bool = True,
        inverses: bool = True,
        label: str = 'neighborhood',
    ) -> None:
        super().__init__((), label=label)
        self.generators = generators
        self.center = center
        self.inverses = inverses

    def _get_elements(self) -> tuple[str, ...]:
        """Return the identity when included, then every generator and its inverse.

        The inverse follows when asked.
        """
        words = [word for name in self.generators for word in ((name, f'{name}^-1') if self.inverses else (name,))]
        return ('1', *words) if self.center else tuple(words)
