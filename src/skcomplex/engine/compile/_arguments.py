"""Parameters and states that a lattice law reads besides its neighbourhoods, by name."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import Any

from ...types import Array, Indices


@dataclass(frozen=True)
class Arguments:
    """Parameters and states that a lattice law reads besides its neighbourhoods, by name.

    Args:
        names:
            The keyword or symbol names.

        vectors:
            Whether every argument is read from the state vector rather than the parameter vector.

        starts:
            The index of the first component of every argument.

        sizes:
            The number of components of every argument, the number of sites for per-site values or 1 for shared ones.
    """

    names: tuple[str, ...] = ()
    vectors: tuple[bool, ...] = ()
    starts: tuple[int, ...] = ()
    sizes: tuple[int, ...] = ()

    def values(self, x: Array, p: Array, sites: Indices | None = None) -> dict[str, Any]:
        """Return the values of the arguments, per-site values restricted to some sites.

        Args:
            x:
                The state vector.

            p:
                The parameter vector.

            sites:
                The sites to keep, or `None` for all.

        Returns:
            values:
                The value of every argument by name.
        """
        found: dict[str, Any] = {}
        for name, is_state, start, size in zip(self.names, self.vectors, self.starts, self.sizes, strict=True):
            source = x if is_state else p
            if size == 1:
                found[name] = float(source[start])
                continue
            values = source[start : start + size]
            found[name] = values if sites is None else values[sites]
        return found
