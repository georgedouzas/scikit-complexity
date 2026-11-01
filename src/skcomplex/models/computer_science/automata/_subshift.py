"""Subshift of finite type: configurations of a space without forbidden patterns."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ....core.spaces import BaseSpace
from ....errors import ValidationError
from ....types import Indices, IntArray
from ._automaton import Reach, _codes, _count, _size, _table


@dataclass(frozen=True)
class Subshift:
    """Subshift of finite type: the configurations in which no forbidden pattern appears around any site.

    A pattern is a tuple of values in the order of a neighbourhood. It appears at a site when the cells of the
    neighbourhood of the site hold its values, and patterns that reach outside a space are not checked, so a lattice
    with fixed edges holds the words of a window.

    Args:
        neighborhood:
            The neighbourhood of offsets on a lattice, or the memory set on a group space, that patterns cover.

        forbidden:
            The forbidden patterns.

        states:
            The number of values of a cell.

    Examples:
        >>> from skcomplex.core.neighborhoods import LatticeNeighborhood
        >>> from skcomplex.core.spaces import LatticeSpace
        >>> from skcomplex.models.computer_science.automata import Subshift
        >>> golden = Subshift(LatticeNeighborhood(((0,), (1,))), [(1, 1)])
        >>> golden.count(LatticeSpace(6, boundary='fixed'))
        21
    """

    neighborhood: Reach
    forbidden: Sequence[tuple[int, ...]]
    states: int = 2

    def _forbidden_codes(self) -> IntArray:
        """Return the codes of the forbidden patterns."""
        patterns = np.asarray([tuple(pattern) for pattern in self.forbidden], dtype=np.int64).reshape(
            -1,
            _count(self.neighborhood),
        )
        return _codes(patterns, self.states)

    def _complete(self, space: BaseSpace) -> tuple[Indices, npt.NDArray[np.bool_]]:
        """Return the neighbour table of a space and the sites whose patterns lie inside it."""
        table = _table(space, self.neighborhood)
        return table, (table < _size(space)).all(axis=1)

    def violations(self, configuration: npt.ArrayLike, space: BaseSpace) -> npt.NDArray[np.bool_]:
        """Return the sites at which a forbidden pattern appears.

        Args:
            configuration:
                The values of the sites.

            space:
                The lattice or the group space.

        Returns:
            violations:
                One flag per site.
        """
        table, complete = self._complete(space)
        values = np.append(np.asarray(configuration, dtype=np.int64), 0)
        codes = _codes(values[table], self.states)
        return complete & np.isin(codes, self._forbidden_codes())

    def is_admissible(self, configuration: npt.ArrayLike, space: BaseSpace) -> bool:
        """Check whether no forbidden pattern appears in a configuration.

        Args:
            configuration:
                The values of the sites.

            space:
                The lattice or the group space.

        Returns:
            admissible:
                Whether the configuration belongs to the subshift.
        """
        return not self.violations(configuration, space).any()

    def count(self, space: BaseSpace) -> int:
        """Return the number of admissible configurations of a space, by dynamic programming over its sites.

        The sites are assigned in order, and the counts of completions are shared between assignments that agree on
        the earlier sites that later patterns read, so a window of a one-dimensional subshift costs linear time.

        Args:
            space:
                The lattice or the group space, a window of the subshift.

        Returns:
            count:
                The number of configurations without a forbidden pattern.
        """
        table, complete = self._complete(space)
        forbidden = {tuple(int(item) for item in pattern) for pattern in self.forbidden}
        checks: list[list[int]] = [[] for _ in range(_size(space))]
        for site in np.flatnonzero(complete):
            checks[int(table[site].max())].append(int(site))
        needed = [
            tuple(
                sorted(
                    {int(cell) for later in checks[site:] for anchor in later for cell in table[anchor] if cell < site},
                ),
            )
            for site in range(_size(space) + 1)
        ]
        values = [0] * _size(space)
        memo: dict[tuple[int, tuple[int, ...]], int] = {}

        def extend(site: int) -> int:
            """Return the number of admissible completions of the values before a site."""
            if site == _size(space):
                return 1
            key = (site, tuple(values[cell] for cell in needed[site]))
            if key not in memo:
                total = 0
                for value in range(self.states):
                    values[site] = value
                    if all(tuple(values[cell] for cell in table[anchor]) not in forbidden for anchor in checks[site]):
                        total += extend(site + 1)
                memo[key] = total
            return memo[key]

        return extend(0)

    def sample(
        self,
        space: BaseSpace,
        sweeps: int = 100,
        *,
        initial: npt.ArrayLike | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> IntArray:
        """Return an admissible configuration drawn by a Markov chain of single-cell moves that keep it admissible.

        Args:
            space:
                The lattice or the group space.

            sweeps:
                The number of sweeps, each proposing a random value at every site in a random order.

            initial:
                The admissible initial configuration, or `None` for every cell at 0.

            random_state:
                The seed or generator of the moves.

        Returns:
            configuration:
                The admissible configuration.

        Raises:
            ValidationError:
                The initial configuration is not admissible.
        """
        rng = np.random.default_rng(random_state)
        values = (
            np.zeros(_size(space), dtype=np.int64) if initial is None else np.asarray(initial, dtype=np.int64).copy()
        )
        if not self.is_admissible(values, space):
            error_msg = 'The initial configuration holds a forbidden pattern. Give an admissible one.'
            raise ValidationError(error_msg)
        table, complete = self._complete(space)
        touching: list[list[int]] = [[] for _ in range(_size(space))]
        for anchor in np.flatnonzero(complete):
            for cell in set(table[anchor].tolist()):
                touching[cell].append(int(anchor))
        forbidden = {tuple(int(item) for item in pattern) for pattern in self.forbidden}
        for _ in range(sweeps):
            for site in rng.permutation(_size(space)):
                previous = values[site]
                values[site] = rng.integers(self.states)
                if any(tuple(values[table[anchor]].tolist()) in forbidden for anchor in touching[site]):
                    values[site] = previous
        return values
