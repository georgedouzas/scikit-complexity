"""Define the layout spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import re
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import RR

from ...base import BaseSpec
from ...components import EntrySpec
from ...types import Array, ComplexArray

DERIVATIVE = re.compile(r"^(?P<base>.*?)(?P<primes>'+)(?P<index>\[[\w, ]+\])?$")
LAG = re.compile(r'^(?P<base>.*?)\[-(?P<steps>\d+)\](?P<index>\[[\w, ]+\])?$')
COMPANION = re.compile(r'_(?P<kind>d|lag)(?P<count>\d+)(?P<index>\[[\w, ]+\])?$')


@dataclass(frozen=True)
class LayoutSpec(BaseSpec):
    """A spec of the named values stored in one flat vector, in order.

    It names each slice of a flat vector, such as `'population.x'` at index 0, and reads a value by its name, so solvers
    work on plain arrays while users read values by name. It lays out the states of a population, the spins of a magnet,
    and the positions of a gas.

    Read more in the user guide, under Extend the library.

    Args:
        entries:
            The entries, in vector order.

    See Also:
        ProblemSpec: Holds the layouts of its states and parameters.

    Examples:
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> problem = LogisticGrowthSystem().prepare().simulate(until=0).problem
        >>> problem.states.names, problem.states.read(problem.initial_state, 'population.x')
        (['population.x'], 10.0)
    """

    entries: tuple[EntrySpec, ...]

    @staticmethod
    def _get_internal_name(name: str) -> str:
        """Return the engine name of a value, such as `x_d1` for `x'`.

        A past value such as `y[-1]` is `y_lag1`.
        """
        derivative = DERIVATIVE.match(name)
        if derivative is not None:
            return f"{derivative['base']}_d{len(derivative['primes'])}{derivative['index'] or ''}"
        lag = LAG.match(name)
        if lag is not None:
            return f"{lag['base']}_lag{lag['steps']}{lag['index'] or ''}"
        return name

    @staticmethod
    def _get_display_name(name: str) -> str:
        """Return the name users read, such as `x'` for `x_d1` and `y[-1]` for `y_lag1`."""
        companion = COMPANION.search(name)
        if companion is None:
            return name
        base = name[: companion.start()]
        count = int(companion['count'])
        suffix = "'" * count if companion['kind'] == 'd' else f'[-{count}]'
        return f"{base}{suffix}{companion['index'] or ''}"

    @functools.cached_property
    def _by_name(self) -> dict[str, EntrySpec]:
        """Entries by dotted name."""
        return {entry.name: entry for entry in self.entries}

    def has(self, name: str) -> bool:
        """Check whether a variable name is part of the layout.

        Args:
            name:
                The dotted name of a variable, such as `ball.x` or its derivative `ball.x'`.

        Returns:
            found:
                Whether the layout has a variable of that name.
        """
        return self._get_internal_name(name) in self._by_name

    @property
    def size(self) -> int:
        """Number of components of the flat vector."""
        return sum(entry.size for entry in self.entries)

    @property
    def names(self) -> list[str]:
        """Dotted names of the variables, in vector order."""
        return [entry.name for entry in self.entries]

    @property
    def component_names(self) -> list[str]:
        """Names of the components, in vector order."""
        return [name for entry in self.entries for name in entry.component_names]

    def get_entry(self, name: str) -> EntrySpec:
        """Return the entry of a variable.

        Args:
            name:
                The dotted name of the variable, such as `ball.x` or its derivative `ball.x'`.

        Returns:
            entry:
                The entry of the variable.

        Raises:
            KeyError:
                The name is not part of the layout.
        """
        name = self._get_internal_name(name)
        if name not in self._by_name:
            error_msg = f'No variable named {name!r}. Available names are: {", ".join(self.names)}.'
            raise KeyError(error_msg)
        return self._by_name[name]

    def locate(self, name: str) -> tuple[EntrySpec, slice]:
        """Return the entry of a variable or of one of its components.

        It also returns the slice of the vector that the variable covers.

        Args:
            name:
                The dotted name of a variable, or of one component such as `regulation.weights[1]`.

        Returns:
            location:
                The entry of the variable, a scalar entry for a component, and the slice of its values.

        Raises:
            KeyError:
                The name is neither a variable nor a component of the layout.

        Examples:
            >>> from skcomplex.core.components import EntrySpec
            >>> from skcomplex.core.engine.problems import LayoutSpec
            >>> from sage.all__sagemath_symbolics import CC, RR
            >>> layout = LayoutSpec((EntrySpec('a.n', 0, (), RR), EntrySpec('b.w', 1, (3,), RR)))
            >>> entry, indices = layout.locate('b.w[1]')
            >>> entry.name, entry.shape, indices
            ('b.w[1]', (), slice(2, 3, None))
        """
        name = self._get_internal_name(name)
        if name in self._by_name:
            entry = self._by_name[name]
            return entry, entry.indices
        found = self._by_name.get(name.partition('[')[0].removesuffix('.real').removesuffix('.imag'))
        if found is not None and name in found.component_names:
            start = found.start + found.component_names.index(name)
            return EntrySpec(name, start, (), RR if found.is_complex else found.domain), slice(
                start,
                start + 1,
            )
        error_msg = f'No variable or component named {name!r}. Available names are: {", ".join(self.names)}.'
        raise KeyError(error_msg)

    def read(self, vector: npt.NDArray[np.float64], name: str) -> Array | ComplexArray | float | complex:
        """Return the value of a variable from a flat vector or from rows of flat vectors.

        Args:
            vector:
                A flat vector, or a 2D array with one flat vector per row.

            name:
                The dotted name of the variable.

        Returns:
            value:
                A number for a scalar in a flat vector, or an array whose shape is the leading shape of `vector`
                followed by the shape of the variable, complex for a complex variable.
        """
        entry = self.get_entry(name)
        values = entry.decode(vector[..., entry.indices]).reshape(vector.shape[:-1] + entry.shape)
        if values.ndim:
            return values
        return complex(values) if entry.is_complex else float(values)

    def get_values(self, vector: npt.NDArray[np.float64]) -> dict[str, Array | ComplexArray | float | complex]:
        """Return the value of every variable of a flat vector.

        Args:
            vector:
                A flat vector.

        Returns:
            values:
                A mapping from dotted name to value.
        """
        return {name: self.read(vector, name) for name in self._by_name}

    def get_relaid(self, vector: npt.NDArray[np.float64], layout: 'LayoutSpec', initial: Array) -> Array:
        """Return a flat vector of another layout laid out in this one.

        Every variable keeps the rows and columns both layouts hold, and the rows only this layout holds, such as the
        free rows of a group that grew, keep their values in `initial`.

        Args:
            vector:
                A flat vector of the other layout.

            layout:
                The other layout.

            initial:
                A flat vector of this layout whose values fill what the other layout lacks.

        Returns:
            vector:
                The flat vector of this layout.
        """
        relaid = np.array(initial, dtype=float)
        known = layout._by_name
        for entry in self.entries:
            before = known.get(entry.name)
            if before is None:
                continue
            rows, columns = min(entry.rows, before.rows), min(entry.columns, before.columns)
            view = entry.get_view(relaid).copy()
            view[:rows, :columns] = before.get_view(vector)[:rows, :columns]
            relaid[entry.indices] = view.ravel()
        return relaid
