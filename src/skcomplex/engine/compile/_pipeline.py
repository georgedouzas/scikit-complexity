"""Order the passes of a compilation by the fields they read and write, and run them over the compilation state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import heapq
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from typing import ClassVar

from ...core.errors import PipelineError
from ..utils import BaseEstimator
from ._compilation import Compilation
from ._issues import Issue

type Check = Callable[[Compilation], list[Issue]]


def _roles(passes: Sequence['BasePass']) -> dict[str, tuple[list[int], list[int], list[int]]]:
    """Return the producers, refiners, and readers of every field by the positions of their passes.

    A producer writes a field without reading it, a refiner reads and writes it, and a reader only reads it.
    """
    roles: dict[str, tuple[list[int], list[int], list[int]]] = defaultdict(lambda: ([], [], []))
    for index, item in enumerate(passes):
        for name in item.writes:
            roles[name][1 if name in item.reads else 0].append(index)
        for name in item.reads:
            if name not in item.writes:
                roles[name][2].append(index)
    return roles


def _check_inputs(
    passes: Sequence['BasePass'],
    roles: dict[str, tuple[list[int], list[int], list[int]]],
    present: Collection[str],
) -> None:
    """Raise an error for a field read but neither present nor produced, or produced by two passes."""
    for name, (producers, refiners, readers) in roles.items():
        if len(producers) > 1:
            names = ' and '.join(type(passes[index]).__name__ for index in producers)
            error_msg = (
                f'The passes {names} both produce the field {name!r}. Let one produce it and the others refine it.'
            )
            raise PipelineError(error_msg)
        if not producers and name not in present and (refiners or readers):
            index = (refiners or readers)[0]
            error_msg = (
                f'The pass {type(passes[index]).__name__} reads the field {name!r}, which no pass produces and the '
                'compilation does not hold. Add the pass that produces it.'
            )
            raise PipelineError(error_msg)


def _edges(roles: dict[str, tuple[list[int], list[int], list[int]]]) -> dict[int, set[int]]:
    """Return the passes that must run after every pass: refiners after producers, readers after both."""
    edges: dict[int, set[int]] = defaultdict(set)
    for producers, refiners, readers in roles.values():
        for first in producers:
            edges[first] |= {*refiners, *readers}
        for first in refiners:
            edges[first] |= set(readers)
    return edges


def _needed(passes: Sequence['BasePass'], field: str) -> set['BasePass']:
    """Return the passes that write a field and, in turn, the passes that write the fields they read."""
    writers: dict[str, list[BasePass]] = defaultdict(list)
    for item in passes:
        for name in item.writes:
            writers[name].append(item)
    needed: set[BasePass] = set()
    pending = [field]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for item in writers[name]:
            needed.add(item)
            pending += item.reads
    return needed


def order_passes(passes: Sequence['BasePass'], present: Collection[str] = ()) -> list['BasePass']:
    """Return passes in an order where every field is produced, then refined, then read.

    A pass that writes a field without reading it produces the field, and runs before the passes that refine it, which
    read and write it, and those run before the passes that only read it. Passes that the fields leave unordered run in
    the order they are given, so refiners of one field run in the order they are given.

    Args:
        passes:
            The passes, in the order they are registered.

        present:
            The fields the compilation already holds, which no pass needs to produce.

    Returns:
        passes:
            The passes in the order they run.

    Raises:
        PipelineError:
            A pass reads a field that no pass produces and the compilation does not hold, two passes produce one
            field, or the passes read each other's fields in a cycle.

    Examples:
        >>> from skcomplex.engine.compile import BasePass, order_passes
        >>> class LowerPass(BasePass):
        ...     reads, writes = ('layout',), ('lowered',)
        ...     def _run(self, compilation):
        ...         pass
        >>> class DeclarePass(BasePass):
        ...     reads, writes = ('nodes',), ('layout',)
        ...     def _run(self, compilation):
        ...         pass
        >>> [type(item).__name__ for item in order_passes([LowerPass(), DeclarePass()], present=['nodes'])]
        ['DeclarePass', 'LowerPass']
    """
    roles = _roles(passes)
    _check_inputs(passes, roles, present)
    edges = _edges(roles)
    waiting = [0] * len(passes)
    for targets in edges.values():
        for target in targets:
            waiting[target] += 1
    ready = [index for index, count in enumerate(waiting) if not count]
    heapq.heapify(ready)
    ordered: list[int] = []
    while ready:
        index = heapq.heappop(ready)
        ordered.append(index)
        for target in edges[index]:
            waiting[target] -= 1
            if not waiting[target]:
                heapq.heappush(ready, target)
    if len(ordered) < len(passes):
        names = ', '.join(type(item).__name__ for index, item in enumerate(passes) if index not in set(ordered))
        error_msg = f'The passes {names} read each other\'s fields in a cycle. Split one of them.'
        raise PipelineError(error_msg)
    return [passes[index] for index in ordered]


def run_passes(compilation: Compilation, passes: Sequence['BasePass'], until: str = 'problem') -> Compilation:
    """Run the passes a field needs that the compilation has not run yet, in the order of their fields.

    Args:
        compilation:
            The compilation state, which the passes change in place.

        passes:
            The passes, in the order they are registered.

        until:
            The field to produce. The passes that neither write it nor write a field that those passes read do not run.

    Returns:
        compilation:
            The same compilation state, holding the field.

    Raises:
        PipelineError:
            The passes cannot be ordered.
    """
    needed = _needed(passes, until)
    for item in order_passes(passes, compilation.present()):
        if item in needed and item not in compilation.done:
            item.run(compilation)
            compilation.done.add(item)
    return compilation


class BasePass(BaseEstimator):
    """Step of a compilation that reads fields of the compilation state and writes others.

    A pass declares the fields it reads and writes as the class variables `reads` and `writes`, which order the passes
    of a compilation. A field it reads is a field of `Compilation` or a fact another pass adds. A pass fits
    nothing, and `run(compilation)` calls its hook `_run`, which reads the fields the pass reads and writes the fields
    it writes.
    """

    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ()

    def _run(self, compilation: Compilation) -> None:
        """Read the fields of the compilation the pass reads and write the fields it writes.

        Args:
            compilation:
                The compilation state.

        Raises:
            NotImplementedError:
                The subclass does not implement the hook.
        """
        raise NotImplementedError
