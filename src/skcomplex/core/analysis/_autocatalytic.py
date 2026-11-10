"""Find autocatalytic sets."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..specs import AnalysisTags, NetworkSnapshotSpec
from ..types import Array, DataKind, Indices
from .types import Source

TAGS = AnalysisTags(DataKind.NETWORK)


@dataclass(frozen=True)
class AutocatalyticSetsResult:
    """A result of `autocatalytic_sets`, the largest such set at every frame.

    Args:
        snapshots:
            The network snapshot of every analysed frame.

        reactions:
            The rows, in the hyperedges of the snapshot, of the reactions of the largest RAF of every frame.

        products:
            The node rows that the reactions of the largest RAF of every frame produce.

        sizes:
            The number of reactions of the largest RAF of every frame.

        fractions:
            The share of the reactions of every frame that the largest RAF holds, 0 for a frame without reactions.

        exists:
            Whether every frame holds a RAF.
    """

    snapshots: list[NetworkSnapshotSpec]
    reactions: list[Indices]
    products: list[Indices]
    sizes: npt.NDArray[np.int_]
    fractions: Array
    exists: npt.NDArray[np.bool_]


def autocatalytic_sets(
    source: Source,
    *,
    food: Sequence[int] | None = None,
    frames: Sequence[int] | None = None,
) -> AutocatalyticSetsResult:
    """Return the maximal autocatalytic set of a catalytic reaction network.

    The set is found at every frame. The method is that of Hordijk and Steel. The network is a network snapshot whose
    hyperedges are the reactions and whose directed edges are the catalysis. Every hyperedge produces its last member
    from its other members, the reactants, and an edge from a node `c` to a node `p` makes `c` a catalyst of every
    reaction that produces `p`, as goods catalyse the production of other goods in the TAP model of Hordijk and Kauffman
    (2022). A set of reactions is reflexively autocatalytic and food-generated, a RAF, when every reaction has a
    catalyst produced by the set or in the food set, and every reactant is reachable from the food set by reactions of
    the set. The algorithm of Hordijk and Steel (2004) finds the largest RAF, the union of every RAF: it removes the
    reactions with a reactant outside the closure of the food set, or without a catalyst in it, until no reaction is
    removed, and returns the reactions left, an empty set when there is no RAF.

    It appears in studies of the origin of life, where sets of molecules catalyse their own production, and in models of
    economies whose goods are made from each other.

    Args:
        source:
            The network snapshot, or the trajectory of a network whose hyperedges change.

        food:
            The node rows of the food set, or `None` for the nodes present that no reaction produces.

        frames:
            The frames of a trajectory to analyse, or `None` for every frame.

    Returns:
        result:
            The snapshots, and the reactions, products, size, and share of the largest RAF of every frame.

    Raises:
        AnalysisError:
            The source is neither a network snapshot nor a trajectory of a network whose hyperedges change.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import autocatalytic_sets
        >>> from skcomplex.core.specs import NetworkSnapshotSpec
        >>> nodes = np.arange(5)
        >>> reactions = np.array([[0, 1, 2], [2, 0, 3], [3, 1, 4]])
        >>> sources, targets = np.array([3, 4, 0]), np.array([2, 3, 4])
        >>> snapshot = NetworkSnapshotSpec(nodes, sources, targets, np.arange(3), reactions)
        >>> result = autocatalytic_sets(snapshot)
        >>> result.reactions[0].tolist(), result.products[0].tolist()
        ([0, 1, 2], [2, 3, 4])
        >>> uncatalysed = NetworkSnapshotSpec(nodes, sources[:2], targets[:2], np.arange(2), reactions)
        >>> autocatalytic_sets(uncatalysed).exists.tolist()
        [False]
    """
    if isinstance(source, NetworkSnapshotSpec):
        snapshots = [source]
    elif isinstance(source, Trajectory):
        chosen = range(len(source.times)) if frames is None else frames
        snapshots = [source.get_network(int(frame)) for frame in chosen]
    else:
        error_msg = f'Autocatalytic sets are read from a network snapshot or a trajectory, not {type(source).__name__}.'
        raise AnalysisError(error_msg)
    if any(snapshot.hyperedges is None for snapshot in snapshots):
        error_msg = (
            'Autocatalytic sets read the reactions as hyperedges that change. Give the network a variable group on '
            'its hyperedges.'
        )
        raise AnalysisError(error_msg)
    found = [_largest_raf(snapshot, food) for snapshot in snapshots]
    reactions = [rows for rows, _ in found]
    sizes = np.asarray([rows.size for rows in reactions], dtype=int)
    totals = np.asarray([len(_reactions_of(snapshot)) for snapshot in snapshots])
    fractions = np.divide(sizes, totals, out=np.zeros(len(totals)), where=totals > 0)
    return AutocatalyticSetsResult(
        snapshots,
        reactions,
        [products for _, products in found],
        sizes,
        fractions,
        sizes > 0,
    )


def _largest_raf(snapshot: NetworkSnapshotSpec, food: Sequence[int] | None) -> tuple[Indices, Indices]:
    """Return the reactions of the largest RAF of a snapshot and the nodes they make.

    Args:
        snapshot:
            The network snapshot, with the reactions as hyperedges.

        food:
            The node rows of the food set, or `None` for the nodes present that no reaction produces.

    Returns:
        reactions:
            The rows of the reactions of the largest RAF, in increasing order.

        products:
            The node rows the reactions produce, in the order of the reactions.
    """
    hyperedges = _reactions_of(snapshot)
    reactants = [[int(node) for node in row[row >= 0][:-1]] for row in hyperedges]
    products = [int(row[row >= 0][-1]) for row in hyperedges]
    catalysts: dict[int, set[int]] = {}
    for source, target in zip(snapshot.sources.tolist(), snapshot.targets.tolist(), strict=True):
        catalysts.setdefault(target, set()).add(source)
    fed = _food(snapshot, products, food)
    kept = list(range(len(products)))
    while True:
        closure = _closure(fed, [(reactants[index], products[index]) for index in kept])
        left = [
            index
            for index in kept
            if set(reactants[index]) <= closure and catalysts.get(products[index], set()) & closure
        ]
        if len(left) == len(kept):
            break
        kept = left
    chosen = np.asarray(kept, dtype=np.intp)
    return chosen, np.asarray([products[index] for index in kept], dtype=np.intp)


def _reactions_of(snapshot: NetworkSnapshotSpec) -> Indices:
    """Return the hyperedges of a snapshot, one row of member nodes per reaction.

    The rows are padded with -1.
    """
    return np.zeros((0, 2), dtype=np.intp) if snapshot.hyperedges is None else snapshot.hyperedges


def _food(snapshot: NetworkSnapshotSpec, products: list[int], food: Sequence[int] | None) -> set[int]:
    """Return the food set.

    It is the given node rows, or the nodes present that no reaction produces.
    """
    if food is not None:
        return {int(node) for node in food}
    return set(snapshot.nodes.tolist()) - set(products)


def _closure(food: set[int], reactions: list[tuple[list[int], int]]) -> set[int]:
    """Return the nodes reachable from the food set by reactions.

    A reaction counts when its reactants are all reachable.

    Args:
        food:
            The food set.

        reactions:
            The reactants and the product of every reaction.

    Returns:
        closure:
            The food set and every product reachable from it.
    """
    closure = set(food)
    changed = True
    while changed:
        changed = False
        for reactants, product in reactions:
            if product not in closure and all(node in closure for node in reactants):
                closure.add(product)
                changed = True
    return closure
