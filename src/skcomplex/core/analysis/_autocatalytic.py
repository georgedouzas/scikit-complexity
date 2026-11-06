"""Define the analysis of autocatalytic sets, the maximal reflexively autocatalytic and food-generated subset."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ..engine.runtime import Trajectory
from ..errors import AnalysisError
from ..specs import NetworkSnapshotSpec
from ..types import DataKind, Indices
from ._base import BaseAnalysis
from .types import Source


class AutocatalyticSetAnalysis(BaseAnalysis):
    """Maximal autocatalytic set of a catalytic reaction network at every frame, by the algorithm of Hordijk and Steel.

    The network is a network snapshot whose hyperedges are the reactions and whose directed edges are the catalysis.
    Every hyperedge produces its last member from its other members, the reactants, and an edge from a node `c` to a
    node `p` makes `c` a catalyst of every reaction that produces `p`, as goods catalyse the production of other goods
    in the TAP model of Hordijk and Kauffman (2022). A set of reactions is reflexively autocatalytic and food-generated,
    a RAF, when every reaction has a catalyst produced by the set or in the food set, and every reactant is reachable
    from the food set by reactions of the set. The algorithm of Hordijk and Steel (2004) finds the largest RAF, the
    union of every RAF: it removes the reactions with a reactant outside the closure of the food set, or without a
    catalyst in it, until no reaction is removed, and returns the reactions left, an empty set when there is no RAF.

    Args:
        food:
            The node rows of the food set, or `None` for the nodes present that no reaction produces.

        frames:
            The frames of a trajectory to analyse, or `None` for every frame.

    Attributes:
        snapshots_:
            The network snapshot of every analysed frame.

        reactions_:
            The rows, in the hyperedges of the snapshot, of the reactions of the largest RAF of every frame.

        products_:
            The node rows that the reactions of the largest RAF of every frame produce.

        sizes_:
            The number of reactions of the largest RAF of every frame.

        fractions_:
            The share of the reactions of every frame that the largest RAF holds, 0 for a frame without reactions.

        exists_:
            Whether every frame holds a RAF.

    Raises:
        AnalysisError:
            The source is neither a network snapshot nor a trajectory of a network whose hyperedges change, raised by
            `prepare`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.analysis import AutocatalyticSetAnalysis
        >>> from skcomplex.core.specs import NetworkSnapshotSpec
        >>> nodes = np.arange(5)
        >>> reactions = np.array([[0, 1, 2], [2, 0, 3], [3, 1, 4]])
        >>> sources, targets = np.array([3, 4, 0]), np.array([2, 3, 4])
        >>> snapshot = NetworkSnapshotSpec(nodes, sources, targets, np.arange(3), reactions)
        >>> analysis = AutocatalyticSetAnalysis().prepare(snapshot).run()
        >>> analysis.reactions_[0].tolist(), analysis.products_[0].tolist()
        ([0, 1, 2], [2, 3, 4])
        >>> uncatalysed = NetworkSnapshotSpec(nodes, sources[:2], targets[:2], np.arange(2), reactions)
        >>> AutocatalyticSetAnalysis().prepare(uncatalysed).run().exists_.tolist()
        [False]
    """

    def __init__(self, food: Sequence[int] | None = None, frames: Sequence[int] | None = None) -> None:
        self.food = food
        self.frames = frames

    def _get_data_kind(self) -> DataKind:
        """Return the network kind."""
        return DataKind.NETWORK

    def _prepare(self, source: Source) -> None:
        """Read the network snapshot of every analysed frame."""
        if isinstance(source, NetworkSnapshotSpec):
            snapshots = [source]
        elif isinstance(source, Trajectory):
            frames = range(len(source.times)) if self.frames is None else self.frames
            snapshots = [source.network(int(frame)) for frame in frames]
        else:
            error_msg = (
                f'Autocatalytic sets are read from a network snapshot or a trajectory, not {type(source).__name__}.'
            )
            raise AnalysisError(error_msg)
        if any(snapshot.hyperedges is None for snapshot in snapshots):
            error_msg = (
                'Autocatalytic sets read the reactions as hyperedges that change. Give the network a variable group on '
                'its hyperedges.'
            )
            raise AnalysisError(error_msg)
        self.snapshots_ = snapshots

    def _run(self, inputs: object = None) -> None:
        """Find the largest RAF of every frame."""
        found = [self._largest_raf(snapshot) for snapshot in self.snapshots_]
        self.reactions_ = [reactions for reactions, _ in found]
        self.products_ = [products for _, products in found]
        self.sizes_ = np.asarray([reactions.size for reactions in self.reactions_], dtype=int)
        totals = np.asarray([len(self._reactions_of(snapshot)) for snapshot in self.snapshots_])
        self.fractions_ = np.divide(self.sizes_, totals, out=np.zeros(len(totals)), where=totals > 0)
        self.exists_ = self.sizes_ > 0

    def _largest_raf(self, snapshot: NetworkSnapshotSpec) -> tuple[Indices, Indices]:
        """Return the reactions of the largest RAF of a snapshot and the nodes they produce.

        Args:
            snapshot:
                The network snapshot, with the reactions as hyperedges.

        Returns:
            reactions:
                The rows of the reactions of the largest RAF, in increasing order.

            products:
                The node rows the reactions produce, in the order of the reactions.
        """
        hyperedges = self._reactions_of(snapshot)
        reactants = [[int(node) for node in row[row >= 0][:-1]] for row in hyperedges]
        products = [int(row[row >= 0][-1]) for row in hyperedges]
        catalysts: dict[int, set[int]] = {}
        for source, target in zip(snapshot.sources.tolist(), snapshot.targets.tolist(), strict=True):
            catalysts.setdefault(target, set()).add(source)
        food = self._food(snapshot, products)
        kept = list(range(len(products)))
        while True:
            closure = self._closure(food, [(reactants[index], products[index]) for index in kept])
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

    @staticmethod
    def _reactions_of(snapshot: NetworkSnapshotSpec) -> Indices:
        """Return the hyperedges of a snapshot, one row of member nodes per reaction, padded with -1."""
        return np.zeros((0, 2), dtype=np.intp) if snapshot.hyperedges is None else snapshot.hyperedges

    def _food(self, snapshot: NetworkSnapshotSpec, products: list[int]) -> set[int]:
        """Return the food set: the given node rows, or the nodes present that no reaction produces."""
        if self.food is not None:
            return {int(node) for node in self.food}
        return set(snapshot.nodes.tolist()) - set(products)

    @staticmethod
    def _closure(food: set[int], reactions: list[tuple[list[int], int]]) -> set[int]:
        """Return the nodes reachable from the food set by reactions whose reactants are all reachable.

        Args:
            food:
                The food set.

            reactions:
                The reactants and the product of every reaction.

        Returns:
            closure:
                The food set and every product reachable from it.

        Examples:
            >>> from skcomplex.core.analysis import AutocatalyticSetAnalysis
            >>> sorted(AutocatalyticSetAnalysis._closure({0, 1}, [([0, 1], 2), ([2, 5], 3), ([2], 4)]))
            [0, 1, 2, 4]
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
