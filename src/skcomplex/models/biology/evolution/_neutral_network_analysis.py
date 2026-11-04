"""Measure the neutral networks, robustness, and evolvability of a map from sequences to phenotypes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Callable, Hashable, Iterator, Sequence

import networkx as nx
import numpy as np

from ....analysis import BaseAnalysis
from ....engine.runtime import Trajectory
from ....errors import AnalysisError
from ....types import DataKind

KINDS = (
    'network',
    'genotype-robustness',
    'genotype-evolvability',
    'phenotype-robustness',
    'phenotype-evolvability',
)


class NeutralNetworkAnalysis(BaseAnalysis):
    """Neutral networks of a map from sequences to phenotypes, with robustness and evolvability as Wagner defines them.

    The neighbours of a genotype are the sequences one point mutation away. Its robustness is the number of its
    neighbours of the same phenotype, and its evolvability the number of different other phenotypes among its
    neighbours. The robustness of a phenotype is the mean robustness of its genotypes, and its evolvability the number
    of different other phenotypes among the neighbours of all its genotypes. The definitions follow Wagner (2008), as
    Raman (2021, §13.3) states them, except that Raman counts the different phenotypes of a neighbourhood without
    saying whether the phenotype of the genotype itself counts, and this class leaves it out.

    Args:
        phenotype_of:
            The function from a sequence to a phenotype key.

        kind:
            `'network'`, `'genotype-robustness'`, `'genotype-evolvability'`, `'phenotype-robustness'`, or
            `'phenotype-evolvability'`.

        alphabet:
            The letters of the sequences.

        state:
            The dotted name of the lattice state of a trajectory that holds one sequence per row, or `None` for the
            first state.

    Attributes:
        genotypes_:
            The sequences of the source, in an array of their shape without the letters.

        values_:
            For a genotype kind, one value per sequence of the source. For a phenotype kind, the value of every
            phenotype of the source, by key.

        networks_:
            For the kind `'network'`, the graph per phenotype of the sequences of the source that one point mutation
            joins.

    Raises:
        AnalysisError:
            The kind is unknown, the mapping is not callable, or a sequence holds a letter outside the alphabet.

    Examples:
        >>> import itertools
        >>> from skcomplex.models.biology.evolution import NeutralNetworkAnalysis
        >>> genotypes = [''.join(letters) for letters in itertools.product('01', repeat=4)]
        >>> analysis = NeutralNetworkAnalysis(lambda sequence: sequence[0], 'genotype-robustness', alphabet='01')
        >>> set(analysis.check().prepare(genotypes).run().values_.tolist())
        {3.0}
    """

    def __init__(
        self,
        phenotype_of: Callable[[str], Hashable],
        kind: str = 'network',
        alphabet: Sequence[str] = 'ACGU',
        state: str | None = None,
    ) -> None:
        self.phenotype_of = phenotype_of
        self.kind = kind
        self.alphabet = alphabet
        self.state = state

    def _get_data_kind(self) -> DataKind:
        """Return the series kind, since the source holds one sequence per row."""
        return DataKind.SERIES

    def _check(self) -> None:
        """Raise an error for an unknown kind, or a map that is not callable."""
        self._check_choice('kind', self.kind, KINDS)
        if not callable(self.phenotype_of):
            error_msg = f'NeutralNetworkAnalysis reads a function from sequences, not {self.phenotype_of!r}.'
            raise AnalysisError(error_msg)

    def _read_sequences(self, source: object) -> np.ndarray:
        """Return the sequences of a trajectory, of an array of strings, or of an array of letters on its last axis."""
        if isinstance(source, Trajectory):
            name = self.state if self.state is not None else source.problem.states.names[0]
            source = source.lattice(name)
        array = np.asarray(source)
        if array.dtype.kind in 'USO' and len(array.flat[0]) > 1:
            return array.astype(object)
        return np.apply_along_axis(lambda row: ''.join(map(str, row)), -1, array).astype(object)

    def _prepare(self, source: object) -> None:
        """Read the sequences of the source, and raise an error for a letter outside the alphabet."""
        self.genotypes_ = self._read_sequences(source)
        foreign = {letter for sequence in self.genotypes_.flat for letter in sequence} - set(self.alphabet)
        if foreign:
            error_msg = f'The sequences hold the letters {sorted(foreign)}, which the alphabet {self.alphabet} lacks.'
            raise AnalysisError(error_msg)

    def _neighbours(self, sequence: str) -> Iterator[str]:
        """Yield the sequences one point mutation away from a sequence."""
        for site, letter in itertools.product(range(len(sequence)), self.alphabet):
            if sequence[site] != letter:
                yield f'{sequence[:site]}{letter}{sequence[site + 1 :]}'

    def _robustness(self, sequence: str) -> float:
        """Return the number of neighbours of a sequence of the same phenotype."""
        own = self.phenotype_of(sequence)
        return float(sum(self.phenotype_of(item) == own for item in self._neighbours(sequence)))

    def _other_phenotypes(self, sequence: str) -> set[Hashable]:
        """Return the phenotypes among the neighbours of a sequence other than its own."""
        return {self.phenotype_of(item) for item in self._neighbours(sequence)} - {self.phenotype_of(sequence)}

    def _by_phenotype(self) -> dict[Hashable, list[str]]:
        """Return the different sequences of the source by their phenotype."""
        groups: dict[Hashable, list[str]] = {}
        for sequence in dict.fromkeys(self.genotypes_.flat):
            groups.setdefault(self.phenotype_of(sequence), []).append(sequence)
        return groups

    def _network(self, genotypes: list[str]) -> nx.Graph:
        """Return the graph of some sequences that one point mutation joins."""
        graph = nx.Graph()
        graph.add_nodes_from(genotypes)
        present = set(genotypes)
        graph.add_edges_from(
            (item, other) for item in genotypes for other in self._neighbours(item) if other in present
        )
        return graph

    def _run(self, inputs: object = None) -> None:
        """Set the values of the kind."""
        if self.kind == 'network':
            self.networks_ = {key: self._network(items) for key, items in self._by_phenotype().items()}
            return
        if self.kind == 'genotype-robustness':
            self.values_ = np.vectorize(self._robustness, otypes=[float])(self.genotypes_)
        elif self.kind == 'genotype-evolvability':
            self.values_ = np.vectorize(lambda item: float(len(self._other_phenotypes(item))), otypes=[float])(
                self.genotypes_,
            )
        elif self.kind == 'phenotype-robustness':
            self.values_ = {
                key: float(np.mean([self._robustness(item) for item in items]))
                for key, items in self._by_phenotype().items()
            }
        else:
            self.values_ = {
                key: float(len(set().union(*(self._other_phenotypes(item) for item in items))))
                for key, items in self._by_phenotype().items()
            }
