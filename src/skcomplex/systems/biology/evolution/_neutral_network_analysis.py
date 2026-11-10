"""Measure the neutral networks of a map from sequences to phenotypes.

The measures also cover the robustness and the evolvability of the map.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np

from ....core.engine.runtime import Trajectory
from ....core.errors import AnalysisError

KINDS = (
    'network',
    'genotype-robustness',
    'genotype-evolvability',
    'phenotype-robustness',
    'phenotype-evolvability',
)


@dataclass(frozen=True)
class NeutralNetworkResult:
    """Neutral networks, robustness, or evolvability of the sequences of a source.

    Args:
        genotypes:
            The sequences of the source, in an array of their shape without the letters.

        values:
            For a genotype kind, one value per sequence of the source. For a phenotype kind, the value of every
            phenotype of the source, by key. `None` for `'network'`.

        networks:
            For the kind `'network'`, the graph per phenotype of the sequences of the source that one point mutation
            joins, or `None` for the other kinds.
    """

    genotypes: np.ndarray
    values: np.ndarray | dict[Hashable, float] | None = None
    networks: dict[Hashable, nx.Graph] | None = None


def neutral_network(
    source: object,
    *,
    phenotype_of: Callable[[str], Hashable],
    kind: str = 'network',
    alphabet: Sequence[str] = 'ACGU',
    state: str | None = None,
) -> NeutralNetworkResult:
    """Return the neutral networks of a map from sequences to phenotypes.

    It returns their robustness and evolvability instead on request.

    The neighbours of a genotype are the sequences one point mutation away. Its robustness is the number of its
    neighbours of the same phenotype, and its evolvability the number of different phenotypes among its neighbours. The
    robustness of a phenotype is the mean robustness of its genotypes, and its evolvability the number of different
    phenotypes among the neighbours of all its genotypes. The definitions follow Wagner (2008), as Raman (2021, §13.3)
    states them, so the phenotype of a genotype itself counts when a neighbour has it.

    Args:
        source:
            A trajectory whose lattice state holds one sequence per row, an array of strings, or an array of letters
            on its last axis.

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

    Returns:
        result:
            The sequences, and the networks or the values of the kind.

    Raises:
        AnalysisError:
            The kind is unknown, the mapping is not callable, or a sequence holds a letter outside the alphabet.

    Examples:
        >>> import itertools
        >>> from skcomplex.systems.biology.evolution import neutral_network
        >>> genotypes = [''.join(letters) for letters in itertools.product('01', repeat=4)]
        >>> result = neutral_network(
        ...     genotypes, phenotype_of=lambda sequence: sequence[0], kind='genotype-robustness', alphabet='01'
        ... )
        >>> set(result.values.tolist())
        {3.0}
    """
    if kind not in KINDS:
        error_msg = f'neutral_network has kind={kind!r}. Use one of {", ".join(KINDS)}.'
        raise AnalysisError(error_msg)
    if not callable(phenotype_of):
        error_msg = f'neutral_network reads a function from sequences, not {phenotype_of!r}.'
        raise AnalysisError(error_msg)
    genotypes = _read_sequences(source, state)
    foreign = {letter for sequence in genotypes.flat for letter in sequence} - set(alphabet)
    if foreign:
        error_msg = f'The sequences hold the letters {sorted(foreign)}, which the alphabet {alphabet} lacks.'
        raise AnalysisError(error_msg)
    if kind == 'network':
        networks = {key: _network(items, alphabet) for key, items in _by_phenotype(genotypes, phenotype_of).items()}
        return NeutralNetworkResult(genotypes, networks=networks)
    values: np.ndarray | dict[Hashable, float]
    if kind == 'genotype-robustness':
        values = np.vectorize(lambda item: _robustness(item, phenotype_of, alphabet), otypes=[float])(genotypes)
    elif kind == 'genotype-evolvability':
        values = np.vectorize(
            lambda item: float(len(_neighbouring_phenotypes(item, phenotype_of, alphabet))),
            otypes=[float],
        )(genotypes)
    elif kind == 'phenotype-robustness':
        values = {
            key: float(np.mean([_robustness(item, phenotype_of, alphabet) for item in items]))
            for key, items in _by_phenotype(genotypes, phenotype_of).items()
        }
    else:
        values = {
            key: float(
                len(set().union(*(_neighbouring_phenotypes(item, phenotype_of, alphabet) for item in items))),
            )
            for key, items in _by_phenotype(genotypes, phenotype_of).items()
        }
    return NeutralNetworkResult(genotypes, values=values)


def _read_sequences(source: object, state: str | None) -> np.ndarray:
    """Return the sequences of a trajectory or an array.

    The array holds strings, or letters on its last axis.
    """
    if isinstance(source, Trajectory):
        name = state if state is not None else source.problem.states.names[0]
        source = source.get_lattice(name)
    array = np.asarray(source)
    if array.dtype.kind in 'USO' and len(array.flat[0]) > 1:
        return array.astype(object)
    return np.apply_along_axis(lambda row: ''.join(map(str, row)), -1, array).astype(object)


def _neighbours(sequence: str, alphabet: Sequence[str]) -> Iterator[str]:
    """Yield the sequences one point mutation away from a sequence."""
    for site, letter in itertools.product(range(len(sequence)), alphabet):
        if sequence[site] != letter:
            yield f'{sequence[:site]}{letter}{sequence[site + 1 :]}'


def _robustness(sequence: str, phenotype_of: Callable[[str], Hashable], alphabet: Sequence[str]) -> float:
    """Return the number of neighbours of a sequence of the same phenotype."""
    own = phenotype_of(sequence)
    return float(sum(phenotype_of(item) == own for item in _neighbours(sequence, alphabet)))


def _neighbouring_phenotypes(
    sequence: str,
    phenotype_of: Callable[[str], Hashable],
    alphabet: Sequence[str],
) -> set[Hashable]:
    """Return the different phenotypes among the neighbours of a sequence."""
    return {phenotype_of(item) for item in _neighbours(sequence, alphabet)}


def _by_phenotype(genotypes: np.ndarray, phenotype_of: Callable[[str], Hashable]) -> dict[Hashable, list[str]]:
    """Return the different sequences of the source by their phenotype."""
    groups: dict[Hashable, list[str]] = {}
    for sequence in dict.fromkeys(genotypes.flat):
        groups.setdefault(phenotype_of(sequence), []).append(sequence)
    return groups


def _network(genotypes: list[str], alphabet: Sequence[str]) -> nx.Graph:
    """Return the graph of some sequences that one point mutation joins."""
    graph = nx.Graph()
    graph.add_nodes_from(genotypes)
    present = set(genotypes)
    graph.add_edges_from(
        (item, other) for item in genotypes for other in _neighbours(item, alphabet) if other in present
    )
    return graph
