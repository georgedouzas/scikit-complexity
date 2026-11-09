"""Evolution system: walks of RNA sequences through point mutations.

The mutations keep or approach a secondary structure.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import lru_cache, partial
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import SamplingInteraction
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.systems import BaseSystem, System
from ....core.errors import ValidationError
from ....core.specs import RowScoreSpec, StateSpec, VisualSpec
from ..._extras import import_extra

SYSTEM = 'rna_neutral_networks'

EXTRA = 'rna'

FOLD_CACHE = 2**18

BASES = 'ACGU'

LAST_BASE = len(BASES) - 1

START = 'GGGGAAACCCCAUCGAUAGCGGGAAACCGC'

TARGET = '((((...)))).......((((....))))'

REGIMES = ('neutral', 'adaptive')


class NucleotideElement(Element):
    """Site of an RNA sequence that holds one base.

    Args:
        label:
            The label of the element.

        base:
            The base of the site.

    Examples:
        >>> from skcomplex.systems.biology.evolution import NucleotideElement
        >>> NucleotideElement('site').base
        'A'
    """

    def __init__(
        self,
        label: str,
        base: Annotated[str, StateSpec(domain=FiniteEnumeratedSet(list(BASES)))] = 'A',
    ) -> None:
        super().__init__(label)
        self.base = base


@lru_cache(maxsize=FOLD_CACHE)
def fold_sequence(sequence: str) -> tuple[str, float]:
    """Return the minimum free energy secondary structure of an RNA sequence.

    It also returns its energy, at 37 °C, memoised.

    Args:
        sequence:
            The sequence over the letters `ACGU`.

    Returns:
        fold:
            The structure in dot-bracket notation and its free energy in kcal/mol.

    Raises:
        MissingExtraError:
            The extra `rna` is not installed.

    Examples:
        >>> from skcomplex.systems.biology.evolution import fold_sequence
        >>> structure, energy = fold_sequence('GGGGAAACCCC')
        >>> structure, energy < 0
        ('((((...))))', True)
    """
    structure, energy = import_extra('RNA', SYSTEM, EXTRA).fold_compound(sequence).mfe()
    return str(structure), float(energy)


def read_base_pair_distance(first: str, second: str) -> int:
    """Return the number of base pairs that only one of two secondary structures holds.

    The structures have one length.

    Args:
        first:
            The first structure in dot-bracket notation.

        second:
            The second structure in dot-bracket notation.

    Returns:
        distance:
            The base-pair distance.

    Raises:
        MissingExtraError:
            The extra `rna` is not installed.

    Examples:
        >>> from skcomplex.systems.biology.evolution import read_base_pair_distance
        >>> read_base_pair_distance('((((...))))', '.(((...))).')
        1
    """
    return int(import_extra('RNA', SYSTEM, EXTRA).bp_distance(first, second))


def read_plane_layout(structure: str) -> npt.NDArray[np.float64]:
    """Return the positions in a plane of the bases of a secondary structure.

    The positions follow the simple layout of ViennaRNA.

    Args:
        structure:
            The structure in dot-bracket notation.

    Returns:
        positions:
            One row of two coordinates per base.

    Raises:
        MissingExtraError:
            The extra `rna` is not installed.

    Examples:
        >>> from skcomplex.systems.biology.evolution import read_plane_layout
        >>> read_plane_layout('((((...))))').shape
        (11, 2)
    """
    layout = import_extra('RNA', SYSTEM, EXTRA).simple_xy_coordinates(structure)
    return np.array([[point.X, point.Y] for point in layout[: len(structure)]], dtype=float)


def read_sequence(row: np.ndarray) -> str:
    """Return the sequence of a row of indices of bases.

    Args:
        row:
            The indices of the bases.

    Returns:
        sequence:
            The sequence over the letters `ACGU`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.biology.evolution import read_sequence
        >>> read_sequence(np.array([0.0, 3.0, 2.0]))
        'AUG'
    """
    return ''.join(BASES[int(value)] for value in row)


def mutate_point(row: np.ndarray, random: np.random.Generator) -> np.ndarray:
    """Return a row of indices of bases with one random site mutated.

    The base at the site is replaced by one of the other three.

    Args:
        row:
            The indices of the bases, which the function changes.

        random:
            The random generator.

    Returns:
        row:
            The row after the mutation.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.systems.biology.evolution import mutate_point
        >>> row = mutate_point(np.zeros(4), np.random.default_rng(0))
        >>> int((row != 0).sum())
        1
    """
    site = random.integers(row.size)
    row[site] = (row[site] + random.integers(1, len(BASES))) % len(BASES)
    return row


class RNANeutralNetworksSystem(BaseSystem):
    """Walkers of RNA sequences at temperature zero.

    They take one proposed point mutation per unit of time.

    In the `'neutral'` regime a mutation is accepted when the minimum free energy fold of the sequence stays on the
    target. In the `'adaptive'` regime a mutation is accepted when the base-pair distance between the fold and the
    target does not grow. The neighbours of the visited sequences give the robustness and evolvability of Raman (2021
    §13.3) through `neutral_network`. The folds come from ViennaRNA at 37 °C with its default energy parameters. Every
    walker records its score as `walk.score`. The system has the lattice `walkers` of the state `base`.

    Args:
        walkers:
            The number of walkers, the rows of the lattice.

        length:
            The length of the sequences, the columns of the lattice.

        start:
            The sequence every walker starts on, or `None` for random sequences.

        target:
            The target structure in dot-bracket notation, of the length of the sequences.

        regime:
            `'neutral'` or `'adaptive'`.

        seed:
            The seed of the random starting sequences.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        MissingExtraError:
            The extra `rna` is not installed.

        ValidationError:
            The regime is unknown, or the start or the target does not have the length of the sequences.

    Examples:
        >>> from skcomplex.systems.biology.evolution import RNANeutralNetworksSystem
        >>> RNANeutralNetworksSystem(walkers=2).prepare().system_.elements[0].space.shape
        (2, 30)
    """

    field = 'biology/evolution'
    topics = ('rna', 'genotype networks', 'neutral networks', 'robustness', 'evolvability', 'monte carlo')
    references = ('Raman 2021', 'Wagner 2008', 'Lorenz et al. 2011')
    until = 300

    def __init__(
        self,
        walkers: int = 8,
        length: int = 30,
        *,
        start: str | None = START,
        target: str = TARGET,
        regime: str = 'neutral',
        seed: int = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.walkers = walkers
        self.length = length
        self.start = start
        self.target = target
        self.regime = regime
        self.seed = seed

    @staticmethod
    def _score_row(target: str, regime: str, row: np.ndarray) -> float:
        """Return the score of a sequence.

        The score is off the target in the neutral regime and the distance in the adaptive one.
        """
        fold = fold_sequence(read_sequence(row))[0]
        return float(fold != target) if regime == 'neutral' else float(read_base_pair_distance(fold, target))

    def _build_system(self) -> System:
        """Return the parts of walkers of RNA sequences."""
        import_extra('RNA', 'rna_neutral_networks', 'rna')
        if self.regime not in REGIMES:
            error_msg = f'Unknown regime {self.regime!r}. Use one of {", ".join(REGIMES)}.'
            raise ValidationError(error_msg)
        if len(self.target) != self.length or (self.start is not None and len(self.start) != self.length):
            error_msg = (
                f'The start {self.start!r} and the target {self.target!r} must have the length {self.length} of the '
                'sequences.'
            )
            raise ValidationError(error_msg)
        if self.start is None:
            bases = np.random.default_rng(self.seed).integers(0, len(BASES), size=(self.walkers, self.length))
        else:
            bases = np.tile([BASES.index(letter) for letter in self.start], (self.walkers, 1))
        return System(
            self.label,
            elements=[
                LatticeElement(
                    'walkers',
                    element=NucleotideElement,
                    space=LatticeSpace((self.walkers, self.length), boundary='fixed'),
                    values={'base': bases},
                    visual=VisualSpec('image', {'low': 0.0, 'high': float(LAST_BASE)}),
                ),
            ],
            interactions=[
                SamplingInteraction(
                    'walk',
                    target='walkers',
                    state='base',
                    objective=RowScoreSpec(partial(self._score_row, self.target, self.regime)),
                    move=mutate_point,
                    intensity=float('inf'),
                ),
            ],
            time=DiscreteTime(),
        )
