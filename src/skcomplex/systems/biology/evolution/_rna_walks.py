"""Evolution system: walks of RNA sequences through point mutations that keep or approach a secondary structure."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from typing import Annotated

import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import SamplingInteraction
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import RowScoreSpec, StateSpec, VisualSpec
from ..._catalog import register_system
from ..._extras import import_extra
from ._rna_folds import fold_sequence, read_base_pair_distance

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
    """Return a row of indices of bases with the base at one random site replaced by one of the other three.

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


def _score_row(target: str, regime: str, row: np.ndarray) -> float:
    """Return the score of a sequence: off the target in the neutral regime, the distance in the adaptive one."""
    fold = fold_sequence(read_sequence(row))[0]
    return float(fold != target) if regime == 'neutral' else float(read_base_pair_distance(fold, target))


@register_system(
    field='biology/evolution',
    tags=('rna', 'genotype networks', 'neutral networks', 'robustness', 'evolvability', 'monte carlo'),
    references=('Raman 2021', 'Wagner 2008', 'Lorenz et al. 2011'),
    defaults={'until': 300},
)
def rna_neutral_networks(
    walkers: int = 8,
    length: int = 30,
    *,
    start: str | None = START,
    target: str = TARGET,
    regime: str = 'neutral',
    seed: int = 0,
) -> System:
    """Return walkers of RNA sequences that take one proposed point mutation per unit of time at temperature zero.

    In the `'neutral'` regime a mutation is accepted when the minimum free energy fold of the sequence stays on the
    target. In the `'adaptive'` regime a mutation is accepted when the base-pair distance between the fold and the
    target does not grow. The neighbours of the visited sequences give the robustness and evolvability of
    Raman (2021 §13.3) through `NeutralNetworkAnalysis`. The folds come from ViennaRNA at 37 °C with its default energy
    parameters. Every walker records its score as `walk.score`.

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

    Returns:
        system:
            The system, with the lattice `walkers` of the state `base`.

    Raises:
        MissingExtraError:
            The extra `rna` is not installed.

        ValidationError:
            The regime is unknown, or the start or the target does not have the length of the sequences.

    Examples:
        >>> from skcomplex.systems.biology.evolution import rna_neutral_networks
        >>> rna_neutral_networks(walkers=2).elements[0].space.shape
        (2, 30)
    """
    import_extra('RNA', 'rna_neutral_networks', 'rna')
    if regime not in REGIMES:
        error_msg = f'Unknown regime {regime!r}. Use one of {", ".join(REGIMES)}.'
        raise ValidationError(error_msg)
    if len(target) != length or (start is not None and len(start) != length):
        error_msg = f'The start {start!r} and the target {target!r} must have the length {length} of the sequences.'
        raise ValidationError(error_msg)
    if start is None:
        bases = np.random.default_rng(seed).integers(0, len(BASES), size=(walkers, length))
    else:
        bases = np.tile([BASES.index(letter) for letter in start], (walkers, 1))
    return System(
        elements=[
            LatticeElement(
                'walkers',
                element=NucleotideElement,
                space=LatticeSpace((walkers, length), boundary='fixed'),
                values={'base': bases},
                visual=VisualSpec('image', {'low': 0.0, 'high': float(LAST_BASE)}),
            ),
        ],
        interactions=[
            SamplingInteraction(
                'walk',
                target='walkers',
                state='base',
                objective=RowScoreSpec(partial(_score_row, target, regime)),
                move=mutate_point,
                intensity=float('inf'),
            ),
        ],
        time=DiscreteTime(),
    )
