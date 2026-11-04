"""Fold RNA sequences into secondary structures through ViennaRNA."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import lru_cache

import numpy as np
import numpy.typing as npt

from ..._extras import import_extra

MODEL = 'rna_neutral_networks'
EXTRA = 'rna'
FOLD_CACHE = 2**18


@lru_cache(maxsize=FOLD_CACHE)
def fold_sequence(sequence: str) -> tuple[str, float]:
    """Return the minimum free energy secondary structure of an RNA sequence at 37 °C and its energy, memoised.

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
        >>> from skcomplex.models.biology.evolution import fold_sequence
        >>> structure, energy = fold_sequence('GGGGAAACCCC')
        >>> structure, energy < 0
        ('((((...))))', True)
    """
    structure, energy = import_extra('RNA', MODEL, EXTRA).fold_compound(sequence).mfe()
    return str(structure), float(energy)


def read_base_pair_distance(first: str, second: str) -> int:
    """Return the number of base pairs that one of two secondary structures of one length holds and the other does not.

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
        >>> from skcomplex.models.biology.evolution import read_base_pair_distance
        >>> read_base_pair_distance('((((...))))', '.(((...))).')
        1
    """
    return int(import_extra('RNA', MODEL, EXTRA).bp_distance(first, second))


def read_plane_layout(structure: str) -> npt.NDArray[np.float64]:
    """Return the positions in a plane of the bases of a secondary structure, in the simple layout of ViennaRNA.

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
        >>> from skcomplex.models.biology.evolution import read_plane_layout
        >>> read_plane_layout('((((...))))').shape
        (11, 2)
    """
    layout = import_extra('RNA', MODEL, EXTRA).simple_xy_coordinates(structure)
    return np.array([[point.X, point.Y] for point in layout[: len(structure)]], dtype=float)
