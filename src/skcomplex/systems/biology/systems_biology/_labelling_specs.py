"""Define the specs of isotope labelling: atom maps, tracers, measured fragments, and their measurements."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ....core.errors import ValidationError
from ....core.specs import BaseSpec

SUM_TOLERANCE = 1e-9


@dataclass(frozen=True)
class AtomMapSpec(BaseSpec):
    """Atom transitions of a reaction, a string of atom letters for every occurrence of a metabolite.

    Args:
        reaction:
            The name of the reaction.

        reactants:
            Every occurrence of a reactant with its letters, as pairs of a metabolite and a string.

        products:
            Every occurrence of a product with its letters, as pairs of a metabolite and a string.

    Raises:
        ValidationError:
            A product names a letter that no reactant carries, or a reactant repeats a letter, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import AtomMapSpec
        >>> atoms = AtomMapSpec('v1', [('AcCoA', 'ab'), ('OAC', 'cdef')], [('Cit', 'fedbac')])
        >>> atoms.origins[('Cit', 1)], atoms.origins[('Cit', 5)]
        (('OAC', 4), ('AcCoA', 1))
    """

    reaction: str
    reactants: Sequence[tuple[str, str]]
    products: Sequence[tuple[str, str]]

    def _check(self) -> None:
        """Raise an error for a product letter that no reactant carries, or a letter two reactant atoms share.

        Raises:
            ValidationError:
                A letter is missing or repeated.
        """
        letters = [letter for _, atoms in self.reactants for letter in atoms]
        for letter in letters:
            if letters.count(letter) > 1:
                error_msg = f'The reactants of the reaction {self.reaction} repeat the atom letter {letter!r}.'
                raise ValidationError(error_msg)
        for metabolite, atoms in self.products:
            for letter in atoms:
                if letter not in letters:
                    error_msg = (
                        f'The product {metabolite} of the reaction {self.reaction} names the atom letter {letter!r}, '
                        'which no reactant carries.'
                    )
                    raise ValidationError(error_msg)

    @property
    def origins(self) -> dict[tuple[str, int], tuple[str, int]]:
        """Reactant atom of every product atom, as pairs of a metabolite and an atom counted from 1."""
        places = {
            letter: (metabolite, position + 1)
            for metabolite, atoms in self.reactants
            for position, letter in enumerate(atoms)
        }
        return {
            (metabolite, position + 1): places[letter]
            for metabolite, atoms in self.products
            for position, letter in enumerate(atoms)
        }


@dataclass(frozen=True)
class TracerSpec(BaseSpec):
    """Labelling of a substrate, the fraction of every isotopomer as a bit string with atom 1 first.

    Args:
        metabolite:
            The metabolite.

        fractions:
            The fraction of every isotopomer, such as `{'00': 0.5, '01': 0.25, '11': 0.25}`, where `1` is a labelled
            atom. The fractions sum to 1.

    Raises:
        ValidationError:
            The fractions do not sum to 1, or their bit strings differ in length, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import TracerSpec
        >>> TracerSpec('AcCoA', {'00': 0.5, '11': 0.5}).size
        2
    """

    metabolite: str
    fractions: Mapping[str, float]

    def _check(self) -> None:
        """Raise an error for fractions that do not sum to 1, or isotopomers of different lengths.

        Raises:
            ValidationError:
                The fractions are not a distribution over isotopomers of one length.
        """
        if abs(sum(self.fractions.values()) - 1.0) > SUM_TOLERANCE:
            error_msg = f'The isotopomer fractions of {self.metabolite} sum to {sum(self.fractions.values())}, not 1.'
            raise ValidationError(error_msg)
        if len({len(item) for item in self.fractions}) != 1:
            error_msg = f'The isotopomers of {self.metabolite} differ in length: {", ".join(self.fractions)}.'
            raise ValidationError(error_msg)

    @property
    def size(self) -> int:
        """Number of atoms of the metabolite."""
        return len(next(iter(self.fractions)))


@dataclass(frozen=True)
class FragmentSpec(BaseSpec):
    """Fragment of a metabolite whose mass isotopomer distribution a simulation gives or an experiment measures.

    Args:
        metabolite:
            The metabolite.

        atoms:
            The atoms of the fragment, counted from 1.

        name:
            The name of the fragment, or the empty string for the name of the metabolite.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import FragmentSpec
        >>> FragmentSpec('Glu', (1, 2, 3, 4, 5)).label
        'Glu'
    """

    metabolite: str
    atoms: Sequence[int]
    name: str = ''

    def _check(self) -> None:
        """Accept the spec, which has nothing to check."""

    @property
    def label(self) -> str:
        """Name of the fragment, the name of its metabolite without one of its own."""
        return self.name or self.metabolite


@dataclass(frozen=True)
class LabelMeasurementSpec(BaseSpec):
    """Measured mass isotopomer distribution of a fragment in one tracer experiment.

    Args:
        experiment:
            The name of the tracer experiment.

        fragment:
            The name of the fragment.

        values:
            The measured distribution.

        deviation:
            The standard deviation of every value.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import LabelMeasurementSpec
        >>> LabelMeasurementSpec('a', 'Glu', (0.5, 0.5), 0.002).deviation
        0.002
    """

    experiment: str
    fragment: str
    values: npt.ArrayLike = field(repr=False)
    deviation: float = 0.002

    def _check(self) -> None:
        """Raise an error for a deviation that is not positive.

        Raises:
            ValidationError:
                The deviation is not positive.
        """
        if not self.deviation > 0:
            error_msg = f'The deviation of a measurement of {self.fragment} must be positive, not {self.deviation}.'
            raise ValidationError(error_msg)

    @property
    def distribution(self) -> npt.NDArray[np.float64]:
        """Measured distribution as an array."""
        return np.asarray(self.values, dtype=float)
