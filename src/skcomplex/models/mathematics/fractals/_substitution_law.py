"""Define substitution laws, which replace the tiles of patches by placed pieces."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

from ....core.laws import BaseLaw
from ....core.spaces import DiscreteTime
from ....specs import RequirementSpec
from ._affine_spec import AffineSpec
from ._rewriting_error import RewritingError


class SubstitutionLaw(BaseLaw):
    """Substitution that replaces every tile of a prototile by pieces placed relative to the tile.

    A patch element returns its substitutions from `_get_laws`, and every generation applies all of them at once.

    Args:
        prototile:
            The name of the prototile replaced.

        pieces:
            Pairs of a prototile name and the affine map from the parent tile's frame.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from skcomplex.models.mathematics.fractals import SubstitutionLaw
        >>> from skcomplex.models.mathematics.fractals import AffineSpec
        >>> len(SubstitutionLaw('interval', [('interval', AffineSpec(scale=1 / 3))]).check().pieces)
        1
    """

    def __init__(self, prototile: str, pieces: Sequence[tuple[str, AffineSpec]], *, label: str = '') -> None:
        super().__init__(label)
        self.prototile = prototile
        self.pieces = pieces

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states, since a substitution rewrites the tiles of its element as a whole."""
        return ()

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: discrete time."""
        return RequirementSpec(time=(DiscreteTime,))

    def _check(self) -> Self:
        """Raise an error for a piece that is not a pair of a prototile name and an affine map.

        Returns:
            law:
                The law itself.
        """
        for piece in self.pieces:
            if not (isinstance(piece, tuple) and len(piece) == 2 and isinstance(piece[1], AffineSpec)):  # noqa: PLR2004
                error_msg = f'A piece of a substitution is a prototile name and an `AffineSpec`, not {piece!r}.'
                raise RewritingError(error_msg)
            piece[1].check()
        return self

    def _run(self, inputs: object) -> object:
        """Return the pieces with the scale, the rotation, and the translation of their maps at given values.

        Args:
            inputs:
                A mapping from the name of every symbol the maps read to its value.

        Returns:
            pieces:
                Pairs of a prototile name and the scale, the rotation, and the translation of its map.
        """
        return [
            (name, self._evaluate((affine.scale, affine.rotation, tuple(affine.translation)), inputs))
            for name, affine in self.pieces
        ]
