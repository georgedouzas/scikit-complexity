"""Define the law that replaces every tile of a prototile by placed pieces."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

from ....core.components.laws import BaseLaw, RequirementSpec, UpdateLaw
from ....core.errors import RewritingError
from ._affine_spec import AffineSpec

PIECE_SIZE = 2


@dataclass(frozen=True)
class SubstitutionLaw(BaseLaw):
    """A law that replaces every tile of a prototile by smaller placed pieces.

    A prototile is a shape of tile. At every generation each tile of that shape is replaced by `pieces`, placed relative
    to the tile. An interaction that binds a patch sets its substitutions, and every generation applies all of them at
    once.

    Unlike `ProductionLaw`, which rewrites symbols of a word, it rewrites tiles of the plane. Choose it when a tiling
    grows by inflation and subdivision.

    It appears in mathematics and materials science. The Penrose tiling grows by substitution, and quasicrystals have
    the order of such tilings.

    Args:
        prototile:
            The name of the prototile replaced.

        pieces:
            Pairs of a prototile name and the affine map from the parent tile's frame.

        label:
            The name of the law, or the empty string for a law without one.

    Examples:
        >>> from skcomplex.systems.general.rewriting import SubstitutionLaw
        >>> from skcomplex.systems.general.rewriting import AffineSpec
        >>> len(SubstitutionLaw('interval', [('interval', AffineSpec(scale=1 / 3))]).pieces)
        1
    """

    law_tag: ClassVar[str] = 'update'

    prototile: str
    pieces: Sequence[tuple[str, AffineSpec]]
    _: KW_ONLY
    label: str = ''

    def _get_targets(self) -> tuple[object, ...]:
        """Return no states.

        A substitution rewrites the tiles of its element as a whole.
        """
        return ()

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: the discrete time of an update."""
        return UpdateLaw.get_requirements()

    def _check(self) -> None:
        """Raise an error for a piece that is not a pair of a prototile name and a map.

        The map must be an affine map.
        """
        super()._check()
        for piece in self.pieces:
            if not (isinstance(piece, tuple) and len(piece) == PIECE_SIZE and isinstance(piece[1], AffineSpec)):
                error_msg = f'A piece of a substitution is a prototile name and an `AffineSpec`, not {piece!r}.'
                raise RewritingError(error_msg)
            piece[1]._check()

    def _get_value(self, inputs: object) -> object:
        """Return the pieces with their maps at given values.

        It gives the scale, the rotation, and the translation of their maps.

        Args:
            inputs:
                A mapping from the name of every symbol the maps read to its value.

        Returns:
            pieces:
                The name of every piece with the scale, the rotation, and the translation of its map.
        """
        return [
            (name, self._evaluate((affine.scale, affine.rotation, tuple(affine.translation)), inputs))
            for name, affine in self.pieces
        ]
