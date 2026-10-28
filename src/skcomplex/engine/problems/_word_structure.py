"""Word of a string rewriting system at one generation, with its segments where a word geometry is declared."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import KW_ONLY, dataclass

import numpy as np
import numpy.typing as npt

from ...specs import ModuleSpec
from ...types import AnyArray
from ._base import BaseStructure


@dataclass(frozen=True)
class WordStructure(BaseStructure):
    """Word of a string rewriting system at one generation, with its segments where a word geometry is declared.

    Args:
        alphabet:
            The symbol names, indexed by code.

        codes:
            The symbol code of every module.

        parameters:
            The parameters of every module, padded with NaN.

        generation:
            The generation.

        strokes:
            The segments of shape `(k, 2, d)`, or `None` without a word geometry.

        depths:
            The branch depth of every segment, or `None` without a word geometry.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.problems import WordStructure
        >>> WordStructure(('A', 'B'), np.array([0, 1, 0]), np.zeros((3, 0)), 2).text
        'ABA'
    """

    alphabet: tuple[str, ...]
    codes: npt.NDArray[np.int64]
    parameters: npt.NDArray[np.float64]
    generation: int
    _: KW_ONLY
    strokes: npt.NDArray[np.float64] | None = None
    depths: npt.NDArray[np.int64] | None = None

    @property
    def parts(self) -> int:
        """Number of modules."""
        return len(self.codes)

    @property
    def has_geometry(self) -> bool:
        """Whether a word geometry has drawn the word."""
        return self.strokes is not None

    def modules(self) -> list[ModuleSpec]:
        """Return the modules of the word with their parameters.

        Returns:
            modules:
                The modules in order.
        """
        modules = []
        for code, row in zip(self.codes.tolist(), self.parameters, strict=True):
            values = [float(value) for value in row if not np.isnan(value)]
            modules.append(ModuleSpec(self.alphabet[code], tuple(values)))
        return modules

    @property
    def text(self) -> str:
        """Word as written, with modules separated by spaces when one has parameters or a long name."""
        modules = self.modules()
        texts = [
            module.name if not module.parameters else f'{module.name}({", ".join(f"{v:g}" for v in module.parameters)})'
            for module in modules
        ]
        spaced = any(module.parameters or len(module.name) > 1 for module in modules)
        return (' ' if spaced else '').join(texts)

    def count(self, name: str) -> int:
        """Return the number of modules of a symbol.

        Args:
            name:
                The symbol.

        Returns:
            count:
                The number of modules with that symbol.
        """
        if name not in self.alphabet:
            return 0
        return int(np.count_nonzero(self.codes == self.alphabet.index(name)))

    def segments(self) -> AnyArray:
        """Return the segments of the word geometry.

        Returns:
            segments:
                The array of shape `(k, 2, d)`, empty without a word geometry.
        """
        return np.zeros((0, 2, 2)) if self.strokes is None else self.strokes
