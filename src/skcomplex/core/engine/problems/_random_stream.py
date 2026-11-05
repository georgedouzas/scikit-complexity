"""Hold the random stream that a run passes to the functions of its laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

import numpy as np

from ...errors import ValidationError


class RandomStream:
    """Random stream of one problem, which a solver binds when it prepares and functions of laws read when it runs.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.problems._random_stream import RandomStream
        >>> stream = RandomStream().bind(np.random.default_rng(0))
        >>> 0.0 <= float(stream.generator.random()) < 1.0
        True
    """

    generator_: np.random.Generator | None = None

    def bind(self, generator: np.random.Generator) -> Self:
        """Bind the generator of a solver.

        Args:
            generator:
                The random number generator of the solver.

        Returns:
            stream:
                The stream.
        """
        self.generator_ = generator
        return self

    @property
    def generator(self) -> np.random.Generator:
        """Generator the solver bound, which raises an error outside a run."""
        if self.generator_ is None:
            error_msg = (
                'A function with `random=True` asks for the random stream outside a run. A solver binds the stream '
                'when it prepares, so run the law through a simulation.'
            )
            raise ValidationError(error_msg)
        return self.generator_
