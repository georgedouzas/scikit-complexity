"""Define the noise law the compiler keeps with the order of its derivative."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._noise import NoiseLaw


@dataclass(frozen=True)
class _OrderedNoiseLaw(NoiseLaw):
    """A noise law on a state that keeps the order of the derivative it drives.

    The compiler writes `NoiseLaw(x.diff(t, k), g)` as this law on `x` of the order `k`. A noise of the order `None`
    drives the derivative of the order of its state.

    Args:
        target:
            The symbol of a scalar state, or the column of symbols of a vector state.

        value:
            A symbolic expression, a `Call`, or a callable `f(t, values)`.

        order:
            The order of the derivative the noise drives, or `None` for the order of its state.

        convention:
            The stochastic calculus of the law, `'ito'`, `'stratonovich'`, or `'anti-ito'`.

        process:
            `None` for independent processes, a label, a tuple of labels, or the handle of a Wiener process.

        conserved:
            Whether the noise is the divergence of a random flux on the links of the lattice of a field.

        hurst:
            The Hurst index of a fractional Brownian motion, or `None` for a Wiener process.

        label:
            The name of the law, or the empty string for a law without one.
    """

    order: float | None = None

    @classmethod
    def _get_name(cls) -> str:
        """Return the name of the law in messages, the class users write it with."""
        return 'NoiseLaw'

    def _get_order(self) -> float | None:
        """Return the order of the derivative the noise drives."""
        return self.order
