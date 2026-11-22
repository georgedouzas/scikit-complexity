"""Define the draw spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ...base import BaseSpec
from ...enums import DrawKind
from ...errors import ValidationError

RELAXED_KINDS = frozenset({DrawKind.BERNOULLI, DrawKind.CATEGORICAL})


@dataclass(frozen=True)
class DrawSpec(BaseSpec):
    """A spec of a random draw in an expression, with its kind and its name.

    The compiler finds the draws of a law and reads each as this spec, such as the spec of kind `'bernoulli'` of
    `bernoulli('flip', 0.5)` inside the update of a coin, which is 1 with probability one half at every step. A draw
    from a distribution of `scipy.stats`, such as `draw('wait', 'expon')`, also holds the name of the distribution and
    its keywords. With a temperature, a Bernoulli or categorical draw is replaced by a smooth draw that the JAX backend
    can take derivatives through. It draws the flip of a coin, the waiting time of a customer, and the mutation of a
    gene.

    Read more in the user guide, under Laws.

    Args:
        kind:
            `'uniform'`, `'normal'`, `'bernoulli'`, `'categorical'`, or `'draw'` for a distribution of `scipy.stats`.

        name:
            The name of the draw.

        distribution:
            The name of the `scipy.stats` distribution of a draw of kind `'draw'`, or `None`.

        keywords:
            The keyword of every argument of a draw of kind `'draw'`.

        arguments:
            The arguments of the distribution in a call, empty for the draw of a function.

        call:
            The symbolic call of the draw in the expression it was found in, or `None` for the draw of a function.

        temperature:
            The temperature of the smooth draw that the JAX backend draws in place of a Bernoulli or categorical draw,
            or `None` for the draw itself.

        hard:
            Whether a draw with a temperature gives the exact value, and the derivatives of the smooth draw, on JAX.

    See Also:
        bernoulli: Returns a draw of 1 with a probability and 0 otherwise.
        draw: Returns a draw from a distribution of `scipy.stats`.

    Examples:
        >>> from skcomplex.core.components.laws.utils import bernoulli
        >>> from skcomplex.core.components.laws.utils._draws import draws_in
        >>> draws_in(bernoulli('flip', 0.5))[0].kind
        'bernoulli'
    """

    kind: str
    name: str
    distribution: str | None = None
    keywords: tuple[str, ...] = ()
    arguments: tuple[Expression, ...] = ()
    call: Expression | None = None
    temperature: float | None = None
    hard: bool = False

    def _check(self) -> None:
        """Raise an error for a bad temperature, or `hard` without a temperature.

        Raises:
            ValidationError:
                The temperature is not a positive number, the draw is neither Bernoulli nor categorical, or `hard` is
                given without a temperature.
        """
        if self.temperature is None:
            if self.hard:
                error_msg = f'The draw {self.name!r} takes `hard=True` only with a temperature.'
                raise ValidationError(error_msg)
            return
        if self.kind not in RELAXED_KINDS:
            error_msg = (
                f'The {self.kind} draw {self.name!r} has a temperature, but only Bernoulli and categorical draws take '
                'one. Remove the temperature.'
            )
            raise ValidationError(error_msg)
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, int | float) or self.temperature <= 0:
            error_msg = f'The draw {self.name!r} takes a positive temperature, but got {self.temperature!r}.'
            raise ValidationError(error_msg)

    @property
    def parameters(self) -> dict[str, Expression]:
        """Arguments of a draw from a `scipy.stats` distribution by keyword."""
        return dict(zip(self.keywords, self.arguments, strict=True)) if self.kind == 'draw' else {}
