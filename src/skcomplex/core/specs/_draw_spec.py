"""Define the draw spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from sage.all__sagemath_symbolics import Expression

from ..errors import ValidationError
from ._base_spec import BaseSpec

RELAXED_KINDS = frozenset({'bernoulli', 'categorical'})


@dataclass(frozen=True)
class DrawSpec(BaseSpec):
    """Random draw, with its kind and its name.

    For a distribution of `scipy.stats`, it also holds the distribution and its keywords.

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
            The temperature of the continuous relaxation of a Bernoulli or categorical draw, which the JAX backend
            draws in its place, or `None` for no relaxation.

        hard:
            Whether a relaxed draw gives the exact value and the gradient of the relaxation on JAX.

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
        """Raise an error unless the relaxed draw is discrete at a positive temperature.

        Raises:
            ValidationError:
                The temperature is not a positive number, or the draw is neither Bernoulli nor categorical, or `hard`
                is given without a temperature.
        """
        if self.temperature is None:
            if self.hard:
                error_msg = f'The draw {self.name!r} takes `hard=True` only with a temperature.'
                raise ValidationError(error_msg)
            return
        if self.kind not in RELAXED_KINDS:
            error_msg = f'The {self.kind} draw {self.name!r} has no relaxation. Relax Bernoulli or categorical draws.'
            raise ValidationError(error_msg)
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, int | float) or self.temperature <= 0:
            error_msg = f'The draw {self.name!r} takes a positive temperature, but got {self.temperature!r}.'
            raise ValidationError(error_msg)

    @property
    def parameters(self) -> dict[str, Expression]:
        """Arguments of a draw from a `scipy.stats` distribution by keyword."""
        return dict(zip(self.keywords, self.arguments, strict=True)) if self.kind == 'draw' else {}
