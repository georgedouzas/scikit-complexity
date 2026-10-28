"""Define transfer laws, the flows of a conserved quantity between states and the environment."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ...specs import RequirementSpec
from ._base import BaseLaw


class Source:
    """Environment a transfer draws a quantity from, outside the system, such as a bank that creates money.

    Args:
        label:
            The name of the environment in transaction-flow matrices.

    Examples:
        >>> from skcomplex.core.laws import Source
        >>> Source().label
        'environment'
    """

    def __init__(self, label: str = 'environment') -> None:
        self.label = label


class Sink:
    """Environment a transfer releases a quantity to, outside the system, such as a drain or the atmosphere.

    Args:
        label:
            The name of the environment in transaction-flow matrices.

    Examples:
        >>> from skcomplex.core.laws import Sink
        >>> Sink('drain').label
        'drain'
    """

    def __init__(self, label: str = 'environment') -> None:
        self.label = label


class TransferLaw(BaseLaw):
    """Movement of an amount of a quantity from a source holder to a sink holder, in continuous or discrete time.

    A transfer is one outflow of its source and one inflow of its sink. Its amount is a rate in a system of rate laws,
    the amount moved per step in a system of update laws, and the size of every jump when a `rate` is given. The
    trajectory records the amount at every frame under `<element path>.<label>`.

    Args:
        source:
            The symbol of the state the quantity leaves, or `Source()` for the environment.

        sink:
            The symbol of the state the quantity enters, or `Sink()` for the environment.

        amount:
            The symbolic amount, a rate, a per-step amount, or a jump size.

        rate:
            The propensity of a jump that moves `amount` at random times, or `None` for a deterministic flow.

        label:
            The name of the flow, unique among the flows of its element, or the empty string for `flow_<index>`.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws import Source, TransferLaw
        >>> level = var('level')
        >>> TransferLaw(Source(), level, 5, label='faucet').check().prepare().targets_
        (level,)
    """

    def __init__(
        self,
        source: Expression | Source,
        sink: Expression | Sink,
        amount: object,
        rate: object = None,
        *,
        label: str = '',
    ) -> None:
        super().__init__(label)
        self.source = source
        self.sink = sink
        self.amount = amount
        self.rate = rate

    def _get_targets(self) -> tuple[object, ...]:
        """Return the states the transfer moves the quantity between, without the environment."""
        return tuple(holder for holder in (self.source, self.sink) if isinstance(holder, Expression))

    def _get_requirements(self) -> RequirementSpec:
        """Return what the law requires of a system: nothing, since it holds in any time."""
        return RequirementSpec()

    def _check_holders(self) -> None:
        """Raise an error for a holder that is neither a scalar state symbol nor the environment of its side."""
        for name, holder, environment in (('source', self.source, Source), ('sink', self.sink, Sink)):
            if isinstance(holder, environment):
                continue
            if not isinstance(holder, Expression) or holder.operator() is not None:
                error_msg = (
                    f'The {name} of a transfer must be the symbol of a scalar state or `{environment.__name__}()`, '
                    f'but got {holder!r}.'
                )
                raise ValidationError(error_msg)
        if isinstance(self.source, Source) and isinstance(self.sink, Sink):
            error_msg = (
                'A transfer moves a quantity into or out of a state, but both of its holders are the environment.'
            )
            raise ValidationError(error_msg)

    def _check(self) -> Self:
        """Raise an error for invalid holders, an amount that is a NumPy array, or a symbolic jump size.

        Returns:
            law:
                The law itself.
        """
        self._check_holders()
        self._check_array('amount')
        if self.rate is not None and not SR(self.amount).is_numeric():
            error_msg = (
                f'The transfer {self.label} has a rate, so its amount is a jump size, but {self.amount} is a symbol.'
            )
            raise ValidationError(error_msg)
        return self

    def _run(self, inputs: object) -> object:
        """Return the amount at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.

        Returns:
            amount:
                The amount, a number when nothing symbolic is left.
        """
        return self._evaluate(self.amount, inputs)
