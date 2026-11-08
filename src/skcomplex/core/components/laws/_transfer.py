"""Define transfer laws.

They are the flows of a conserved quantity between states and the environment.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ...specs import RequirementSpec
from ._base import BaseLaw

SHIFT = 'shift_member'


class Source:
    """Environment a transfer draws a quantity from, outside the system.

    An example is a bank that creates money.

    Args:
        label:
            The name of the environment in transaction-flow matrices.

    Examples:
        >>> from skcomplex.core.components.laws import Source
        >>> Source().label
        'environment'
    """

    def __init__(self, label: str = 'environment') -> None:
        self.label = label


class Sink:
    """Environment a transfer releases a quantity to, outside the system.

    Examples are a drain or the atmosphere.

    Args:
        label:
            The name of the environment in transaction-flow matrices.

    Examples:
        >>> from skcomplex.core.components.laws import Sink
        >>> Sink('drain').label
        'drain'
    """

    def __init__(self, label: str = 'environment') -> None:
        self.label = label


class TransferLaw(BaseLaw):
    """Movement of an amount of a quantity from a source holder to a sink holder.

    It runs in continuous or discrete time. A transfer is one outflow of its source and one inflow of its sink. Its
    amount is a rate in a system of rate laws, the amount moved per step in a system of update laws, and the size of
    every jump when a `rate` is given. The trajectory records the amount at every frame under `<path>.<label>`, the path
    of the part that declares the transfer, or `owner` when given. One holder may be a state of a group read by
    `shift_member` at a member offset, so every member transfers with the member that many places along the group order.
    Past the ends of a fixed boundary no member is there, and the transfer moves nothing.

    Args:
        source:
            The symbol of the state the quantity leaves, the symbol read at a member offset, or `Source()` for the
            environment.

        sink:
            The symbol of the state the quantity enters, the symbol read at a member offset, or `Sink()` for the
            environment.

        amount:
            The symbolic amount, a rate, a per-step amount, or a jump size.

        rate:
            The propensity of a jump that moves `amount` at random times, or `None` for a deterministic flow.

        label:
            The name of the flow, unique among the flows reported under one part, or the empty string for
            `flow_<index>`.

        owner:
            The path of the part the flow is reported under, relative to the system of the part that declares the
            transfer, such as the element an interaction moves a quantity within, or the empty string for the part
            that declares it.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws import Source, TransferLaw
        >>> level = var('level')
        >>> TransferLaw(Source(), level, 5, label='faucet').prepare().targets_
        (level,)
    """

    law_tag: ClassVar[str] = 'update'

    def __init__(
        self,
        source: Expression | Source,
        sink: Expression | Sink,
        amount: object,
        rate: object = None,
        *,
        label: str = '',
        owner: str = '',
    ) -> None:
        super().__init__(label)
        self.source = source
        self.sink = sink
        self.amount = amount
        self.rate = rate
        self.owner = owner

    def _get_targets(self) -> tuple[object, ...]:
        """Return the states the transfer moves the quantity between.

        It leaves out the environment.
        """
        return tuple(holder for holder in (self.source, self.sink) if isinstance(holder, Expression))

    @classmethod
    def _get_requirements(cls) -> RequirementSpec:
        """Return what the law requires of a system: nothing.

        It holds in any time.
        """
        return RequirementSpec()

    @staticmethod
    def _is_shifted(holder: object) -> bool:
        """Check whether a holder is a state symbol read at a shifted member.

        The member is offset by `shift_member`.
        """
        if not isinstance(holder, Expression) or holder.operator() is None:
            return False
        operand = holder.operands()[0]
        return getattr(holder.operator(), 'name', lambda: '')() == SHIFT and operand.operator() is None

    def _check_holders(self) -> None:
        """Raise an error for a holder that is not a scalar state or the environment.

        The environment is that of its side.
        """
        for name, holder, environment in (('source', self.source, Source), ('sink', self.sink, Sink)):
            if isinstance(holder, environment) or self._is_shifted(holder):
                continue
            if not isinstance(holder, Expression) or holder.operator() is not None:
                error_msg = (
                    f'The {name} of a transfer must be the symbol of a scalar state, the symbol read at a member '
                    f'offset, or `{environment.__name__}()`, but got {holder!r}.'
                )
                raise ValidationError(error_msg)
        if self._is_shifted(self.source) and self._is_shifted(self.sink):
            error_msg = 'Both holders of a transfer are read at a member offset. Read one of them at the member itself.'
            raise ValidationError(error_msg)
        if isinstance(self.source, Source) and isinstance(self.sink, Sink):
            error_msg = (
                'A transfer moves a quantity into or out of a state, but both of its holders are the environment.'
            )
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for bad holders, an array amount, a symbolic jump, or bad owner.

        The jump size may not be symbolic, and the owner must be a string.
        """
        self._check_holders()
        if not isinstance(self.owner, str):
            error_msg = f'The owner of the transfer {self.label} is {self.owner!r}. Give a path, or an empty string.'
            raise ValidationError(error_msg)
        self._check_array('amount')
        if self.rate is not None and not SR(self.amount).is_numeric():
            error_msg = (
                f'The transfer {self.label} has a rate, so its amount is a jump size, but {self.amount} is a symbol.'
            )
            raise ValidationError(error_msg)

    def _run(self, inputs: object) -> None:
        """Set `value_` to the amount at given values of the symbols it reads.

        Args:
            inputs:
                A mapping from the name of every symbol the law reads to its value.
        """
        self.value_ = self._evaluate(self.amount, inputs)
