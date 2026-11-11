"""Define the law that moves an amount between two holders."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass
from typing import ClassVar

from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ...specs import BaseSpec, RequirementSpec
from ._base import BaseLaw

SHIFT = 'shift_member'


@dataclass(frozen=True)
class Source(BaseSpec):
    """A holder outside the system that a transfer takes an amount from.

    Unlike a state, it has no value and never runs out. Choose it when an amount enters the system from outside, as the
    source of a `TransferLaw` or of a leg of a `TransferInteraction`.

    It appears in many fields as the origin of what enters a system, such as a bank that creates money, births into a
    population, or an inflow into a reservoir.

    Args:
        label:
            The name of the environment in transaction-flow matrices.

    Examples:
        >>> from skcomplex.core.components.laws import Source
        >>> Source().label
        'environment'
    """

    label: str = 'environment'


@dataclass(frozen=True)
class Sink(BaseSpec):
    """A holder outside the system that a transfer releases an amount to.

    Unlike a state, it has no value and never fills. Choose it when an amount leaves the system, as the sink of a
    `TransferLaw` or of a leg of a `TransferInteraction`.

    It appears in many fields as the end of what leaves a system, such as a drain, the atmosphere, or deaths out of a
    population.

    Args:
        label:
            The name of the environment in transaction-flow matrices.

    Examples:
        >>> from skcomplex.core.components.laws import Sink
        >>> Sink('drain').label
        'drain'
    """

    label: str = 'environment'


class TransferLaw(BaseLaw):
    """A law that moves an amount from a source holder to a sink holder.

    A transfer is one outflow of its source and one inflow of its sink, so what one loses the other gains. Its `amount`
    is a rate in continuous time, the amount moved per step in discrete time, and the size of every jump when a `rate`
    is given. The trajectory records the amount at every frame under `<path>.<label>`, the path of the component that
    declares the transfer, or `owner`. One holder may be a state read by `shift_member` some places along a group, so
    every member transfers with the member that many places along. Past the ends of a fixed boundary no member is there,
    and the transfer moves nothing.

    Unlike `RateLaw`, which changes one state, it pairs a loss with a gain, so the amount is conserved. Choose it when a
    quantity moves between holders and its flow is worth recording.

    It appears in many fields. People move from susceptible to infected, money moves between accounts, water flows
    between reservoirs, and stock moves along a supply chain.

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
            The name of the flow, unique among the flows reported under one component, or the empty string for
            `flow_<index>`.

        owner:
            The path of the component the flow is reported under, relative to the system of the component that declares
            the transfer, such as the element an interaction moves a quantity within, or the empty string for the
            component that declares it.

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

    def _prepare(self, context: str | None = None) -> None:
        """Raise an error for bad holders, an array amount, a symbolic jump, or bad owner.

        The jump size may not be symbolic, and the owner must be a string.

        Args:
            context:
                The path of the element or interaction that returns the law, or `None`.
        """
        super()._prepare(context)
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
