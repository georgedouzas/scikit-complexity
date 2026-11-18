"""Define the rate a flow adds to one of its stocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ..spaces._lattice import BOUNDARIES
from ._rate import RateLaw
from ._update import UpdateLaw
from .utils._reductions import shift_member

SIGNS = (-1, 1)
SHIFT = 'shift_member'


class FlowTermLaw(RateLaw):
    """A law of the rate `sign * f` that a flow adds to one of its stocks.

    It keeps the sign the stock has in the flow. A flow `RateLaw({s.alive: -1, s.dead: +1}, f, label='deaths')`
    becomes the term `-f` on `alive` with the sign `-1` and the term `f` on `dead` with the sign `+1`, so the compiler
    knows which stock the flow leaves and which it enters without reading the sign of `f`. In discrete time a flow is
    an increment, `UpdateLaw({s.lane: +1}, f, increment=True, label='entry')`, and its terms are increments too.

    Unlike a `RateLaw` on one state, it is one side of a flow, so the loops analysis and the flow matrix draw the flow
    from the stock it leaves to the stock it enters. Choose it when a pass of the compiler needs one side of a flow,
    and write the flow itself in a model.

    It appears in many fields. People die from the living into the dead, water drains from a tank, and capital
    depreciates out of an economy.

    Args:
        target:
            The symbol of the stock.

        value:
            The rate the flow adds to the stock, `sign * f`.

        order:
            The order of the derivative the rate gives, 1.

        label:
            The name of the flow.

        sign:
            The sign of the change of the stock, `-1` for the stock the flow leaves and `+1` for the one it enters.

        increment:
            Whether the term is the increment `sign * f` of the stock in one step of discrete time, not a rate.

        shifted:
            Whether the stock is a member at an offset, such as the next site of `{s.n: -1, shift_member(s.n, 1): +1}`,
            so every member takes the term of the member behind it.
    """

    def __init__(
        self,
        target: Expression,
        value: object,
        order: int | None = None,
        *,
        label: str = '',
        sign: int = 1,
        increment: bool = False,
        shifted: bool = False,
    ) -> None:
        super().__init__(target, value, order, label=label)
        self.sign = sign
        self.increment = increment
        self.shifted = shifted

    @classmethod
    def _get_terms(cls, law: RateLaw | UpdateLaw, owner: str) -> list[RateLaw | UpdateLaw]:
        """Return the terms of a flow, one per stock, or a law of one state as it is.

        A flow is a rate law, or in discrete time an increment update law, whose target maps stocks to signs.

        Args:
            law:
                The rate or update law.

            owner:
                The label of the interaction that returns the law, for messages.

        Returns:
            terms:
                The term of every stock, in the order of the mapping.

        Raises:
            ValidationError:
                The flow has no label, maps a stock to a sign other than `-1` or `+1`, or names no stock, two stocks it
                leaves, or two stocks it enters. An update law that maps stocks is no increment.
        """
        if not isinstance(law.target, Mapping):
            return [law]
        example = "such as RateLaw({s.alive: -1, s.dead: +1}, f, label='deaths')"
        if not law.label:
            error_msg = f'A flow of {owner!r} has no label. Name it, {example}.'
            raise ValidationError(error_msg)
        increment = isinstance(law, UpdateLaw)
        if increment and not law.increment:
            error_msg = (
                f'The flow {law.label!r} of {owner!r} is an update law that maps stocks, but not an increment. Write '
                f"it as one, such as UpdateLaw({{s.lane: +1}}, f, increment=True, label={law.label!r})."
            )
            raise ValidationError(error_msg)
        signs = list(law.target.values())
        if not signs or any(sign not in SIGNS for sign in signs) or len(set(signs)) < len(signs):
            error_msg = (
                f'The flow {law.label!r} of {owner!r} maps its stocks to the signs {signs}. A flow leaves at most one '
                f'stock, with the sign -1, and enters at most one, with the sign +1, {example}.'
            )
            raise ValidationError(error_msg)
        presence = cls._get_presence(law.target)
        terms = []
        for stock, sign in law.target.items():
            state, value = cls._get_unshifted(stock, int(sign) * SR(law.value), presence)
            shifted = state is not stock
            terms.append(cls(state, value, 1, label=law.label, sign=int(sign), increment=increment, shifted=shifted))
        return terms

    @staticmethod
    def _get_offset(stock: object) -> tuple[Expression, int, str] | None:
        """Return the state, the offset, and the boundary of a stock read at a member offset, or `None`."""
        if not isinstance(stock, Expression) or getattr(stock.operator(), 'name', lambda: '')() != SHIFT:
            return None
        state, offset, boundary, _ = stock.operands()
        return state, int(offset), BOUNDARIES[int(boundary)].value

    @classmethod
    def _get_presence(cls, stocks: Mapping[object, object]) -> Expression:
        """Return 1 where the member every stock of a flow names exists, and 0 otherwise.

        A member at an offset is missing past the ends of a fixed boundary, so nothing flows out of an open line.
        """
        presence = SR(1)
        for stock in stocks:
            found = cls._get_offset(stock)
            if found is not None and found[2] == 'fixed':
                state, offset, _ = found
                ahead = shift_member(state, offset, boundary='fixed')
                presence *= 1 + ahead - shift_member(state, offset, boundary='fixed', fill=1.0)
        return presence

    @classmethod
    def _get_unshifted(cls, stock: object, value: Expression, presence: Expression) -> tuple[object, Expression]:
        """Return the state of a stock and the term it takes.

        A flow into the member at an offset, such as `{s.n: -1, shift_member(s.n, 1): +1}` from every site to the next,
        gives every member the term of the member behind it. A member flows only where the member at the offset
        exists, which the presence says, and a member at an offset always has a member behind it.
        """
        found = cls._get_offset(stock)
        if found is None:
            return stock, value * presence
        state, offset, boundary = found
        return state, shift_member(value, -offset, boundary=boundary)
