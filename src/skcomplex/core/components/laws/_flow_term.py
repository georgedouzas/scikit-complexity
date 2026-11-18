"""Define the rate a flow adds to one of its stocks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from sage.all__sagemath_symbolics import SR, Expression

from ...errors import ValidationError
from ._rate import RateLaw

SIGNS = (-1, 1)


class FlowTermLaw(RateLaw):
    """The rate a flow adds to one of its stocks, `sign * f`, with the sign it has in the flow.

    A flow `RateLaw({s.alive: -1, s.dead: +1}, f, label='deaths')` becomes the term `-f` on `alive` with the sign `-1`
    and the term `f` on `dead` with the sign `+1`, so the compiler knows which stock the flow leaves and which it
    enters without reading the sign of `f`.

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
    """

    def __init__(
        self,
        target: Expression,
        value: object,
        order: int | None = None,
        *,
        label: str = '',
        sign: int = 1,
    ) -> None:
        super().__init__(target, value, order, label=label)
        self.sign = sign

    @classmethod
    def _get_terms(cls, law: RateLaw, owner: str) -> list[RateLaw]:
        """Return the terms of a flow, one per stock, or the law itself when its target is no mapping.

        Args:
            law:
                The rate law.

            owner:
                The label of the interaction that returns the law, for messages.

        Returns:
            terms:
                The term of every stock, in the order of the mapping.

        Raises:
            ValidationError:
                The flow has no label, maps a stock to a sign other than `-1` or `+1`, or names no stock, two stocks it
                leaves, or two stocks it enters.
        """
        if not isinstance(law.target, Mapping):
            return [law]
        example = "such as RateLaw({s.alive: -1, s.dead: +1}, f, label='deaths')"
        if not law.label:
            error_msg = f'A flow of {owner!r} has no label. Name it, {example}.'
            raise ValidationError(error_msg)
        signs = list(law.target.values())
        if not signs or any(sign not in SIGNS for sign in signs) or len(set(signs)) < len(signs):
            error_msg = (
                f'The flow {law.label!r} of {owner!r} maps its stocks to the signs {signs}. A flow leaves at most one '
                f'stock, with the sign -1, and enters at most one, with the sign +1, {example}.'
            )
            raise ValidationError(error_msg)
        return [
            cls(stock, int(sign) * SR(law.value), 1, label=law.label, sign=int(sign))
            for stock, sign in law.target.items()
        ]
