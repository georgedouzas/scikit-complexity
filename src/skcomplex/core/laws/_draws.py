"""Define the named random draws that update and renewal laws take from the simulation's random stream."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sage.all__sagemath_symbolics import SR
from sage.symbolic.function import BuiltinFunction

from ..base import Expression, SymbolicVector, ValidationError

DRAW_KINDS = ('uniform', 'normal', 'bernoulli', 'categorical', 'draw')


class _Draw(BuiltinFunction):
    """Random value of one kind under one name, which stays symbolic until the compiler lowers it."""

    def __init__(self, kind: str, name: str, distribution: str | None = None, keywords: tuple[str, ...] = ()) -> None:
        self.kind, self.draw, self.distribution, self.keywords = kind, name, distribution, keywords
        suffix = ''.join(f'_{keyword}' for keyword in keywords)
        super().__init__(f'{distribution or kind}_{name}{suffix}', nargs=0)

    def __reduce__(self) -> tuple[type['_Draw'], tuple[str, str, str | None, tuple[str, ...]]]:
        """Return the kind, the name, the distribution, and the keywords that rebuild the function."""
        return _Draw, (self.kind, self.draw, self.distribution, self.keywords)

    def _eval_(self, *arguments: Expression) -> None:
        """Return `None`, so the draw stays unevaluated."""
        del arguments

    def _tderivative_(self, *arguments: Expression, diff_param: Expression) -> Expression:
        """Return zero, since a draw does not depend on the symbols of its arguments smoothly."""
        del arguments, diff_param
        return SR(0)


def _call(
    kind: str,
    name: str,
    arguments: Sequence[object],
    distribution: str | None = None,
    keywords: tuple[str, ...] = (),
) -> Expression:
    """Return the symbolic call of a draw of a kind under a name, after checking the name."""
    if not isinstance(name, str) or not name.isidentifier():
        error_msg = f'A {kind} draw takes a name that is a Python identifier, such as `turn`, but got {name!r}.'
        raise ValidationError(error_msg)
    return _Draw(kind, name, distribution, keywords)(*(SR(argument) for argument in arguments))


def _is_symbolic(item: object) -> bool:
    """Check whether an item converts to a symbolic expression."""
    try:
        SR(item)
    except (TypeError, ValueError):
        return False
    return True


def uniform(name: str, low: object = 0, high: object = 1) -> Expression:
    """Return a value drawn uniformly between two bounds at every step.

    Every use of the same name in one component reads the same value within a step, and every member of a group
    draws its own value.

    Args:
        name:
            The name of the draw, a Python identifier.

        low:
            The lower bound, a number or an expression of states and parameters.

        high:
            The upper bound.

    Returns:
        draw:
            The symbolic draw.

    Raises:
        ValidationError:
            The name is not a Python identifier.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import uniform
        >>> x = symbol('x')
        >>> uniform('u', 0, x) + uniform('u', 0, x)
        2*uniform_u(0, x)
    """
    return _call('uniform', name, (low, high))


def normal(name: str, mean: object = 0, deviation: object = 1) -> Expression:
    """Return a value drawn from a normal distribution at every step.

    Args:
        name:
            The name of the draw, a Python identifier.

        mean:
            The mean, a number or an expression of states and parameters.

        deviation:
            The standard deviation.

    Returns:
        draw:
            The symbolic draw.

    Raises:
        ValidationError:
            The name is not a Python identifier.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import normal
        >>> x = symbol('x')
        >>> x + normal('step', 0, 1)
        x + normal_step(0, 1)
    """
    return _call('normal', name, (mean, deviation))


def bernoulli(name: str, p: object) -> Expression:
    """Return 1 with a probability and 0 otherwise, drawn at every step.

    Args:
        name:
            The name of the draw, a Python identifier.

        p:
            The probability of 1, a number or an expression of states and parameters.

    Returns:
        draw:
            The symbolic draw.

    Raises:
        ValidationError:
            The name is not a Python identifier.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import bernoulli
        >>> q = symbol('q')
        >>> 1 - bernoulli('flip', q)
        -bernoulli_flip(q) + 1
    """
    return _call('bernoulli', name, (p,))


def categorical(name: str, weights: Sequence[object]) -> Expression:
    """Return the index of a category drawn with probabilities proportional to weights at every step.

    Args:
        name:
            The name of the draw, a Python identifier.

        weights:
            The nonnegative weights of the categories `0` to `len(weights) - 1`, numbers or expressions of states and
            parameters.

    Returns:
        draw:
            The symbolic draw.

    Raises:
        ValidationError:
            The name is not a Python identifier, or there are no weights.

    Examples:
        >>> from skcomplex.core.laws import categorical
        >>> categorical('map', [1, 1, 2])
        categorical_map(1, 1, 2)
    """
    if isinstance(weights, str) or not len(weights):
        error_msg = f'The categorical draw {name!r} takes a nonempty sequence of weights, but got {weights!r}.'
        raise ValidationError(error_msg)
    return _call('categorical', name, tuple(weights))


def draw(name: str, distribution: str, **parameters: object) -> Expression:
    """Return a value drawn from a distribution of `scipy.stats` at every step or renewal.

    The compiler checks the distribution name against `scipy.stats` and the keywords against the distribution's shape
    parameters, `loc`, and `scale`, and samples with the distribution's `rvs` and the simulation's random stream.

    Args:
        name:
            The name of the draw, a Python identifier.

        distribution:
            The name of a distribution of `scipy.stats`, such as `'pareto'`, `'lomax'`, or `'levy_stable'`.

        **parameters:
            The shape parameters, `loc`, and `scale` of the distribution, numbers or expressions of states and
            parameters.

    Returns:
        draw:
            The symbolic draw.

    Raises:
        ValidationError:
            The name is not a Python identifier, or the distribution is not given by its name.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import Draw, draw
        >>> alpha = symbol('alpha')
        >>> wait = draw('wait', 'pareto', b=alpha, scale=2)
        >>> wait
        pareto_wait_b_scale(alpha, 2)
        >>> Draw.of(wait).parameters
        {'b': alpha, 'scale': 2}
    """
    if not isinstance(distribution, str):
        error_msg = f'The draw {name!r} takes the name of a scipy.stats distribution, but got {distribution!r}.'
        raise ValidationError(error_msg)
    return _call('draw', name, tuple(parameters.values()), distribution, tuple(parameters))


@dataclass(frozen=True)
class Draw:
    """Random draw found in an expression.

    Args:
        kind:
            The kind of the draw, one of `'uniform'`, `'normal'`, `'bernoulli'`, `'categorical'`, and `'draw'` for a
            distribution of `scipy.stats`.

        name:
            The name of the draw.

        arguments:
            The arguments of the distribution.

        call:
            The symbolic call of the draw.

        distribution:
            The name of the `scipy.stats` distribution of a draw of kind `'draw'`, or `None`.

        keywords:
            The keyword of every argument of a draw of kind `'draw'`.

    Examples:
        >>> from skcomplex.core.laws import Draw, bernoulli
        >>> Draw.of(bernoulli('flip', 0.5)).kind
        'bernoulli'
    """

    kind: str
    name: str
    arguments: tuple[Expression, ...]
    call: Expression
    distribution: str | None = None
    keywords: tuple[str, ...] = ()

    @property
    def parameters(self) -> dict[str, Expression]:
        """Arguments of a draw from a `scipy.stats` distribution by keyword."""
        return dict(zip(self.keywords, self.arguments, strict=True)) if self.kind == 'draw' else {}

    @classmethod
    def of(cls, call: Expression) -> 'Draw':
        """Return the draw of a symbolic draw call.

        Args:
            call:
                A call returned by `uniform`, `normal`, `bernoulli`, `categorical`, or `draw`.

        Returns:
            draw:
                The draw.

        Raises:
            ValidationError:
                The expression is not a draw call.
        """
        operator = SR(call).operator()
        if not isinstance(operator, _Draw):
            error_msg = (
                f'{call} is not a random draw. Build draws with `uniform`, `normal`, `bernoulli`, `categorical`, or '
                '`draw`.'
            )
            raise ValidationError(error_msg)
        arguments = tuple(SR(call).operands())
        return cls(operator.kind, operator.draw, arguments, SR(call), operator.distribution, operator.keywords)


def is_draw(expression: object) -> bool:
    """Check whether an expression is a draw call.

    Args:
        expression:
            A symbolic expression or a number.

    Returns:
        found:
            Whether the expression is a call of a draw.

    Examples:
        >>> from skcomplex.core.laws import is_draw, uniform
        >>> is_draw(uniform('u')), is_draw(uniform('u') + 1)
        (True, False)
    """
    return isinstance(SR(expression).operator(), _Draw)


def draws_in(expression: object) -> list[Draw]:
    """Return every draw of an expression, including draws in the arguments of other draws, each listed once.

    Args:
        expression:
            A symbolic expression, a number, a vector of expressions, or a sequence of them.

    Returns:
        draws:
            The draws, outer draws before the draws in their arguments.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import bernoulli, draws_in, uniform
        >>> x = symbol('x')
        >>> [draw.name for draw in draws_in(x * uniform('u') + bernoulli('flip', uniform('u')))]
        ['u', 'flip']
    """
    found: dict[str, Draw] = {}
    items: list[Any] = list(expression) if isinstance(expression, SymbolicVector | list | tuple) else [expression]
    stack = [SR(item) for item in items if _is_symbolic(item)]
    while stack:
        item = stack.pop()
        if item.operator() is None:
            continue
        if is_draw(item):
            found.setdefault(str(item), Draw.of(item))
        stack.extend(reversed(item.operands()))
    return list(found.values())
