"""Define the draw spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

from sage.all__sagemath_symbolics import SR, Expression, function
from sage.modules.free_module_element import FreeModuleElement

from ..errors import ValidationError
from ._base_spec import BaseSpec


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

    Examples:
        >>> from skcomplex.core.components.laws.utils import bernoulli, draws_in
        >>> draws_in(bernoulli('flip', 0.5))[0].kind
        'bernoulli'
    """

    kind: str
    name: str
    distribution: str | None = None
    keywords: tuple[str, ...] = ()
    arguments: tuple[Expression, ...] = ()
    call: Expression | None = None

    @property
    def parameters(self) -> dict[str, Expression]:
        """Arguments of a draw from a `scipy.stats` distribution by keyword."""
        return dict(zip(self.keywords, self.arguments, strict=True)) if self.kind == 'draw' else {}


DRAWS: dict[str, DrawSpec] = {}


def _call(
    kind: str,
    name: str,
    arguments: Sequence[object],
    distribution: str | None = None,
    keywords: tuple[str, ...] = (),
) -> Expression:
    """Return the symbolic call of a draw of a kind under a name.

    The name is checked first.
    """
    if not isinstance(name, str) or not name.isidentifier():
        error_msg = f'A {kind} draw takes a name that is a Python identifier, such as `turn`, but got {name!r}.'
        raise ValidationError(error_msg)
    label = f'{distribution or kind}_{name}{"".join(f"_{keyword}" for keyword in keywords)}'
    DRAWS[label] = DrawSpec(kind, name, distribution, keywords)
    draw_function = function(label, nargs=0, derivative_func=lambda *_, **__: SR(0))
    return draw_function(*(SR(argument) for argument in arguments))


def uniform(name: str, low: object = 0, high: object = 1) -> Expression:
    """Return a value drawn uniformly between two bounds at every step.

    Every use of the same name in one element reads the same value within a step, and every member of a group draws
    its own value.

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
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import uniform
        >>> x = var('x')
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
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import normal
        >>> var('x') + normal('step', 0, 1)
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
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import bernoulli
        >>> 1 - bernoulli('flip', var('q'))
        -bernoulli_flip(q) + 1
    """
    return _call('bernoulli', name, (p,))


def categorical(name: str, weights: Sequence[object]) -> Expression:
    """Return the index of a category drawn at every step.

    The probabilities of the categories are proportional to the weights.

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
        >>> from skcomplex.core.components.laws.utils import categorical
        >>> categorical('map', [1, 1, 2])
        categorical_map(1, 1, 2)
    """
    if isinstance(weights, str) or not len(weights):
        error_msg = f'The categorical draw {name!r} takes a nonempty sequence of weights, but got {weights!r}.'
        raise ValidationError(error_msg)
    return _call('categorical', name, tuple(weights))


def draw(name: str, distribution: str, **parameters: object) -> Expression:
    """Return a value drawn from a distribution of `scipy.stats`.

    A new value is drawn at every step or renewal. The compiler checks the distribution name against `scipy.stats` and
    the keywords against the distribution's shape parameters, `loc`, and `scale`, and samples with the distribution's
    `rvs` and the simulation's random stream.

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
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import draw, draws_in
        >>> wait = draw('wait', 'pareto', b=var('alpha'), scale=2)
        >>> wait
        pareto_wait_b_scale(alpha, 2)
        >>> draws_in(wait)[0].parameters
        {'b': alpha, 'scale': 2}
    """
    if not isinstance(distribution, str):
        error_msg = f'The draw {name!r} takes the name of a scipy.stats distribution, but got {distribution!r}.'
        raise ValidationError(error_msg)
    return _call('draw', name, tuple(parameters.values()), distribution, tuple(parameters))


def draws_in(expression: object) -> list[DrawSpec]:
    """Return every draw of an expression with its arguments.

    Draws in the arguments of other draws are included.

    Args:
        expression:
            A symbolic expression, a number, a vector of expressions, or a sequence of them.

    Returns:
        draws:
            The draws, each call listed once, outer draws before the draws in their arguments.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import bernoulli, draws_in, uniform
        >>> x = var('x')
        >>> [item.name for item in draws_in(x * uniform('u') + bernoulli('flip', uniform('u')))]
        ['u', 'flip']
    """
    items = list(expression) if isinstance(expression, FreeModuleElement | list | tuple) else [expression]
    stack = [SR(item) for item in items if isinstance(item, Expression | int | float)]
    found: dict[str, DrawSpec] = {}
    while stack:
        item = stack.pop()
        operator = item.operator()
        if operator is None:
            continue
        name = getattr(operator, 'name', lambda: '')()
        if name in DRAWS:
            known = DRAWS[name]
            found.setdefault(
                str(item),
                DrawSpec(known.kind, known.name, known.distribution, known.keywords, tuple(item.operands()), item),
            )
        stack.extend(reversed(item.operands()))
    return list(found.values())
