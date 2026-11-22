"""Define random draws, values drawn from distributions at every step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Sequence
from dataclasses import replace

from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....errors import ValidationError
from .._draw_spec import DrawSpec
from ._operators import _OperatorFunction

DRAWS: dict[str, DrawSpec] = {}
DRAW_FUNCTIONS: dict[str, _OperatorFunction] = {}


class _DrawFunction(_OperatorFunction):
    """Value drawn at every step under a label, with a derivative of 0."""

    def _derivative_(self, *arguments: Expression, diff_param: object = None) -> Expression:
        """Return 0, since a draw does not change with what it is differentiated by."""
        del arguments, diff_param
        return SR(0)


def _call(
    kind: str,
    name: str,
    arguments: Sequence[object],
    distribution: str | None = None,
    keywords: tuple[str, ...] = (),
    *,
    temperature: float | None = None,
    hard: bool = False,
) -> Expression:
    """Return the symbolic call of a draw of a kind under a name.

    The name and the relaxation are checked first. A relaxed draw has its own label, which names its temperature.
    """
    if not isinstance(name, str) or not name.isidentifier():
        error_msg = f'The name of a {kind} draw must be a valid Python name, such as `turn`, but it is {name!r}.'
        raise ValidationError(error_msg)
    spec = DrawSpec(kind, name, distribution, keywords, temperature=temperature, hard=hard)
    spec._check()
    label = f'{distribution or kind}_{name}{"".join(f"_{keyword}" for keyword in keywords)}'
    if temperature is not None:
        label += '_relaxed' + re.sub(r'\W', '_', repr(float(temperature))) + ('_hard' if hard else '')
    DRAWS[label] = spec
    if label not in DRAW_FUNCTIONS:
        DRAW_FUNCTIONS[label] = _DrawFunction(label)
    return DRAW_FUNCTIONS[label](*(SR(argument) for argument in arguments))


def uniform(name: str, low: object = 0, high: object = 1) -> Expression:
    """Return a value drawn uniformly between two bounds at every step.

    A new value between `low` and `high` is drawn at every step, every use of the same name in one element reads the
    same value within a step, and every member of a group draws its own, so `uniform('u', 0, x) + uniform('u', 0, x)` is
    twice one draw. It gives the random heading of a walker, the noise of a sensor, and the time a customer waits.

    Read more in the user guide, under Laws.

    Args:
        name:
            The name of the draw, a valid Python name.

        low:
            The lower bound, a number or an expression of states and parameters.

        high:
            The upper bound.

    Returns:
        draw:
            The symbolic draw.

    Raises:
        ValidationError:
            The name is not a valid Python name.

    See Also:
        normal: Draws from a normal distribution.
        draw: Draws from any distribution of `scipy.stats`.

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

    A new value of mean `mean` and standard deviation `deviation` is drawn at every step, shared by every use of the
    same name in one element and drawn apart for every member of a group, such as `x + normal('step', 0, 1)` for a
    random walk. It gives the shocks to a price, the errors of a measurement, and the jitter of a firing neuron.

    Read more in the user guide, under Laws.

    Args:
        name:
            The name of the draw, a valid Python name.

        mean:
            The mean, a number or an expression of states and parameters.

        deviation:
            The standard deviation.

    Returns:
        draw:
            The symbolic draw.

    Raises:
        ValidationError:
            The name is not a valid Python name.

    See Also:
        uniform: Draws uniformly between two bounds.
        draw: Draws from any distribution of `scipy.stats`.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import normal
        >>> var('x') + normal('step', 0, 1)
        x + normal_step(0, 1)
    """
    return _call('normal', name, (mean, deviation))


def bernoulli(name: str, p: object, temperature: float | None = None, hard: bool = False) -> Expression:
    """Return 1 with a probability and 0 otherwise, drawn at every step.

    A new value is drawn at every step, 1 with probability `p`, such as `1 - bernoulli('flip', q)` that is 0 with
    probability `q`. With a temperature `T`, the JAX backend gives a smooth value close to the draw,
    `sigmoid((logit p - logit u) / T)` for the uniform value `u` of the step (Maddison et al. 2017), which tends to the
    exact draw as `T` tends to 0, so runs have gradients in `p`. Every other backend draws exactly. It decides whether a
    contact infects, whether a trader buys, and whether a spin flips.

    Read more in the user guide, under Laws.

    Args:
        name:
            The name of the draw, a valid Python name.

        p:
            The probability of 1, a number or an expression of states and parameters.

        temperature:
            The positive temperature of the smooth value on JAX, or `None` for the exact draw.

        hard:
            Whether the JAX value is the exact draw with the gradient of the smooth value.

    Returns:
        draw:
            The symbolic draw.

    Raises:
        ValidationError:
            The name is not a valid Python name, the temperature is not positive, or `hard` is given without a
            temperature.

    See Also:
        categorical: Draws one of several categories.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import bernoulli
        >>> 1 - bernoulli('flip', var('q'))
        -bernoulli_flip(q) + 1
    """
    return _call('bernoulli', name, (p,), temperature=temperature, hard=hard)


def categorical(
    name: str,
    weights: Sequence[object],
    temperature: float | None = None,
    hard: bool = False,
) -> Expression:
    """Return the index of a category drawn at every step.

    A new index is drawn at every step, each category with a probability proportional to its weight, such as
    `categorical('map', [1, 1, 2])` that gives 2 half the time. The index is the number of running totals `c_k` of the
    weights below `u W`, for the uniform value `u` of the step and the total weight `W`. With a temperature `T`, the JAX
    backend gives the smooth value `Σ_k sigmoid((u - c_k / W) / T)`, so runs have gradients in the weights, and every
    other backend draws exactly. It picks the action of an agent, the strategy of a player, and the next word of a text.

    Read more in the user guide, under Laws.

    Args:
        name:
            The name of the draw, a valid Python name.

        weights:
            The nonnegative weights of the categories `0` to `len(weights) - 1`, numbers or expressions of states and
            parameters.

        temperature:
            The positive temperature of the smooth value on JAX, or `None` for the exact draw.

        hard:
            Whether the JAX value is the exact draw with the gradient of the smooth value.

    Returns:
        draw:
            The symbolic draw.

    Raises:
        ValidationError:
            The name is not a valid Python name, there are no weights, the temperature is not positive, or `hard` is
            given without a temperature.

    See Also:
        bernoulli: Draws 1 with a probability and 0 otherwise.

    Examples:
        >>> from skcomplex.core.components.laws.utils import categorical
        >>> categorical('map', [1, 1, 2])
        categorical_map(1, 1, 2)
    """
    if isinstance(weights, str) or not len(weights):
        error_msg = f'The categorical draw {name!r} needs a list of at least one weight, but got {weights!r}.'
        raise ValidationError(error_msg)
    return _call('categorical', name, tuple(weights), temperature=temperature, hard=hard)


def draw(name: str, distribution: str, **parameters: object) -> Expression:
    """Return a value drawn from a distribution of `scipy.stats`.

    A new value is drawn at every step or renewal, such as `draw('wait', 'pareto', b=alpha, scale=2)` for a waiting time
    with a heavy tail. The compiler checks that `scipy.stats` has the distribution and that the keywords are its shape
    parameters, `loc`, or `scale`, and it samples with the `rvs` method of the distribution and the random numbers of
    the run. It gives the sizes of firms, the waiting times between trades, and the jumps of a foraging animal.

    Read more in the user guide, under Laws.

    Args:
        name:
            The name of the draw, a valid Python name.

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
            The name is not a valid Python name, or the distribution is not given by its name.

    See Also:
        uniform: Draws uniformly between two bounds.
        normal: Draws from a normal distribution.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import draw
        >>> draw('wait', 'pareto', b=var('alpha'), scale=2)
        pareto_wait_b_scale(alpha, 2)
    """
    if not isinstance(distribution, str):
        error_msg = (
            f"The draw {name!r} takes the name of a distribution of scipy.stats, such as 'pareto', not "
            f'{distribution!r}.'
        )
        raise ValidationError(error_msg)
    return _call('draw', name, tuple(parameters.values()), distribution, tuple(parameters))


def draws_in(expression: object) -> list[DrawSpec]:
    """Return every random draw of an expression with its arguments.

    The compiler reads them to draw the values of every step, such as the draws `u` and `flip` in
    `x * uniform('u') + bernoulli('flip', uniform('u'))`, each call listed once and outer draws before the draws in
    their arguments. It finds the random headings of walkers, the shocks to prices, and the flips of spins.

    Read more in the user guide, under Extend the library.

    Args:
        expression:
            A symbolic expression, a number, a vector of expressions, or a sequence of them.

    Returns:
        draws:
            The draws, each call listed once, outer draws before the draws in their arguments.

    See Also:
        draw: Builds a draw from a distribution of `scipy.stats`.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import bernoulli, uniform
        >>> from skcomplex.core.components.laws.utils._draws import draws_in
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
                replace(known, arguments=tuple(item.operands()), call=item),
            )
        stack.extend(reversed(item.operands()))
    return list(found.values())
