"""Lower random draws and Markov laws to values drawn into an extended parameter vector or at renewals."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import difflib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR
from scipy import stats

from ..base import (
    Categorical,
    Domain,
    Expression,
    IncompatibleLawsError,
    SymbolicVector,
    ValidationError,
    components_of,
    symbol,
    symbols_of,
    vector,
)
from ..model import (
    Boundary,
    Call,
    Draw,
    Emission,
    Equation,
    Event,
    Jump,
    Law,
    Noise,
    Rate,
    Transition,
    Update,
    categorical,
    draws_in,
    where,
)
from ..model._laws import Renewal
from ._kernels import Reference, Source
from ._layout import Entry
from ._printing import numpy_function
from ._problem import Draws, MarkovMatrix, SampledDraw, Vector

ALPHABET_DOMAINS = (Domain.INTEGER, Domain.NONNEGATIVE_INTEGER, Domain.BOOLEAN)
NORMAL_KINDS = frozenset({'normal'})
DISTRIBUTIONS = tuple(
    sorted(
        name
        for name in dir(stats)
        if not name.startswith('_') and isinstance(getattr(stats, name), stats.rv_continuous | stats.rv_discrete)
    ),
)


def _is_symbolic(value: object) -> bool:
    """Check whether a law value is symbolic, as opposed to a call or a Python function."""
    return isinstance(value, Expression | SymbolicVector) or not (isinstance(value, Call) or callable(value))


def _law_values(law: Law) -> list[object]:
    """Return the values of a law that can hold random draws."""
    if isinstance(law, Rate | Noise | Boundary | Update):
        return [law.value]
    if isinstance(law, Jump):
        return [law.rate, *law.changes.values()]
    if isinstance(law, Equation):
        return [law.lhs, law.rhs]
    if isinstance(law, Event):
        return [law.guard, *(law.reset or {}).values()]
    if isinstance(law, Renewal):
        return [law.waiting, *(law.reset or {}).values()]
    return []


def distribution_of(owner: str, description: str, draw: Draw) -> Any:  # noqa: ANN401
    """Return the `scipy.stats` distribution of a draw, after checking its name and its keywords.

    Args:
        owner:
            The path of the component the draw belongs to.

        description:
            The description of the law that holds the draw.

        draw:
            The draw, of kind `'draw'`.

    Returns:
        distribution:
            The distribution of `scipy.stats`.

    Raises:
        ValidationError:
            SciPy has no distribution of that name, or the draw gives a keyword the distribution does not accept or
            leaves out one of its shape parameters.
    """
    name = str(draw.distribution)
    if name not in DISTRIBUTIONS:
        closest = ', '.join(difflib.get_close_matches(name, DISTRIBUTIONS, n=3)) or 'none'
        error_msg = (
            f'The {description} of {owner} draws {draw.name!r} from the distribution {name!r}, which scipy.stats does '
            f'not provide. The closest available names are: {closest}.'
        )
        raise ValidationError(error_msg)
    distribution = getattr(stats, name)
    shapes = [item.strip() for item in (distribution.shapes or '').split(',') if item.strip()]
    unknown = sorted(set(draw.keywords) - {*shapes, 'loc', 'scale'})
    missing = sorted(set(shapes) - set(draw.keywords))
    if unknown or missing:
        accepted = ', '.join([*shapes, 'loc', 'scale'])
        error_msg = (
            f'The {description} of {owner} draws {draw.name!r} from {name} with the keywords '
            f'{", ".join(draw.keywords) or "none"}, but {name} takes {accepted}, and needs its shape parameters '
            f'{", ".join(shapes) or "none"}.'
        )
        raise ValidationError(error_msg)
    return distribution


def sample_draw(draw: Draw, arguments: Sequence[Any], size: int, rng: np.random.Generator) -> Vector:
    """Return values of a draw for a number of members, with arguments that hold one value or one per member.

    Args:
        draw:
            The draw.

        arguments:
            The values of the draw's arguments, numbers or arrays with one value per member.

        size:
            The number of values.

        rng:
            The random number generator.

    Returns:
        values:
            One value per member.
    """
    values = [np.broadcast_to(np.asarray(item, dtype=float), (size,)) for item in arguments]
    if draw.kind == 'uniform':
        return values[0] + (values[1] - values[0]) * rng.random(size)
    if draw.kind == 'normal':
        return values[0] + values[1] * rng.standard_normal(size)
    if draw.kind == 'bernoulli':
        return (rng.random(size) < values[0]).astype(float)
    if draw.kind == 'categorical':
        weights = np.stack(values, axis=-1)
        cumulative = np.cumsum(weights, axis=-1)
        chosen = rng.random(size)[:, np.newaxis] * cumulative[:, -1:]
        return np.sum(chosen >= cumulative[:, :-1], axis=-1).astype(float)
    distribution = getattr(stats, str(draw.distribution))
    sampled = distribution.rvs(**dict(zip(draw.keywords, values, strict=True)), size=size, random_state=rng)
    return np.asarray(sampled, dtype=float)


def check_draw_placement(owner: str, laws: list[Law]) -> None:
    """Raise an error when a law other than an update or a renewal law holds a random draw, or a draw is unknown.

    Args:
        owner:
            The path of the component the laws belong to.

        laws:
            The laws of the component.

    Raises:
        IncompatibleLawsError:
            A rate, noise, jump, boundary, equation, or event law holds a draw.

        ValidationError:
            A draw names a distribution that `scipy.stats` does not provide, or keywords it does not take.
    """
    for law in laws:
        values = [value for value in _law_values(law) if _is_symbolic(value)]
        found = draws_in(values)
        if found and not isinstance(law, Update | Renewal):
            error_msg = (
                f'The {type(law).__name__} law of {owner} holds the random draw {found[0].call}, but only update and '
                'renewal laws take random draws. Model randomness in continuous time with Noise laws for Wiener '
                'processes, Jump laws for random events, or Renewal laws for events after drawn waiting times.'
            )
            raise IncompatibleLawsError(error_msg)
        for item in found:
            if item.kind == 'draw':
                distribution_of(owner, f'{type(law).__name__} law', item)


def _alphabet_size(description: str, reference: Reference | None) -> int | None:
    """Return the number of values of a state over a finite alphabet, or `None` for the integers."""
    if reference is None or not reference.is_state:
        error_msg = f'The {description} must target a scalar state of its component.'
        raise ValidationError(error_msg)
    domain = reference.entry.domain
    if isinstance(domain, Categorical):
        return len(domain.labels)
    if domain is Domain.BOOLEAN:
        return 2
    if domain in ALPHABET_DOMAINS:
        return None
    error_msg = (
        f'The {description} moves {reference.entry.name}, whose domain {domain} is not a finite alphabet. Declare the '
        'state with a categorical or an integer domain.'
    )
    raise ValidationError(error_msg)


def _check_shape(
    description: str,
    matrix: Sequence[Sequence[object]],
    rows: int | None,
    columns: int | None,
) -> None:
    """Raise an error when a matrix does not have the rows and columns its alphabets need."""
    shape = (len(matrix), len(matrix[0]))
    if (rows is None or shape[0] == rows) and (columns is None or shape[1] == columns):
        return
    needed = f'{rows if rows is not None else shape[0]} rows and {columns if columns is not None else shape[1]} columns'
    error_msg = (
        f'The matrix of the {description} has {shape[0]} rows and {shape[1]} columns, but it needs {needed}, one row '
        'per value of the selecting state and one column per value of the target.'
    )
    raise ValidationError(error_msg)


def _selected_row(selector: Expression, matrix: Sequence[Sequence[object]]) -> list[Expression]:
    """Return the row of a matrix that the value of a selecting expression picks, as nested conditions."""
    weights = []
    for column in range(len(matrix[0])):
        weight = SR(matrix[-1][column])
        for row in range(len(matrix) - 2, -1, -1):
            weight = where(selector == row, matrix[row][column], weight)
        weights.append(weight)
    return weights


def _draw_name(kind: str, target: Expression) -> str:
    """Return the draw name of a Markov law, which no user draw name starts with."""
    return f'_{kind}_{re.sub(r"[^0-9A-Za-z]", "_", str(target))}'


@dataclass(frozen=True)
class MatrixLaw:
    """Matrix of a Markov law of a component, with the description of the law.

    Args:
        owner:
            The path of the component.

        description:
            The description of the law.

        matrix:
            The rows of the matrix.
    """

    owner: str
    description: str
    matrix: Sequence[Sequence[object]]


def _transition(owner: str, law: Transition, references: dict[Expression, Reference]) -> tuple[Update, MatrixLaw]:
    """Return the update law a transition law becomes and its matrix."""
    description = f'transition law of {owner}.{law.target}'
    size = _alphabet_size(description, references.get(law.target))
    _check_shape(description, law.matrix, size, size)
    _check_shape(description, law.matrix, len(law.matrix), len(law.matrix))
    value = categorical(_draw_name('transition', law.target), _selected_row(law.target, law.matrix))
    return Update(law.target, value), MatrixLaw(owner, description, law.matrix)


def _emission(
    owner: str,
    law: Emission,
    references: dict[Expression, Reference],
    updates: dict[str, object],
) -> tuple[Update, MatrixLaw]:
    """Return the update law an emission law becomes, reading the hidden state's next value, and its matrix."""
    description = f'emission law of {owner}.{law.target}'
    columns = _alphabet_size(description, references.get(law.target))
    rows = _alphabet_size(description, references.get(law.source))
    _check_shape(description, law.matrix, rows, columns)
    hidden = updates.get(str(law.source))
    if hidden is None:
        error_msg = (
            f'The {description} reads the next value of {law.source}, which no update or transition law of {owner} '
            'sets. Give the hidden state a transition law in the same component.'
        )
        raise ValidationError(error_msg)
    value = categorical(_draw_name('emission', law.target), _selected_row(SR(hidden), law.matrix))
    return Update(law.target, value), MatrixLaw(owner, description, law.matrix)


def markov_laws(
    owner: str,
    laws: list[Law],
    references: dict[Expression, Reference],
) -> tuple[list[Law], list[MatrixLaw]]:
    """Return the laws of a component with transition and emission laws rewritten as update laws with draws.

    Args:
        owner:
            The path of the component.

        laws:
            The laws of the component.

        references:
            The references of the symbols the laws read.

    Returns:
        rewritten:
            The laws with every Markov law replaced by its update law, in the same order, and the matrices.

    Raises:
        ValidationError:
            A Markov law targets a state that is not over a finite alphabet, its matrix has the wrong shape, or an
            emission reads a hidden state that no update law of the component sets.
    """
    matrices: list[MatrixLaw] = []
    rewritten: list[Law] = []
    for law in laws:
        if isinstance(law, Transition):
            update, matrix = _transition(owner, law, references)
            rewritten.append(update)
            matrices.append(matrix)
        else:
            rewritten.append(law)
    updates = {str(law.target): law.value for law in rewritten if isinstance(law, Update)}
    for index, law in enumerate(rewritten):
        if isinstance(law, Emission):
            rewritten[index], matrix = _emission(owner, law, references, updates)
            matrices.append(matrix)
    return rewritten, matrices


def _matrix_reader(reference: Reference) -> Callable[[Vector], Any]:
    """Return the function that reads a parameter of a matrix for every evaluation row."""
    if reference.source is Source.MEMBER:
        members = reference.members
        return lambda p: p[members]
    index = reference.index
    return lambda p: p[index]


def compile_matrix(
    law: MatrixLaw,
    references: dict[Expression, Reference],
    rows: int,
    time: Expression,
) -> MarkovMatrix:
    """Return the matrix of a Markov law as a function of the parameter vector.

    Args:
        law:
            The matrix of the law.

        references:
            The references of the symbols of the component.

        rows:
            The number of evaluation rows, the members of a group.

        time:
            The time symbol.

    Returns:
        matrix:
            The matrix as a function of the parameter vector.

    Raises:
        ValidationError:
            An entry of the matrix reads a state, time, or a symbol of another component.
    """
    entries = [SR(entry) for row in law.matrix for entry in row]
    arguments = sorted(symbols_of(entries), key=str)
    for item in arguments:
        reference = references.get(item)
        if reference is None or reference.is_state:
            error_msg = (
                f'The matrix of the {law.description} reads {item}, but its entries must be numbers or expressions of '
                'the parameters of its component.'
            )
            raise ValidationError(error_msg)
    function = numpy_function(time, arguments, entries)
    readers = [_matrix_reader(references[item]) for item in arguments]
    shape = (len(law.matrix), len(law.matrix[0]))
    reads = frozenset(
        int(index)
        for item in arguments
        for index in (
            references[item].flat_indices if references[item].source is Source.MEMBER else [references[item].index]
        )
    )

    def evaluate(p: Vector) -> npt.NDArray[np.float64]:
        values = function(0.0, *(reader(p) for reader in readers))
        stacked = np.stack([np.broadcast_to(np.asarray(value, dtype=float), (rows,)) for value in values], axis=-1)
        return stacked.reshape(rows, *shape)

    return MarkovMatrix(law.description, evaluate, reads)


def _replacement(draw: Draw, arguments: list[Expression], slot: Expression) -> Expression:
    """Return the expression of a draw in terms of a uniform or standard normal value, or its sampled value."""
    if draw.kind == 'draw':
        return slot
    if draw.kind == 'uniform':
        low, high = arguments
        return low + (high - low) * slot
    if draw.kind == 'normal':
        mean, deviation = arguments
        return mean + deviation * slot
    if draw.kind == 'bernoulli':
        return where(slot < arguments[0], 1, 0)
    total = sum(arguments[1:], arguments[0])
    cumulative, index = SR(0), SR(0)
    for weight in arguments[:-1]:
        cumulative = cumulative + weight
        index = index + where(slot * total >= cumulative, 1, 0)
    return index


def _lowered(expression: Expression, slots: dict[str, Expression]) -> Expression:
    """Return an expression with every draw replaced by its expression of the draw's value, innermost draws first."""
    found = draws_in(expression)
    while found:
        draw = next(item for item in found if not draws_in(list(item.arguments)))
        expression = expression.subs({draw.call: _replacement(draw, list(draw.arguments), slots[draw.name])})
        found = draws_in(expression)
    return expression


@dataclass
class _Request:
    """Draw of a component that needs a value per evaluation row at every step."""

    owner: str
    name: str
    normal: bool
    rows: int
    slot: Expression
    references: dict[Expression, Reference]
    draw: Draw | None = None
    slots: dict[str, Expression] = field(default_factory=dict)


def _sampler(request: _Request, time: Expression) -> Callable[[float, Vector, Vector, np.random.Generator], Vector]:
    """Return the function that samples a draw from a `scipy.stats` distribution for every row of a request."""
    draw = cast('Draw', request.draw)
    if any(item.kind == 'draw' for item in draws_in(list(draw.arguments))):
        error_msg = (
            f'The draw {draw.name!r} of {request.owner} takes another draw from a scipy.stats distribution as an '
            'argument. Give its arguments as states, parameters, and uniform, normal, Bernoulli, or categorical draws.'
        )
        raise ValidationError(error_msg)
    arguments = [_lowered(SR(item), request.slots) for item in draw.arguments]
    symbols = sorted(symbols_of(arguments) - {time}, key=str)
    readers = [request.references.get(item) for item in symbols]
    if any(reader is None or reader.source not in {Source.SINGLE, Source.MEMBER} for reader in readers):
        error_msg = (
            f'The draw {draw.name!r} of {request.owner} reads a symbol that is not a state or a parameter of its '
            'component. Give the arguments of draws as expressions of the states and parameters of the member.'
        )
        raise ValidationError(error_msg)
    function = numpy_function(time, symbols, arguments)
    rows = request.rows

    def sample(t: float, x: Vector, q: Vector, rng: np.random.Generator) -> Vector:
        values = function(t, *(cast('Reference', reader).column_of(x, q) for reader in readers)) if arguments else []
        return sample_draw(draw, values, rows, rng)

    return sample


@dataclass
class DrawLowering:
    """Draws of every component, lowered to values in an extended parameter vector.

    Args:
        offset:
            The size of the parameter vector, where the draws start.

        time:
            The time symbol, which the arguments of draws from `scipy.stats` distributions may read.
    """

    offset: int
    time: Expression
    requests: list[_Request] = field(default_factory=list)
    draws: Draws | None = None

    def _slots(self, owner: str, rows: int, laws: list[Law], references: dict[Expression, Reference]) -> dict[str, Any]:
        """Return the value symbol of every draw of a component, after checking that each name has one declaration."""
        declared: dict[str, Draw] = {}
        for law in laws:
            if not isinstance(law, Update) or not _is_symbolic(law.value):
                continue
            for draw in draws_in(law.value):
                known = declared.setdefault(draw.name, draw)
                if (known.kind, known.distribution, known.parameters, known.arguments) != (
                    draw.kind,
                    draw.distribution,
                    draw.parameters,
                    draw.arguments,
                ):
                    error_msg = (
                        f'{owner} uses the draw name {draw.name!r} for {known.call} and for {draw.call}. Use one '
                        'declaration per name, or give the draws different names.'
                    )
                    raise ValidationError(error_msg)
        sanitized = re.sub(r'[^0-9A-Za-z]', '_', owner)
        slots: dict[str, Expression] = {}
        for name, draw in declared.items():
            slot = symbol(f'{draw.kind}_{name}_{sanitized}')
            slots[name] = slot
            sampled = draw if draw.kind == 'draw' else None
            request = _Request(owner, name, draw.kind in NORMAL_KINDS, rows, slot, references, sampled, slots)
            self.requests.append(request)
        return slots

    def lower(self, owner: str, rows: int, laws: list[Law], references: dict[Expression, Reference]) -> list[Law]:
        """Return the laws of a component with every draw of its update laws replaced by a drawn value.

        Args:
            owner:
                The path of the component.

            rows:
                The number of evaluation rows, one value of every draw per row.

            laws:
                The laws of the component.

            references:
                The references of the component's symbols, which receive the references of the drawn values.

        Returns:
            laws:
                The lowered laws.

        Raises:
            ValidationError:
                One draw name has two declarations in the component.
        """
        slots = self._slots(owner, rows, laws, references)
        if not slots:
            return laws
        lowered: list[Law] = []
        for law in laws:
            if not isinstance(law, Update) or not _is_symbolic(law.value):
                lowered.append(law)
                continue
            values = [_lowered(SR(item), slots) for item in components_of(law.value)]
            value = vector(values) if isinstance(law.value, SymbolicVector) else values[0]
            lowered.append(Update(law.target, value))
        return lowered

    def allocate(self) -> None:
        """Place the drawn values after the parameters, by component path, draw name, and member index.

        The draws taken at every step are kept in `draws`, which stays `None` when no update law draws.
        """
        if not self.requests:
            return
        start, flags = self.offset, []
        ordered = sorted(self.requests, key=lambda item: (item.draw is not None, item.owner, item.name))
        for request in ordered:
            grouped = request.rows > 1
            shape = (request.rows,) if grouped else ()
            members = request.rows if grouped else None
            entry = Entry(f'{request.owner}.{request.name}~draw', start, shape, Domain.REAL, members)
            source = Source.MEMBER if grouped else Source.SINGLE
            request.references[request.slot] = Reference(entry, False, 0, source)
            if request.draw is None:
                flags.extend([request.normal] * request.rows)
            start += request.rows
        sampled: list[SampledDraw] = []
        position = self.offset + len(flags)
        for request in ordered:
            if request.draw is not None:
                sampled.append(SampledDraw(position, request.rows, _sampler(request, self.time)))
                position += request.rows
        self.draws = Draws(self.offset, np.array(flags, dtype=bool), tuple(sampled))


def stochastic_update(update: Callable[..., Vector], draws: Draws) -> Callable[..., Vector]:
    """Return the update of a stochastic map, which draws its values from a generator before every step.

    Args:
        update:
            The update `F(t, x, q)` of the extended parameter vector.

        draws:
            The draws of the map.

    Returns:
        update:
            The update `F(t, x, p, rng)`.
    """

    def stochastic(t: float, x: Vector, p: Vector, rng: np.random.Generator) -> Vector:
        extended = draws.extend(p)
        draws.fill(extended, rng, t, x)
        return update(t, x, extended)

    return stochastic
