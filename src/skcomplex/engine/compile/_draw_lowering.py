"""Draws of every component, lowered to values in an extended parameter vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import difflib
import re
from collections.abc import Callable
from typing import Any, Self, cast

import numpy as np
from sage.all__sagemath_symbolics import RR, SR, Expression
from sage.modules.free_module_element import FreeModuleElement
from scipy import stats

from ...core.laws import BaseLaw, UpdateLaw
from ...core.laws.utils import draws_in, where
from ...errors import ValidationError
from ...specs import CallSpec, DrawSpec, EntrySpec, SampledDrawSpec
from ...types import Array, ReadKind
from .._utils import components_of, symbol, symbols_of, vector
from ..problems import DeclarationsSpec, ExpressionPrinter, KernelSpec, ReferenceSpec
from ..problems._draw_kernel import DrawKernel
from ..problems._request_spec import _RequestSpec
from ._base import BaseLowering

_NORMAL_KINDS = frozenset({'normal'})


DISTRIBUTIONS = tuple(
    sorted(
        name
        for name in dir(stats)
        if not name.startswith('_') and isinstance(getattr(stats, name), stats.rv_continuous | stats.rv_discrete)
    ),
)


class DrawLowering(BaseLowering):
    """Lowering of the draws of every component to values in an extended parameter vector.

    `prepare(compilation)` reads where the parameters end and the time symbol, and `run(laws)` takes the laws of every
    kernel spec, replaces every draw of their update laws by a drawn value, and places the drawn values after the
    parameters, by component path, draw name, and member index. `_apply_references()` then gives every spec the
    references of its drawn values.

    Attributes:
        offset_:
            The size of the parameter vector, where the draws start.

        time_:
            The time symbol, which the arguments of draws from `scipy.stats` distributions may read.

        requests_:
            The draws of every component that need a value per evaluation row at every step.

        draws_:
            The draws taken at every step, or `None` when no update law draws.
    """

    offset_: int
    time_: Expression
    requests_: list[_RequestSpec]
    draws_: DrawKernel | None

    def _prepare(self, context: object) -> Self:
        """Read where the parameters end and the time symbol of the compilation, with no draws yet.

        Args:
            context:
                The compilation.

        Returns:
            self:
                The prepared lowering.
        """
        super()._prepare(context)
        declarations = cast('DeclarationsSpec', self.compilation_.layout)
        self.offset_, self.time_ = declarations.ends[False], declarations.time
        self.requests_, self.draws_ = [], None
        return self

    def _run(self, inputs: object) -> list[tuple[KernelSpec, list[BaseLaw]]]:
        """Return the laws of every kernel spec with the draws of their update laws replaced by drawn values.

        Args:
            inputs:
                The laws of every kernel spec.

        Returns:
            laws:
                The lowered laws of every kernel spec.

        Raises:
            ValidationError:
                One draw name has two declarations in a component.
        """
        lowered = [
            (spec, self._lower(spec, laws)) for spec, laws in cast('list[tuple[KernelSpec, list[BaseLaw]]]', inputs)
        ]
        self._allocate()
        return lowered

    def _slots(self, spec: KernelSpec, laws: list[BaseLaw]) -> dict[str, Any]:
        """Return the value symbol of every draw of a component, after checking that each name has one declaration."""
        owner = spec.owner
        declared: dict[str, DrawSpec] = {}
        for law in laws:
            if not isinstance(law, UpdateLaw) or not self._is_symbolic(law.value):
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
            request = _RequestSpec(owner, name, draw.kind in _NORMAL_KINDS, spec.rows, slot, spec, sampled, slots)
            self.requests_.append(request)
        return slots

    def _lower(self, spec: KernelSpec, laws: list[BaseLaw]) -> list[BaseLaw]:
        """Return the laws of a component with every draw of its update laws replaced by a drawn value.

        Args:
            spec:
                The kernel spec of the component, with one value of every draw per evaluation row, which receives
                the references of the drawn values when the lowering applies them.

            laws:
                The laws of the component.

        Returns:
            laws:
                The lowered laws.

        Raises:
            ValidationError:
                One draw name has two declarations in the component.
        """
        slots = self._slots(spec, laws)
        if not slots:
            return laws
        lowered: list[BaseLaw] = []
        for law in laws:
            if not isinstance(law, UpdateLaw) or not self._is_symbolic(law.value):
                lowered.append(law)
                continue
            values = [self._lowered(SR(item), slots) for item in components_of(law.value)]
            value = vector(values) if isinstance(law.value, FreeModuleElement) else values[0]
            lowered.append(UpdateLaw(law._get_targets()[0], value))
        return lowered

    def _allocate(self) -> None:
        """Place the drawn values after the parameters, by component path, draw name, and member index.

        The reference of every drawn value is noted for the spec of its component, and the draws taken at every step
        are kept in `draws_`, which stays `None` when no update law draws.
        """
        if not self.requests_:
            return
        start, flags = self.offset_, []
        ordered = sorted(self.requests_, key=lambda item: (item.draw is not None, item.owner, item.name))
        for request in ordered:
            grouped = request.rows > 1
            shape = (request.rows,) if grouped else ()
            members = request.rows if grouped else None
            entry = EntrySpec(f'{request.owner}.{request.name}~draw', start, shape, RR, members)
            source = ReadKind.MEMBER if grouped else ReadKind.SINGLE
            self._add_reference(request.spec, request.slot, ReferenceSpec(entry, False, 0, source))
            if request.draw is None:
                flags.extend([request.normal] * request.rows)
            start += request.rows
        sampled: list[SampledDrawSpec] = []
        position = self.offset_ + len(flags)
        for request in ordered:
            if request.draw is not None:
                sampled.append(SampledDrawSpec(position, request.rows, self._sampler(request)))
                position += request.rows
        self.draws_ = DrawKernel(self.offset_, np.array(flags, dtype=bool), tuple(sampled)).check()

    @staticmethod
    def _replacement(draw: DrawSpec, arguments: list[Expression], slot: Expression) -> Expression:
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

    @staticmethod
    def _lowered(expression: Expression, slots: dict[str, Expression]) -> Expression:
        """Return an expression with every draw replaced by its expression of its value, innermost draws first."""
        found = draws_in(expression)
        while found:
            draw = next(item for item in found if not draws_in(list(item.arguments)))
            expression = expression.subs(
                {draw.call: DrawLowering._replacement(draw, list(draw.arguments), slots[draw.name])},
            )
            found = draws_in(expression)
        return expression

    def _sampler(self, request: _RequestSpec) -> Callable[[float, Array, Array, np.random.Generator], Array]:
        """Return the function that samples a draw from a `scipy.stats` distribution for every row of a request."""
        time = self.time_
        draw = cast('DrawSpec', request.draw)
        if any(item.kind == 'draw' for item in draws_in(list(draw.arguments))):
            error_msg = (
                f'The draw {draw.name!r} of {request.owner} takes another draw from a scipy.stats distribution as an '
                'argument. Give its arguments as states, parameters, and uniform, normal, Bernoulli, or categorical '
                'draws.'
            )
            raise ValidationError(error_msg)
        arguments = [DrawLowering._lowered(SR(item), request.slots) for item in draw.arguments]
        symbols = sorted(symbols_of(arguments) - {time}, key=str)
        readers = [self._reference_of(item, request.spec) for item in symbols]
        if any(reader is None or reader.source not in {ReadKind.SINGLE, ReadKind.MEMBER} for reader in readers):
            error_msg = (
                f'The draw {draw.name!r} of {request.owner} reads a symbol that is not a state or a parameter of its '
                'component. Give the arguments of draws as expressions of the states and parameters of the member.'
            )
            raise ValidationError(error_msg)
        function = ExpressionPrinter(time=time, compiled=True).check().prepare(symbols).run(arguments)
        rows = request.rows

        def sample(t: float, x: Array, q: Array, rng: np.random.Generator) -> Array:
            values = (
                function(t, *(cast('ReferenceSpec', reader).column_of(x, q) for reader in readers)) if arguments else []
            )
            return DrawKernel._sample_draw(draw, values, rows, rng)

        return sample

    @staticmethod
    def _is_symbolic(value: object) -> bool:
        """Check whether a law value is symbolic, as opposed to a call or a Python function."""
        return isinstance(value, Expression | FreeModuleElement) or not (isinstance(value, CallSpec) or callable(value))

    @staticmethod
    def _distribution_of(owner: str, description: str, draw: DrawSpec) -> Any:  # noqa: ANN401
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
                f'The {description} of {owner} draws {draw.name!r} from the distribution {name!r}, which scipy.stats '
                f'does not provide. The closest available names are: {closest}.'
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
