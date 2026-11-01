"""Draws of every component, lowered to values in an extended parameter vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import difflib
import re
from collections.abc import Callable, Mapping

import numpy as np
from sage.all__sagemath_symbolics import RR, SR, Expression
from sage.modules.free_module_element import FreeModuleElement
from scipy import stats

from ...core.laws import BaseLaw, UpdateLaw
from ...core.laws.utils import draws_in, where
from ...errors import ValidationError
from ...specs import CallSpec, DrawSpec, EntrySpec, SampledDrawSpec
from ...types import Array
from .._utils import components_of, symbol, symbols_of, vector
from ..problems import ExpressionPrinter, KernelSpec, ReferenceSpec
from ..problems._draw_kernel import DrawKernel
from ..problems._request_spec import _RequestSpec
from ..types import ReadKind
from ._base import BaseLowering, BasePass
from ._compilation import Compilation
from ._members import MemberLowering

_NORMAL_KINDS = frozenset({'normal'})
_SHIFT = 'shift_member'


DISTRIBUTIONS = tuple(
    sorted(
        name
        for name in dir(stats)
        if not name.startswith('_') and isinstance(getattr(stats, name), stats.rv_continuous | stats.rv_discrete)
    ),
)


class DrawLowering(BaseLowering[list[tuple[KernelSpec, list[BaseLaw]]]]):
    """Lowering of the draws of every component to values in an extended parameter vector.

    `prepare(compilation)` reads where the parameters end and the time symbol, and `run(laws)` takes the laws of every
    kernel spec, replaces every draw of their update laws by a drawn value, and places the drawn values after the
    parameters, by component path, draw name, and member index. A draw read at a member offset reads the value the
    offset member draws, so the members of a group can share one draw. `_apply_references()` then gives every spec the
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

        shifted_:
            The symbol of every draw read at a member offset, by the spec, the draw name, the offset, the boundary,
            and the fill of the read.
    """

    offset_: int
    time_: Expression
    requests_: list[_RequestSpec]
    draws_: DrawKernel | None
    shifted_: dict[tuple[int, str, int, str, float], Expression]

    def _prepare(self, context: Compilation | None) -> None:
        """Read where the parameters end and the time symbol of the compilation, with no draws yet.

        Args:
            context:
                The compilation.

        Raises:
            ValidationError:
                The compilation has no layout yet.
        """
        super()._prepare(context)
        declarations = self.compilation_.layout
        if declarations is None:
            error_msg = 'The compilation has no layout yet. Lower draws after the layout pass.'
            raise ValidationError(error_msg)
        self.offset_, self.time_ = declarations.ends[False], declarations.time
        self.requests_, self.draws_, self.shifted_ = [], None, {}

    def _run(self, inputs: list[tuple[KernelSpec, list[BaseLaw]]] | None) -> None:
        """Set the laws of every kernel spec with the draws of their update laws replaced by drawn values.

        Args:
            inputs:
                The laws of every kernel spec.

        Raises:
            ValidationError:
                One draw name has two declarations in a component.
        """
        self.lowered_ = [(spec, self._lower(spec, laws)) for spec, laws in inputs or []]
        self._allocate()

    def _slots(self, spec: KernelSpec, laws: list[BaseLaw]) -> dict[str, Expression]:
        """Return the value symbol of every draw of a component, after checking that each name has one declaration.

        A draw read at a member offset is the draw of another member, so its arguments may read that member.
        """
        owner = spec.owner
        declared: dict[str, DrawSpec] = {}
        for law in laws:
            if not isinstance(law, UpdateLaw) or not self._is_symbolic(law.value):
                continue
            items = [SR(item) for item in (*components_of(law.value), *([] if law.order is None else [law.order]))]
            shifted = [call for item in items for call in self._shifted_draws(item)]
            outside = [item.subs(dict.fromkeys(shifted, 0)) for item in items]
            for draw in draws_in(outside):
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
            for call in shifted:
                draw = draws_in(call.operands()[0])[0]
                declared.setdefault(draw.name, draw)
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
            values = [self._lowered(self._unshifted(spec, SR(item), slots), slots) for item in components_of(law.value)]
            value = vector(values) if isinstance(law.value, FreeModuleElement) else values[0]
            order = None if law.order is None else self._lowered(SR(law.order), slots)
            lowered.append(BasePass._replaced(law, value=value, order=order))
        return lowered

    @staticmethod
    def _shifted_draws(expression: Expression) -> list[Expression]:
        """Return the member offset reads of draws in an expression."""
        return [call for call in BasePass._calls_of(expression, _SHIFT) if draws_in(call.operands()[0])]

    def _unshifted(self, spec: KernelSpec, expression: Expression, slots: Mapping[str, Expression]) -> Expression:
        """Return an expression with every draw read at a member offset replaced by the value the offset member draws.

        Args:
            spec:
                The kernel spec of the component.

            expression:
                The expression.

            slots:
                The value symbol of every draw of the component.

        Returns:
            expression:
                The expression with the offset reads of draws replaced.
        """
        replacements = {}
        for call in self._shifted_draws(expression):
            drawn, offset, boundary, fill = MemberLowering._shift_arguments(call)
            draw = draws_in(drawn)[0]
            key = (id(spec), draw.name, offset, boundary, fill)
            slot = self.shifted_.setdefault(key, symbol(f'{slots[draw.name]}__shift{len(self.shifted_)}'))
            replacements[call] = self._replacement(draw, list(draw.arguments), slot)
        return expression.subs(replacements) if replacements else expression

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
            for (spec, name, offset, boundary, fill), slot in self.shifted_.items():
                if spec == id(request.spec) and name == request.name:
                    read = MemberLowering._neighbor_read(request.rows, offset, boundary, fill)
                    self._add_reference(request.spec, slot, ReferenceSpec(entry, False, 0, ReadKind.NEIGHBOR, read))
            if request.draw is None:
                flags.extend([request.normal] * request.rows)
            start += request.rows
        sampled: list[SampledDrawSpec] = []
        position = self.offset_ + len(flags)
        for request in ordered:
            if request.draw is not None:
                sampled.append(SampledDrawSpec(position, request.rows, self._sampler(request)))
                position += request.rows
        self.draws_ = DrawKernel(self.offset_, np.array(flags, dtype=bool), tuple(sampled)).check().prepare()

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
    def _lowered(expression: Expression, slots: Mapping[str, Expression]) -> Expression:
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
        draw = request.draw
        if draw is None:
            error_msg = f'The draw {request.name!r} of {request.owner} takes no distribution to sample from.'
            raise ValidationError(error_msg)
        if any(item.kind == 'draw' for item in draws_in(list(draw.arguments))):
            error_msg = (
                f'The draw {draw.name!r} of {request.owner} takes another draw from a scipy.stats distribution as an '
                'argument. Give its arguments as states, parameters, and uniform, normal, Bernoulli, or categorical '
                'draws.'
            )
            raise ValidationError(error_msg)
        arguments = [DrawLowering._lowered(SR(item), request.slots) for item in draw.arguments]
        symbols = sorted(symbols_of(arguments) - {time}, key=str)
        found = [self._reference_of(item, request.spec) for item in symbols]
        readers = [
            reader for reader in found if reader is not None and reader.source in {ReadKind.SINGLE, ReadKind.MEMBER}
        ]
        if len(readers) < len(found):
            error_msg = (
                f'The draw {draw.name!r} of {request.owner} reads a symbol that is not a state or a parameter of its '
                'component. Give the arguments of draws as expressions of the states and parameters of the member.'
            )
            raise ValidationError(error_msg)
        function = ExpressionPrinter(time=time, compiled=True).check().prepare(symbols).run(arguments).code_
        if not callable(function):
            error_msg = f'A compiled printer returns a function, not a {type(function).__name__}.'
            raise TypeError(error_msg)
        rows = request.rows

        def sample(t: float, x: Array, q: Array, rng: np.random.Generator) -> Array:
            values = function(t, *(reader.column_of(x, q) for reader in readers)) if arguments else []
            return DrawKernel._sample_draw(draw, values, rows, rng)

        return sample

    @staticmethod
    def _is_symbolic(value: object) -> bool:
        """Check whether a law value is symbolic, as opposed to a call or a Python function."""
        return isinstance(value, Expression | FreeModuleElement) or not (isinstance(value, CallSpec) or callable(value))

    @staticmethod
    def _distribution_of(owner: str, description: str, draw: DrawSpec) -> stats.rv_continuous | stats.rv_discrete:
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
