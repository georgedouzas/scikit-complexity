"""Draws of every component, lowered to values in an extended parameter vector."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import difflib
import re
from collections.abc import Callable, Mapping

import numpy as np
from sage.all__sagemath_symbolics import RR, SR, Expression, log
from sage.modules.free_module_element import FreeModuleElement
from scipy import stats

from ...components.laws import BaseLaw, UpdateLaw
from ...components.laws.utils import clip, logistic, stop_gradient, where
from ...components.laws.utils._draws import draws_in
from ...components.laws.utils._reductions import reductions_in
from ...errors import ValidationError
from ...specs import CallSpec, DrawSpec, EntrySpec, NamedDrawSpec, NeighborReadSpec, SampledDrawSpec
from ...types import Array, Indices
from .._utils import components_of, symbol, symbols_of, vector
from ..problems._draw_kernel import DrawKernel
from ..problems._expression_printer import ExpressionPrinter
from ..problems._kernel_spec import KernelSpec
from ..problems._reference_spec import ReferenceSpec
from ..problems._request_spec import _RequestSpec
from ..types import ReadKind
from ._base import BaseLowering, BasePass
from ._compilation import Compilation
from ._members import MemberLowering

NORMAL_KINDS = frozenset({'normal'})
SHIFT = 'shift_member'
RELAXATION_FLOOR = 1e-12
DISTRIBUTIONS = tuple(
    sorted(
        name
        for name in dir(stats)
        if not name.startswith('_') and isinstance(getattr(stats, name), stats.rv_continuous | stats.rv_discrete)
    ),
)


class DrawLowering(BaseLowering[list[tuple[KernelSpec, list[BaseLaw]]], list[tuple[KernelSpec, list[BaseLaw]]]]):
    """A lowering of the draws of every component to values in a parameter vector.

    The parameter vector is extended with the drawn values. `prepare(compilation)` reads where the parameters end and
    the time symbol, and `run(laws)` takes the laws of every kernel spec, replaces every draw of their update laws by a
    drawn value, and places the drawn values after the parameters, by component path, draw name, and member index. A
    draw read at a member offset reads the value the offset member draws, so the members of a group can share one draw.
    `_apply_references()` then gives every spec the references of its drawn values. On the JAX backend a Bernoulli or
    categorical draw with a temperature is lowered to its continuous relaxation of the same uniform value, and with
    `hard` to the exact value with the gradient of the relaxation, while every other backend draws it exactly.

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

        relaxes_:
            Whether relaxed draws are lowered to their relaxation, true on the JAX backend.
    """

    offset_: int
    time_: Expression
    requests_: list[_RequestSpec]
    draws_: DrawKernel | None
    shifted_: dict[tuple[int, str, int, str, float], Expression]
    relaxes_: bool

    def _prepare(self, context: Compilation | None) -> None:
        """Read where the parameters end and the time symbol of the compilation.

        No draws are known yet.

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
        backend = self.compilation_.backend
        self.relaxes_ = backend is not None and backend._get_name() == 'jax'

    def _run(self, inputs: list[tuple[KernelSpec, list[BaseLaw]]] | None) -> None:
        """Set the laws of every kernel spec with the draws of their updates replaced.

        Every draw of their update laws is replaced by its drawn value.

        Args:
            inputs:
                The laws of every kernel spec.

        Raises:
            ValidationError:
                One draw name has two declarations in a component.
        """
        self.lowered_ = [(spec, self._lower(spec, laws)) for spec, laws in inputs or []]
        self._allocate(DrawLowering._get_orders_of([spec for spec, _ in self.lowered_]))

    def _get_slots(self, spec: KernelSpec, laws: list[BaseLaw]) -> dict[str, Expression]:
        """Return the value symbol of every draw of a component.

        It first checks that each name has one declaration. A draw read at a member offset is the draw of another
        member, so its arguments may read that member.
        """
        owner = spec.owner
        declared: dict[str, DrawSpec] = {}
        for law in laws:
            if not isinstance(law, UpdateLaw) or not self._is_symbolic(law.value):
                continue
            items = [SR(item) for item in (*components_of(law.value), *([] if law.order is None else [law.order]))]
            shifted = [call for item in items for call in self._get_shifted_draws(item)]
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
            request = _RequestSpec(
                owner,
                name,
                draw.kind in NORMAL_KINDS,
                spec.rows,
                slot,
                spec,
                sampled,
                slots,
                draw,
            )
            self.requests_.append(request)
        return slots

    def _lower(self, spec: KernelSpec, laws: list[BaseLaw]) -> list[BaseLaw]:
        """Return the laws of a component with its draws replaced by drawn values.

        Every draw of its update laws is replaced.

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
        slots = self._get_slots(spec, laws)
        if not slots:
            return laws
        lowered: list[BaseLaw] = []
        for law in laws:
            if not isinstance(law, UpdateLaw) or not self._is_symbolic(law.value):
                lowered.append(law)
                continue
            relaxes = self.relaxes_
            values = [
                self._get_lowered(self._get_unshifted(spec, SR(item), slots), slots, relaxes=relaxes)
                for item in components_of(law.value)
            ]
            value = vector(values) if isinstance(law.value, FreeModuleElement) else values[0]
            order = None if law.order is None else self._get_lowered(SR(law.order), slots, relaxes=relaxes)
            lowered.append(BasePass._get_replaced(law, value=value, order=order))
        return lowered

    @staticmethod
    def _get_shifted_draws(expression: Expression) -> list[Expression]:
        """Return the member offset reads of draws in an expression."""
        return [call for call in BasePass._get_calls_of(expression, SHIFT) if draws_in(call.operands()[0])]

    def _get_unshifted(self, spec: KernelSpec, expression: Expression, slots: Mapping[str, Expression]) -> Expression:
        """Return an expression with every draw at a member offset replaced.

        The replacement is the value the offset member draws.

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
        for call in self._get_shifted_draws(expression):
            drawn, offset, boundary, fill = MemberLowering._shift_arguments(call)
            draw = draws_in(drawn)[0]
            key = (id(spec), draw.name, offset, boundary, fill)
            slot = self.shifted_.setdefault(key, symbol(f'{slots[draw.name]}__shift{len(self.shifted_)}'))
            replacements[call] = self._get_replacement(draw, list(draw.arguments), slot, relaxes=self.relaxes_)
        return expression.subs(replacements) if replacements else expression

    @staticmethod
    def _get_orders_of(specs: list[KernelSpec]) -> tuple[Indices, ...]:
        """Return the distinct orders drawn at every step.

        The gather tables of kernel specs read through these orders.
        """
        orders: dict[int, Indices] = {}
        for spec in specs:
            for reference in spec.references.values():
                read = reference.neighbor
                if isinstance(read, NeighborReadSpec) and read.order is not None:
                    orders.setdefault(id(read.order), read.order)
        return tuple(orders.values())

    def _allocate(self, orders: tuple[Indices, ...]) -> None:
        """Place the drawn values after the parameters.

        They are placed by component path, draw name, and member index. The reference of every drawn value is noted for
        the spec of its component, and the draws taken at every step are kept in `draws_` with the orders drawn at every
        step, which stays `None` when no update law draws and no table reads through an order.
        """
        if not self.requests_ and not orders:
            return
        start, flags = self.offset_, []
        ordered = sorted(self.requests_, key=lambda item: (item.draw is not None, item.owner, item.name))
        names: dict[str, int] = {}
        for request in ordered:
            grouped = request.rows > 1
            shape = (request.rows,) if grouped else ()
            members = request.rows if grouped else None
            entry = EntrySpec(self._get_entry_name(request, names), start, shape, RR, members)
            source = ReadKind.MEMBER if grouped else ReadKind.SINGLE
            self._add_reference(request.spec, request.slot, ReferenceSpec(entry, False, 0, source))
            for (spec, name, offset, boundary, fill), slot in self.shifted_.items():
                if spec == id(request.spec) and name == request.name:
                    read = MemberLowering._get_neighbor_read(request.rows, offset, boundary, fill)
                    self._add_reference(request.spec, slot, ReferenceSpec(entry, False, 0, ReadKind.NEIGHBOR, read))
            if request.draw is None:
                flags.extend([request.normal] * request.rows)
            start += request.rows
        sampled: list[SampledDrawSpec] = []
        named: list[NamedDrawSpec] = []
        position = self.offset_
        for request in ordered:
            if request.draw is None:
                named.append(self._get_named(request, position))
                position += request.rows
        for request in ordered:
            if request.draw is not None:
                sampled.append(SampledDrawSpec(position, request.rows, self._get_sampler(request)))
                named.append(self._get_named(request, position))
                position += request.rows
        self.draws_ = DrawKernel(
            self.offset_,
            np.array(flags, dtype=bool),
            tuple(sampled),
            orders,
            tuple(named),
        ).prepare()

    def _get_named(self, request: _RequestSpec, start: int) -> NamedDrawSpec:
        """Return a draw of a request as runs record and replay it.

        Its arguments are evaluated from the states and parameters of the member.
        """
        draw = request.declared or request.draw
        if draw is None:
            error_msg = f'The draw {request.name!r} of {request.owner} declares no distribution.'
            raise ValidationError(error_msg)
        nested = draws_in(list(draw.arguments))
        arguments = None if nested else self._get_arguments_of(request, draw)
        return NamedDrawSpec(f'{request.owner}.{request.name}', draw, start, request.rows, arguments)

    def _get_arguments_of(
        self,
        request: _RequestSpec,
        draw: DrawSpec,
    ) -> Callable[[float, Array, Array], list[Array]] | None:
        """Return the function of the values of the arguments of a draw.

        It is `None` when they read a symbol that is not a state or a parameter of the member, or reduce over other
        members, such as the mean of a group.
        """
        time = self.time_
        arguments = [DrawLowering._get_lowered(SR(item), request.slots) for item in draw.arguments]
        if any(reductions_in(item) for item in arguments):
            return None
        symbols = sorted(symbols_of(arguments) - {time}, key=str)
        found = [self._get_reference_of(item, request.spec) for item in symbols]
        readers = [
            reader for reader in found if reader is not None and reader.source in {ReadKind.SINGLE, ReadKind.MEMBER}
        ]
        if len(readers) < len(found):
            return None
        if not arguments:
            return lambda t, x, q: []
        function = ExpressionPrinter(time=time, compiled=True).prepare(symbols).run(arguments).code_
        if not callable(function):
            error_msg = f'A compiled printer returns a function, not a {type(function).__name__}.'
            raise TypeError(error_msg)

        def evaluate(t: float, x: Array, q: Array) -> list[Array]:
            return list(function(t, *(reader.get_column_of(x, q) for reader in readers)))

        return evaluate

    @staticmethod
    def _get_entry_name(request: _RequestSpec, names: dict[str, int]) -> str:
        """Return the name of the entry of the drawn values of a request.

        The name is unique among the entries of the draws. A component with several kernel specs, such as a hyperedge
        interaction with one spec per size of hyperedge, draws one name in every spec, so the entries after the first of
        a name end with their number. Backends that place entries by name, as Numba does, would otherwise read the
        values of every spec from one entry.

        Args:
            request:
                The request of the drawn values.

            names:
                The number of entries already named after every component path and draw name, which the call updates.

        Returns:
            name:
                The name of the entry.
        """
        name = f'{request.owner}.{request.name}~draw'
        count = names.get(name, 0)
        names[name] = count + 1
        return name if count == 0 else f'{name}{count}'

    @staticmethod
    def _get_replacement(
        draw: DrawSpec,
        arguments: list[Expression],
        slot: Expression,
        *,
        relaxes: bool = False,
    ) -> Expression:
        """Return the expression of a draw in terms of a uniform or normal value.

        The value is uniform or standard normal, or else the sampled value of the draw. A relaxed draw gives its
        relaxation when `relaxes` is true, and a hard one `exact + (relaxed - stop_gradient(relaxed))`. The difference
        is held in a choice of the uniform value, which is never negative, so Sage keeps it apart from the exact value
        and it evaluates to exactly 0.
        """
        if draw.kind == 'draw':
            return slot
        if draw.kind == 'uniform':
            low, high = arguments
            return low + (high - low) * slot
        if draw.kind == 'normal':
            mean, deviation = arguments
            return mean + deviation * slot
        if draw.kind == 'bernoulli':
            exact = where(slot < arguments[0], 1, 0)
        else:
            total = sum(arguments[1:], arguments[0])
            cumulative, exact = SR(0), SR(0)
            for weight in arguments[:-1]:
                cumulative = cumulative + weight
                exact = exact + where(slot * total >= cumulative, 1, 0)
        if not relaxes or draw.temperature is None:
            return exact
        relaxed = DrawLowering._get_relaxation(draw, arguments, slot)
        if not draw.hard:
            return relaxed
        return exact + where(slot >= 0, relaxed - stop_gradient(relaxed), 0)

    @staticmethod
    def _get_relaxation(draw: DrawSpec, arguments: list[Expression], slot: Expression) -> Expression:
        """Return the continuous relaxation of a Bernoulli or categorical draw.

        The draw is of a uniform value.

        A Bernoulli draw of probability `p` gives `sigmoid((logit p - logit u) / T)`, the binary concrete relaxation of
        the threshold `u < p` (Maddison et al. 2017). A categorical draw gives `sum_k sigmoid((u - c_k / W) / T)` over
        the cumulative weights `c_k` before the last and the total weight `W`, the relaxation of its inverse
        distribution function. Both tend to the exact draw as the temperature `T` tends to 0. The probability is clipped
        into `[1e-12, 1 - 1e-12]`, so its logit and its gradient stay finite.
        """
        temperature = SR(draw.temperature)
        if draw.kind == 'bernoulli':
            p = clip(arguments[0], RELAXATION_FLOOR, 1 - RELAXATION_FLOOR)
            return logistic(log(p) - log(1 - p) - log(slot) + log(1 - slot), 0, temperature)
        total = sum(arguments[1:], arguments[0])
        cumulative, relaxed = SR(0), SR(0)
        for weight in arguments[:-1]:
            cumulative = cumulative + weight
            relaxed = relaxed + logistic(slot - cumulative / total, 0, temperature)
        return relaxed

    @staticmethod
    def _get_lowered(expression: Expression, slots: Mapping[str, Expression], *, relaxes: bool = False) -> Expression:
        """Return an expression with every draw replaced by its value expression.

        The innermost draws are replaced first, relaxed draws by their relaxation when `relaxes` is true.
        """
        found = draws_in(expression)
        while found:
            draw = next(item for item in found if not draws_in(list(item.arguments)))
            replacement = DrawLowering._get_replacement(draw, list(draw.arguments), slots[draw.name], relaxes=relaxes)
            expression = expression.subs({draw.call: replacement})
            found = draws_in(expression)
        return expression

    def _get_sampler(self, request: _RequestSpec) -> Callable[[float, Array, Array, np.random.Generator], Array]:
        """Return the function that samples a draw for every row of a request.

        It samples from a `scipy.stats` distribution.
        """
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
        evaluate = self._get_arguments_of(request, draw)
        if evaluate is None:
            error_msg = (
                f'The draw {draw.name!r} of {request.owner} reads a symbol that is not a state or a parameter of its '
                'component. Give the arguments of draws as expressions of the states and parameters of the member.'
            )
            raise ValidationError(error_msg)
        rows = request.rows

        def sample(t: float, x: Array, q: Array, rng: np.random.Generator) -> Array:
            return DrawKernel._sample_draw(draw, evaluate(t, x, q), rows, rng)

        return sample

    @staticmethod
    def _is_symbolic(value: object) -> bool:
        """Check whether a law value is symbolic.

        A value that is not symbolic is a call or a Python function.
        """
        return isinstance(value, Expression | FreeModuleElement) or not (isinstance(value, CallSpec) or callable(value))

    @staticmethod
    def _get_distribution_of(owner: str, description: str, draw: DrawSpec) -> stats.rv_continuous | stats.rv_discrete:
        """Return the `scipy.stats` distribution of a draw.

        It first checks the name and the keywords of the draw.

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
