"""Pass that adds the compiled renewal laws and the symbols of their counts to the problem."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.laws.utils import draws_in, reductions_in
from ....errors import ValidationError
from ....specs import DrawSpec
from ....specs._draw_spec import DRAWS
from ....types import Indices, ReadKind
from ..._utils import components_of, symbol, symbols_of
from ...problems import DeclarationsSpec, ExpressionPrinter, ProblemSpec, ReferenceSpec
from ...problems._renewal_kernel import _READ_SOURCES, MemberFunction, RenewalKernel, Sampler
from .._base import BasePass
from .._bound_renewal_spec import BoundRenewalSpec
from .._draw_lowering import DrawLowering
from .._renewal_law import RenewalLaw


class RenewalProblemPass(BasePass):
    """Pass that adds the compiled renewal laws and the symbols of their counts to the problem."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'renewals', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the renewal laws to the problem."""
        compilation = self.compilation_
        problem = cast('ProblemSpec', compilation.problem)
        time = cast('DeclarationsSpec', compilation.layout).time
        symbols = {item.entry.name: item.symbol for item in compilation.renewals}
        compilation.problem = replace(
            problem,
            renewals=self._compile_renewals(compilation.renewals, time, compilation.owners),
            symbols={**problem.symbols, **symbols},
        )

    @staticmethod
    def _compile_renewals(
        renewals: list[BoundRenewalSpec],
        time: Expression,
        owners: frozenset[str] | None,
    ) -> tuple[RenewalKernel, ...]:
        """Return the renewal laws of the components a problem holds, compiled for their members."""
        return tuple(
            RenewalProblemPass._compile_renewal(
                item.spec.owner,
                item.law,
                item.spec.references,
                time=time,
                rows=item.spec.rows,
                label=item.label,
                group=item.group,
                counts=item.entry.start + np.arange(item.entry.size, dtype=np.intp),
            )
            for item in renewals
            if owners is None or item.group in owners
        )

    @staticmethod
    def _declared(owner: str, found: list[DrawSpec]) -> dict[str, DrawSpec]:
        """Return the draws of a renewal law by name, after checking that each name has one declaration."""
        declared: dict[str, DrawSpec] = {}
        for item in found:
            known = declared.setdefault(item.name, item)
            if (known.kind, known.distribution, known.parameters, known.arguments) != (
                item.kind,
                item.distribution,
                item.parameters,
                item.arguments,
            ):
                error_msg = (
                    f'A renewal law of {owner} uses the draw name {item.name!r} for {known.call} and for {item.call}. '
                    'Use one declaration per name, or give the draws different names.'
                )
                raise ValidationError(error_msg)
        return declared

    @staticmethod
    def _lowest_value(owner: str, description: str, waiting: Expression) -> float:
        """Return the lowest value a waiting time that is a single draw can take, or 0 when it is not known."""
        operator = waiting.operator()
        if operator is None or getattr(operator, 'name', lambda: '')() not in DRAWS:
            return 0.0
        found = draws_in(waiting)[0]
        numbers = [float(item) if SR(item).is_numeric() else None for item in found.arguments]
        if found.kind == 'normal':
            return -math.inf
        if found.kind == 'uniform':
            return numbers[0] if numbers[0] is not None else 0.0
        if found.kind != 'draw':
            return 0.0
        distribution = DrawLowering._distribution_of(owner, description, found)
        values = dict(zip(found.keywords, numbers, strict=True))
        shapes = {key: value for key, value in values.items() if key not in {'loc', 'scale'}}
        lowest = (
            float(distribution.support(**shapes)[0])
            if all(value is not None for value in shapes.values())
            else float(distribution.a)
        )
        if lowest < 0:
            return lowest
        loc = values.get('loc', 0.0)
        return 0.0 if loc is None else loc

    @staticmethod
    def _member_function(
        owner: str,
        expressions: Sequence[Expression],
        references: dict[Expression, ReferenceSpec],
        slots: dict[Expression, str],
        time: Expression,
    ) -> MemberFunction:
        """Return the function of renewing members that evaluates expressions, after checking the symbols they read."""
        arguments = sorted(symbols_of(list(expressions)) - {time}, key=str)
        readers: list[ReferenceSpec | str] = []
        for item in arguments:
            if item in slots:
                readers.append(slots[item])
                continue
            reference = references.get(item)
            if reference is None or reference.source not in _READ_SOURCES:
                error_msg = (
                    f'A renewal law of {owner} reads {item}, which is not a state or a parameter of its component. '
                    'Renewal laws read the states and parameters of the renewing member.'
                )
                raise ValidationError(error_msg)
            readers.append(reference)
        printer = ExpressionPrinter(time=time, compiled=True).check().prepare(arguments)
        return printer.run(list(expressions)), tuple(readers)

    @staticmethod
    def _ordered(found: list[DrawSpec]) -> list[DrawSpec]:
        """Return draws ordered so that every draw comes after the draws in its arguments."""
        ordered: list[DrawSpec] = []
        pending = list(found)
        while pending:
            done = {item.name for item in ordered}
            ready = next(item for item in pending if {inner.name for inner in draws_in(list(item.arguments))} <= done)
            ordered.append(ready)
            pending.remove(ready)
        return ordered

    @staticmethod
    def _recorded_states(
        group: str, references: dict[Expression, ReferenceSpec], rows: int,
    ) -> tuple[Indices, tuple[str, ...]]:
        """Return the flat indices and the names of the states of every member of the group a renewal law renews."""
        columns: dict[int, tuple[Indices, str]] = {}
        for reference in references.values():
            owned = reference.entry.name.rpartition('.')[0] == group
            if not reference.is_state or reference.source not in _READ_SOURCES or not owned:
                continue
            flat = reference.flat_indices if reference.source is ReadKind.MEMBER else np.asarray([reference.index])
            if flat.size != rows:
                continue
            name = (
                reference.entry.name if reference.entry.columns == 1 else f'{reference.entry.name}[{reference.column}]'
            )
            columns[int(flat[0])] = (np.asarray(flat, dtype=np.intp), name)
        ordered = [columns[key] for key in sorted(columns)]
        indices = np.stack([flat for flat, _ in ordered], axis=-1) if ordered else np.zeros((rows, 0), dtype=np.intp)
        return indices, tuple(name for _, name in ordered)

    @staticmethod
    def _target_places(
        owner: str,
        targets: list[Expression],
        references: dict[Expression, ReferenceSpec],
        rows: int,
    ) -> Indices:
        """Return the flat indices every reset component writes, one row per component and one column per member."""
        places = []
        for item in targets:
            reference = references.get(item)
            if reference is None or not reference.is_state or reference.source not in _READ_SOURCES:
                error_msg = (
                    f'A renewal law of {owner} resets {item}, which is not a state of an element or a group member.'
                )
                raise ValidationError(error_msg)
            if reference.source is ReadKind.SINGLE and rows > 1:
                error_msg = (
                    f'A renewal law of {owner} resets {item}, a state of a single element, while its members renew on '
                    'their own clocks. Reset the states of the renewing members.'
                )
                raise ValidationError(error_msg)
            flat = reference.flat_indices if reference.source is ReadKind.MEMBER else np.asarray([reference.index])
            places.append(np.asarray(flat, dtype=np.intp))
        return np.asarray(places, dtype=np.intp).reshape(len(places), rows)

    @staticmethod
    def _compile_renewal(
        owner: str,
        law: RenewalLaw,
        references: dict[Expression, ReferenceSpec],
        *,
        time: Expression,
        rows: int,
        label: str,
        group: str,
        counts: Indices,
    ) -> RenewalKernel:
        """Return a renewal law compiled for the members of its component.

        Args:
            owner:
                The path of the component the law belongs to.

            law:
                The renewal law.

            references:
                The references of the symbols of the component.

            time:
                The time symbol.

            rows:
                The number of members that renew on their own clocks.

            label:
                The name of the law.

            group:
                The path of the element or group whose states the law renews.

            counts:
                The flat index of the hidden renewal count of every member.

        Returns:
            renewal:
                The compiled renewal law.

        Raises:
            ValidationError:
                The waiting time can be negative, the law reads a reduction or a symbol of another component, resets a
                symbol that is not a state of the renewing members, or gives one draw name two declarations.
        """
        description = f'renewal law {label}'
        reset = law.reset or {}
        targets = [item for target in reset for item in components_of(target)]
        values = [
            SR(item)
            for target, value in reset.items()
            for item in (
                components_of(value) if isinstance(value, FreeModuleElement) else [value] * len(components_of(target))
            )
        ]
        waiting = SR(law.waiting)
        if any(reductions_in(item) for item in [*values, waiting]):
            error_msg = (
                f'The {description} of {owner} reads a reduction over a group. Renewal laws read the states and '
                'parameters of the renewing member.'
            )
            raise ValidationError(error_msg)
        if RenewalProblemPass._lowest_value(owner, description, waiting) < 0:
            error_msg = (
                f'The {description} of {owner} draws its waiting time from {waiting}, which can be negative. Draw '
                "waiting times from a distribution with nonnegative support, such as 'expon', 'pareto', or 'lomax'."
            )
            raise ValidationError(error_msg)
        first = RenewalProblemPass._declared(owner, draws_in(values))
        later = {
            name: item
            for name, item in RenewalProblemPass._declared(owner, [*first.values(), *draws_in(waiting)]).items()
            if name not in first
        }
        slots = {name: symbol(f'_renewal_{name}') for name in [*first, *later]}
        calls = {item.call: slots[item.name] for item in [*first.values(), *later.values()]}
        names = {slot: name for name, slot in slots.items()}

        def samplers(found: dict[str, DrawSpec]) -> tuple[Sampler, ...]:
            return tuple(
                (
                    item,
                    RenewalProblemPass._member_function(
                        owner, [SR(arg).subs(calls) for arg in item.arguments], references, names, time,
                    ),
                )
                for item in RenewalProblemPass._ordered(list(found.values()))
            )

        lowered = [item.subs(calls) for item in values]
        return RenewalKernel(
            label=label,
            members=rows,
            counts=counts,
            states=RenewalProblemPass._recorded_states(group, references, rows)[0],
            names=RenewalProblemPass._recorded_states(group, references, rows)[1],
            targets=RenewalProblemPass._target_places(owner, targets, references, rows),
            reset=RenewalProblemPass._member_function(owner, lowered, references, names, time) if lowered else None,
            waiting=RenewalProblemPass._member_function(owner, [waiting.subs(calls)], references, names, time),
            before=samplers(first),
            after=samplers(later),
        ).check()
