"""Read the residence of states through hidden states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

import numpy as np
from sage.all__sagemath_symbolics import RR, SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....components.laws import BaseLaw, UpdateLaw
from ....components.laws.utils import where
from ....errors import ValidationError
from ....specs import EntrySpec
from ..._utils import components_of, symbol, vector
from ...problems._kernel_spec import KernelSpec
from ...problems._reference_spec import ReferenceSpec
from ...types import ReadKind
from .._base import BasePass
from .._compilation import Compilation
from ._layout import LayoutPass

RESIDENCE_PREFIX = '__residence'


class ResidencesPass(BasePass):
    """A pass that reads the residence of states through hidden states.

    The hidden states keep their previous values and counts.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('lowered', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('lowered',)

    def _run(self, inputs: object = None) -> None:
        """Replace the residences of the lowered laws by hidden states.

        The update laws of those states are added.
        """
        compilation = self.compilation_
        lowered = Compilation._get_required(compilation.lowered, 'lowered')
        replaced: dict[int, KernelSpec] = {}
        for index, (spec, laws, block) in enumerate(lowered):
            if not any(self._get_calls_in(law) for law in laws):
                continue
            references: dict[Expression, ReferenceSpec] = {}
            found = self._copy_with_residences(spec, laws, compilation, references)
            lowered[index] = (spec, found, block)
            replaced[id(spec)] = replace(spec, references={**spec.references, **references})
        compilation._replace_specs(replaced)

    @staticmethod
    def _get_calls_in(law: BaseLaw) -> list[Expression]:
        """Return the residences that the values of a law read."""
        values = [item for item in law._get_values() if isinstance(item, Expression | FreeModuleElement)]
        return [
            call
            for value in values
            for item in components_of(value)
            if isinstance(item, Expression)
            for call in BasePass._get_calls_of(item, 'residence')
        ]

    @staticmethod
    def _declare(
        spec: KernelSpec,
        reference: ReferenceSpec,
        compilation: Compilation,
        references: dict[Expression, ReferenceSpec],
    ) -> tuple[Expression, Expression]:
        """Return the symbols of the hidden previous value and count of a state.

        They are declared in the layout.
        """
        declarations = Compilation._get_required(compilation.layout, 'layout')
        entry = reference.entry
        members = entry.members if reference.source is ReadKind.MEMBER else None
        indices = reference.flat_indices if members is not None else np.asarray([reference.index])
        initial = np.concatenate([np.zeros(0), *declarations.initial_states])[indices]
        position = sum(item.name.startswith(RESIDENCE_PREFIX) for item in declarations.states) // 2
        shape = (members,) if members is not None else ()
        found = []
        for name, value in (('previous', initial), ('count', np.full(initial.shape, -1.0))):
            hidden = EntrySpec(f'{RESIDENCE_PREFIX}_{position}.{name}', declarations.ends[True], shape, RR, members)
            declarations = LayoutPass._copy_with_entry(
                declarations,
                hidden,
                np.asarray(value, dtype=float),
                is_state=True,
            )
            item = symbol(f'_residence_{position}_{name}')
            references[item] = ReferenceSpec(hidden, True, 0, reference.source)
            found.append(item)
        compilation.layout = declarations
        return found[0], found[1]

    @staticmethod
    def _copy_with_residences(
        spec: KernelSpec,
        laws: list[BaseLaw],
        compilation: Compilation,
        references: dict[Expression, ReferenceSpec],
    ) -> list[BaseLaw]:
        """Return update laws with every residence replaced by its count.

        The laws of the hidden states follow them. The references of the hidden states are added to `references`, which
        the spec receives after its laws.
        """
        counts: dict[Expression, Expression] = {}
        added: list[BaseLaw] = []
        for law in laws:
            for call in ResidencesPass._get_calls_in(law):
                if not isinstance(law, UpdateLaw):
                    error_msg = (
                        f'A {type(law).__name__} of {spec.owner} reads {call}, which only update laws of discrete time '
                        'read.'
                    )
                    raise ValidationError(error_msg)
                item = call.operands()[0]
                reference = spec.references.get(item)
                if (
                    reference is None
                    or not reference.is_state
                    or reference.source
                    not in {
                        ReadKind.SINGLE,
                        ReadKind.MEMBER,
                    }
                ):
                    error_msg = f'{spec.owner} reads {call}, but {item} is not a state of an element or a group member.'
                    raise ValidationError(error_msg)
                if call in counts:
                    continue
                previous, count = ResidencesPass._declare(spec, reference, compilation, references)
                counts[call] = where(item == previous, count + 1, 0)
                added += [UpdateLaw(previous, item), UpdateLaw(count, counts[call])]
        return [*(ResidencesPass._get_substituted(law, counts) for law in laws), *added]

    @staticmethod
    def _get_substituted(law: BaseLaw, counts: dict[Expression, Expression]) -> BaseLaw:
        """Return an update law with its residences replaced by their counts.

        Any other law is returned unchanged.
        """
        if not isinstance(law, UpdateLaw) or not counts:
            return law
        value = law.value
        if isinstance(value, FreeModuleElement):
            value = vector(SR(item).subs(counts) for item in value)
        elif isinstance(value, Expression):
            value = value.subs(counts)
        order = law.order.subs(counts) if isinstance(law.order, Expression) else law.order
        return BasePass._get_replaced(law, value=value, order=order)
