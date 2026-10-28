"""Define the kernel spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING

from sage.all__sagemath_symbolics import Expression

from ..errors import ValidationError
from ._base_spec import BaseSpec
from ._problem_spec import ProblemSpec
from ._reference_spec import ReferenceSpec
from ._set_symbol_spec import SetSymbolSpec

if TYPE_CHECKING:
    from ._reference_spec import Search


@dataclass(frozen=True)
class KernelSpec(BaseSpec):
    """Namespace, references, and evaluation indices that the laws of one component are lowered with.

    Args:
        owner:
            The path of the component.

        namespace:
            The namespace the laws of the component read.

        references:
            The component of a state or parameter every symbol of the namespace stands for.

        rows:
            The number of rows the laws are evaluated over.

        pairs:
            The search of the pairs an interaction over pairs or edges reads, or `None`.

        roles:
            The path, or the paths, every role binds to.

        group:
            The group whose members the laws are evaluated over, or `None`.

        groups:
            The size of every group the component reads member by member, by path.

        sets:
            The set symbol of every symbol that stands for one element of an element set.

        set_index:
            The index of the element of a set the spec is evaluated for.

        aligned:
            The groups of one size the laws read member by member, row `i` of each together.
    """

    owner: str
    namespace: SimpleNamespace
    references: Mapping[Expression, ReferenceSpec]
    rows: int
    pairs: Search | None = None
    roles: dict[str, str | tuple[str, ...]] = field(default_factory=dict)
    group: str | None = None
    groups: dict[str, int] = field(default_factory=dict)
    sets: dict[Expression, SetSymbolSpec] = field(default_factory=dict)
    set_index: int = 0
    aligned: tuple[str, ...] = ()

    def _target_references(self, target: object) -> list[ReferenceSpec]:
        """Return the references of the state components a law targets."""
        components = ProblemSpec._real_components_of(target)
        references = [self.references.get(component) for component in components]
        if not all(reference is not None and reference.is_state for reference in references):
            error_msg = f'A law of {self.owner} targets {target!r}, which is not a state symbol of the namespace `s`.'
            raise ValidationError(error_msg)
        return references  # type: ignore[return-value]

    def _state_name(self, target: Expression) -> str:
        """Return the dotted name of the state a target symbol of the namespace stands for, or the symbol itself."""
        reference = self.references.get(target)
        return reference.entry.name if reference is not None else str(target)
