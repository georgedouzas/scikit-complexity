"""Define the kernel spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ...components.elements import FieldElement
from ...errors import ValidationError
from ...specs._base_spec import BaseSpec
from ...types import Indices
from ._problem_spec import ProblemSpec
from ._reference_spec import PairsSearch, ReferenceSpec
from ._set_symbol_spec import SetSymbolSpec


@dataclass(frozen=True)
class KernelSpec(BaseSpec):
    """Namespace, references, and evaluation indices of the laws of one component.

    The laws of the component are lowered with them.

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

        edges:
            The spec of the edges of an interaction over neighbours, whose laws that read one end of an edge the spec
            evaluates once per member of the group on the nodes, or `None`.

        hyperedge:
            `'member'` for the laws of the focal member of every hyperedge of a hyperedge interaction, `'hyperedge'`
            for its laws of the member of a group on hyperedges or edges, or `None` for other components.

        reduce:
            `'sum'` or `'mean'` for the update laws of a grouping, which set every member to the sum or the mean of
            their values over the groups that hold it, or `None`.

        counts:
            The number of groups that hold the member of every row of a grouping, or `None`.

        place:
            The field on the nodes of an interaction over neighbours, whose laws per edge lower to the field's
            Laplacian, or on a lattice to whole-site shifted reads, or `None`.
    """

    owner: str
    namespace: SimpleNamespace
    references: Mapping[Expression, ReferenceSpec]
    rows: int
    pairs: PairsSearch | None = None
    roles: dict[str, str | tuple[str, ...]] = field(default_factory=dict)
    group: str | None = None
    groups: dict[str, int] = field(default_factory=dict)
    sets: dict[Expression, SetSymbolSpec] = field(default_factory=dict)
    set_index: int = 0
    aligned: tuple[str, ...] = ()
    edges: KernelSpec | None = field(default=None, compare=False, repr=False)
    hyperedge: str | None = None
    reduce: str | None = None
    counts: Indices | None = field(default=None, compare=False, repr=False)
    place: FieldElement | None = field(default=None, compare=False, repr=False)

    def _target_references(self, target: object) -> list[ReferenceSpec]:
        """Return the references of the state components a law targets."""
        components = ProblemSpec._real_components_of(target)
        found = [self.references.get(component) for component in components]
        references = [reference for reference in found if reference is not None and reference.is_state]
        if len(references) != len(components):
            error_msg = f'A law of {self.owner} targets {target!r}, which is not a state symbol of the namespace `s`.'
            raise ValidationError(error_msg)
        return references

    def _relative_names(self, names: Iterable[str]) -> dict[str, str]:
        """Return the dotted name of every variable by its name in the component.

        A variable of the component takes its own name, and a variable of the element bound to a role, or of its
        elements, takes the role and its name, such as `target.x`.
        """
        prefixes = {self.owner: ''}
        for role, paths in self.roles.items():
            if isinstance(paths, str):
                prefixes[paths] = f'{role}.'
            else:
                prefixes.update({path: f'{role}.{index}.' for index, path in enumerate(paths)})
        return {
            f'{local}{name.removeprefix(path + ".")}': name
            for path, local in prefixes.items()
            for name in names
            if name.startswith(f'{path}.')
        }

    def _state_name(self, target: Expression) -> str:
        """Return the dotted name of the state a target symbol of the namespace stands for.

        Other symbols return themselves.
        """
        reference = self.references.get(target)
        return reference.entry.name if reference is not None else str(target)
