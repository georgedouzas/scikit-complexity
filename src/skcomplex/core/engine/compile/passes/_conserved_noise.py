"""Pass that lowers every conserved noise law to the divergence of random fluxes.

There is one random flux per link of its field.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ....components.elements import FieldElement
from ....components.laws import NoiseLaw
from ....components.spaces import LatticeSpace
from ....components.specs import CensusRecordSpec
from ....errors import ValidationError
from ....specs import IssueSpec, NoiseProcessSpec
from ....types import Indices
from ...problems._declarations_spec import DeclarationsSpec
from ...problems._output_spec import OutputSpec
from ...problems._reference_spec import ReferenceSpec
from ...types import ReadKind
from .._base import BasePass
from .._compilation import Compilation, Laws
from .._field_lowering import FieldLowering


class ConservedNoisePass(BasePass):
    """Pass that lowers every conserved noise law to the divergence of random fluxes.

    There is one random flux per link of its field.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('reduced', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('outputs',)

    def _run(self, inputs: object = None) -> None:
        """Set the outputs of the conserved noise laws."""
        compilation = self.compilation_
        declarations = Compilation._required(compilation.layout, 'layout')
        reduced = Compilation._required(compilation.reduced, 'reduced')
        compilation.outputs = self._conserved_outputs(reduced, declarations)

    @staticmethod
    def _conserved_outputs(specs: Laws, declarations: DeclarationsSpec) -> dict[int, list[OutputSpec]]:
        """Return the outputs of every conserved noise law, by law identity.

        There is one process per link of its field.
        """
        places = FieldLowering._field_entries(declarations)
        outputs = {}
        for spec, laws in specs:
            for law in laws:
                if isinstance(law, NoiseLaw) and law.conserved:
                    targets = spec._target_references(law._get_targets()[0])
                    outputs[id(law)] = ConservedNoisePass._law_outputs(
                        spec.owner,
                        law,
                        targets,
                        places,
                        spec.references,
                    )
        return outputs

    @staticmethod
    def _amplitude(owner: str, law: NoiseLaw, references: Mapping[Expression, ReferenceSpec]) -> Expression:
        """Return the amplitude of a conserved noise.

        An amplitude that reads a state or is not symbolic raises an error.
        """
        if not isinstance(law.value, Expression | int | float):
            error_msg = (
                f'The conserved noise law of {owner} has the value {law.value!r}. Give a number or an expression.'
            )
            raise ValidationError(error_msg)
        amplitude = SR(law.value)
        reads = [item for item in amplitude.variables() if item in references and references[item].is_state]
        if reads:
            error_msg = (
                f'The amplitude of the conserved noise law of {owner} reads the states {", ".join(map(str, reads))}. '
                'Write it as a number or an expression of parameters.'
            )
            raise ValidationError(error_msg)
        return amplitude

    @staticmethod
    def _check_place(owner: str, reference: ReferenceSpec, place: object) -> tuple[FieldElement, LatticeSpace]:
        """Return the field a conserved noise drives and its lattice.

        An error is raised when there is none.
        """
        if (
            not isinstance(place, FieldElement)
            or not isinstance(place.space, LatticeSpace)
            or place.space.geometry != 'square'
            or place.discretization != 'finite-difference'
            or reference.source is not ReadKind.MEMBER
        ):
            error_msg = (
                f'The conserved noise law of {owner} drives {reference.entry.name}, which is not a state of a '
                'finite-difference field on a square lattice. Declare the state on a `FieldElement` of a '
                '`LatticeSpace`.'
            )
            raise ValidationError(error_msg)
        return place, place.space

    @staticmethod
    def _link_tables(dimensions: tuple[int, ...], periodic: bool) -> list[tuple[Indices, Indices]]:
        """Return the links of every site along every dimension of a lattice.

        The links are numbered `k n + i` for site `i`. The link `(i, k)` joins site `i` to its next neighbour along
        dimension `k`. Without periodic edges, the links that leave the lattice carry no process.

        Args:
            dimensions:
                The number of sites along every dimension.

            periodic:
                Whether opposite edges are joined.

        Returns:
            tables:
                For every dimension, the link to the next neighbour and the link from the previous neighbour of every
                site in member order, or -1 where the link carries no process.
        """
        size = int(np.prod(dimensions))
        sites = np.arange(size, dtype=np.intp).reshape(dimensions)
        tables = []
        for axis in range(len(dimensions)):
            following = axis * size + sites
            preceding = np.roll(following, 1, axis=axis)
            if not periodic:
                following = following.copy()
                np.moveaxis(following, axis, 0)[-1] = -1
                np.moveaxis(preceding, axis, 0)[0] = -1
            tables.append((following.ravel(), preceding.ravel()))
        return tables

    @staticmethod
    def _law_outputs(
        owner: str,
        law: NoiseLaw,
        targets: list[ReferenceSpec],
        places: Mapping[str, FieldElement],
        references: Mapping[Expression, ReferenceSpec],
    ) -> list[OutputSpec]:
        """Return the outputs of a conserved noise law.

        They are the divergence of a random flux on the links of a lattice. Every target component gets one process per
        link. Site `i` takes `a / h` times the increment of the link from its previous neighbour and `-a / h` times the
        increment of the link to its next neighbour along every dimension.

        Args:
            owner:
                The path of the component that returned the law.

            law:
                The conserved noise law.

            targets:
                The target components of the law.

            places:
                The fields by the name of each of their entries.

            references:
                A mapping from every symbol the law may use to the component it stands for.

        Returns:
            outputs:
                Two outputs per dimension and target component, with the link process of every site.

        Raises:
            ValidationError:
                A target is not a state of a finite-difference field on a square lattice, or the amplitude reads a
                state.
        """
        amplitude = ConservedNoisePass._amplitude(owner, law, references)
        outputs = []
        for reference in targets:
            place = places.get(reference.entry.name)
            field, space = ConservedNoisePass._check_place(owner, reference, place)
            kind, _ = field._get_boundary(reference.entry.name.rpartition('.')[2], reference.column)
            tables = ConservedNoisePass._link_tables(space.shape_, kind == 'periodic')
            key = ('conserved', reference.entry.name, reference.column)
            dimension = len(tables) * space.size_
            for spacing, (following, preceding) in zip(space.spacings_, tables, strict=True):
                scaled = amplitude / spacing
                outputs.append(
                    OutputSpec(reference, -scaled, process=NoiseProcessSpec(key, dimension=dimension, links=following)),
                )
                outputs.append(
                    OutputSpec(reference, scaled, process=NoiseProcessSpec(key, dimension=dimension, links=preceding)),
                )
        return outputs

    @staticmethod
    def _is_lattice_field(component: object) -> bool:
        """Check whether a component is a finite-difference field on a square lattice."""
        return (
            isinstance(component, FieldElement)
            and isinstance(component.space, LatticeSpace)
            and component.space.geometry == 'square'
            and component.discretization == 'finite-difference'
        )

    @staticmethod
    def _conserved_issue(record: CensusRecordSpec, components: Mapping[str, object]) -> IssueSpec | None:
        """Return the issue of a conserved noise law outside a lattice field.

        The law drives a state outside a finite-difference lattice field.
        """
        outside = [
            name
            for name, _ in record.driven
            if not ConservedNoisePass._is_lattice_field(components.get(name.rpartition('.')[0]))
        ]
        if not outside:
            return None
        message = (
            f'The conserved {type(record.law).__name__} of {record.path} drives {", ".join(dict.fromkeys(outside))}, '
            'but conserved noise needs a state of a finite-difference field on a square lattice, with one process per '
            'link.'
        )
        fix = 'Declare the state on a `FieldElement` of a `LatticeSpace`, or remove `conserved=True`.'
        return IssueSpec('noise-unsupported', record.path, message, fix)

    @staticmethod
    def _conserved_issues(compilation: Compilation) -> list[IssueSpec]:
        """Return the conserved noise laws that drive a state outside a lattice field.

        The field must be a finite-difference field on a lattice.
        """
        components = {node.path: node.component for node in Compilation._required(compilation.nodes, 'nodes')}
        records = Compilation._required(compilation.census, 'census').records
        found = [
            ConservedNoisePass._conserved_issue(record, components)
            for record in records
            if isinstance(record.law, NoiseLaw) and record.law.conserved
        ]
        return [issue for issue in found if issue is not None]
