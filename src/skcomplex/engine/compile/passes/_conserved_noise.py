"""Pass that lowers every conserved noise law to the divergence of one random flux per link of its field."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ....core.elements import FieldElement
from ....core.laws import NoiseLaw
from ....core.spaces import LatticeSpace
from ....errors import ValidationError
from ....types import Indices
from ...problems._noise_process import NoiseProcess
from ...problems._output import Output
from ...problems._read_kind import ReadKind
from ...problems._reference import Reference
from .._base import BasePass
from .._census import Census
from .._census_record import CensusRecord
from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._field_lowering import FieldLowering
from .._issue import Issue
from .._kernel_spec import KernelSpec
from .._tree import Node


class ConservedNoisePass(BasePass):
    """Pass that lowers every conserved noise law to the divergence of one random flux per link of its field."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('reduced', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('outputs',)

    def _run(self, inputs: object = None) -> None:
        """Set the outputs of the conserved noise laws."""
        compilation = self.compilation_
        declarations = cast('Declarations', compilation.layout)
        compilation.outputs = self._conserved_outputs(cast('Laws', compilation.reduced), declarations)

    @staticmethod
    def _conserved_outputs(specs: Laws, declarations: Declarations) -> dict[int, list[Output]]:
        """Return the outputs of every conserved noise law, one process per link of its field, by law identity."""
        places = FieldLowering._field_entries(declarations)
        outputs = {}
        for spec, laws in specs:
            for law in laws:
                if isinstance(law, NoiseLaw) and law.conserved:
                    targets = KernelSpec._target_references(law._get_targets()[0], spec)
                    outputs[id(law)] = ConservedNoisePass._law_outputs(
                        spec.owner, law, targets, places, spec.references,
                    )
        return outputs

    @staticmethod
    def _amplitude(owner: str, law: NoiseLaw, references: Mapping[Expression, Reference]) -> Expression:
        """Return the amplitude of a conserved noise, or raise an error when it reads a state or is not symbolic."""
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
    def _check_place(owner: str, reference: Reference, place: object) -> LatticeSpace:
        """Return the lattice of the field a conserved noise drives, or raise an error when there is none."""
        space = getattr(place, 'space', None)
        is_lattice = (
            isinstance(place, FieldElement)
            and isinstance(space, LatticeSpace)
            and space.geometry == 'square'
            and place.discretization == 'finite-difference'
            and reference.source is ReadKind.MEMBER
        )
        if not is_lattice:
            error_msg = (
                f'The conserved noise law of {owner} drives {reference.entry.name}, which is not a state of a '
                'finite-difference field on a square lattice. Declare the state on a `FieldElement` of a '
                '`LatticeSpace`.'
            )
            raise ValidationError(error_msg)
        return space  # type: ignore[return-value]

    @staticmethod
    def _link_tables(dimensions: tuple[int, ...], periodic: bool) -> list[tuple[Indices, Indices]]:
        """Return the links of every site along every dimension of a lattice, numbered `k n + i` for site `i`.

        The link `(i, k)` joins site `i` to its next neighbour along dimension `k`. Without periodic edges, the links
        that leave the lattice carry no process.

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
        targets: list[Reference],
        places: Mapping[str, FieldElement],
        references: Mapping[Expression, Reference],
    ) -> list[Output]:
        """Return the outputs of a conserved noise law, the divergence of a random flux on the links of a lattice.

        Every target component gets one process per link. Site `i` takes `a / h` times the increment of the link from
        its previous neighbour and `-a / h` times the increment of the link to its next neighbour along every dimension.

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
            space = ConservedNoisePass._check_place(owner, reference, place)
            kind, _ = place._get_boundary(reference.entry.name.rpartition('.')[2])  # type: ignore[arg-type]
            tables = ConservedNoisePass._link_tables(space.shape_, kind == 'periodic')
            key = ('conserved', reference.entry.name, reference.column)
            dimension = len(tables) * space.size_
            for spacing, (following, preceding) in zip(space.spacings_, tables, strict=True):
                scaled = amplitude / spacing
                outputs.append(
                    Output(reference, -scaled, process=NoiseProcess(key, dimension=dimension, links=following)),
                )
                outputs.append(
                    Output(reference, scaled, process=NoiseProcess(key, dimension=dimension, links=preceding)),
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
    def _conserved_issue(record: CensusRecord, components: Mapping[str, object]) -> Issue | None:
        """Return the issue of a conserved noise law that drives a state outside a finite-difference lattice field."""
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
        return Issue('noise-unsupported', record.path, message, fix)

    @staticmethod
    def _conserved_issues(compilation: Compilation) -> list[Issue]:
        """Return the conserved noise laws that drive a state that is not of a finite-difference field on a lattice."""
        components = {node.path: node.component for node in cast('list[Node]', compilation.nodes)}
        records = cast('Census', compilation.census).records
        found = [
            ConservedNoisePass._conserved_issue(record, components)
            for record in records
            if isinstance(record.law, NoiseLaw) and record.law.conserved
        ]
        return [issue for issue in found if issue is not None]
