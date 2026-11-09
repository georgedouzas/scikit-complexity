"""Pass that makes the requests of update laws vanish on free rows.

The free rows are those of capacity layouts.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....components.laws import BaseLaw, UpdateLaw
from ..._utils import vector
from ...problems._declarations_spec import ACTIVE, DeclarationsSpec
from ...problems._edge_search import EdgeSearch
from ...problems._kernel_spec import KernelSpec
from ...types import ReadKind
from .._base import BasePass
from .._compilation import Compilation


class MaskRequestsPass(BasePass):
    """Pass that makes the requests of update laws vanish on free rows.

    The free rows are those of capacity layouts.
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('drawn_apart', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('drawn_apart',)

    def _run(self, inputs: object = None) -> None:
        """Mask the requests of the update laws set apart."""
        compilation = self.compilation_
        declarations = Compilation._get_required(compilation.layout, 'layout')
        compilation.drawn_apart = [
            (spec, self._get_masked_requests(spec, laws, declarations))
            for spec, laws in Compilation._get_required(compilation.drawn_apart, 'drawn_apart')
        ]

    @staticmethod
    def _get_masked_requests(spec: KernelSpec, laws: list[BaseLaw], declarations: DeclarationsSpec) -> list[BaseLaw]:
        """Return update laws that request topology changes, masked on free rows.

        Their probabilities vanish on free rows.
        """
        mask = MaskRequestsPass._mask_of(spec, declarations)
        if mask is None:
            return laws
        return [
            (
                UpdateLaw(law._get_targets()[0], MaskRequestsPass._get_masked_value(law.value, mask))
                if isinstance(law, UpdateLaw)
                else law
            )
            for law in laws
        ]

    @staticmethod
    def _get_activity_symbol(spec: KernelSpec, group: str | None) -> Expression | None:
        """Return the symbol of the activity of the rows of a capacity group.

        The spec reads it per member, and it is `None` otherwise.

        Args:
            spec:
                The kernel spec.

            group:
                The path of the group, or `None`.

        Returns:
            activity:
                The symbol, or `None`.
        """
        name = f'{group}.{ACTIVE}'
        return next(
            (
                item
                for item, reference in spec.references.items()
                if reference.entry.name == name and reference.source is ReadKind.MEMBER
            ),
            None,
        )

    @staticmethod
    def _mask_of(spec: KernelSpec, declarations: DeclarationsSpec) -> Expression | None:
        """Return the activity of the rows a spec runs on.

        The rows are the edges of a changing layer or the members of a capacity group.

        Args:
            spec:
                The kernel spec.

            declarations:
                The declarations of the components.

        Returns:
            mask:
                The symbol of the activity, or `None` for a spec that runs on no capacity rows.
        """
        if isinstance(spec.pairs, EdgeSearch) and spec.pairs.source is not None:
            name = spec.pairs.source.name.rpartition('.')[0]
            return MaskRequestsPass._get_activity_symbol(spec, name)
        if spec.group in declarations.capacity:
            return MaskRequestsPass._get_activity_symbol(spec, spec.group)
        return None

    @staticmethod
    def _get_masked_value(value: object, mask: Expression) -> object:
        """Return a law value that vanishes on the free rows of a capacity layout.

        Args:
            value:
                The law value.

            mask:
                The activity of the rows.

        Returns:
            value:
                The masked value, or the value itself when it is neither symbolic nor a number.
        """
        if isinstance(value, FreeModuleElement):
            return vector(SR(item) * mask for item in value)
        if isinstance(value, Expression | int | float):
            return SR(value) * mask
        return value
