"""Pass that makes the requests of update laws vanish on the free rows of capacity layouts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from sage.all__sagemath_symbolics import SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.laws import BaseLaw, UpdateLaw
from ...problems import (
    EdgeSearch,
    Source,
    vector,
)
from .._compilation import Compilation, Laws
from .._declarations import _ACTIVE, Declarations
from .._kernel_spec import KernelSpec
from .._pipeline import BasePass


class MaskRequestsPass(BasePass):
    """Pass that makes the requests of update laws vanish on the free rows of capacity layouts."""

    reads = ('drawn_apart', 'layout')
    writes = ('drawn_apart',)

    def _run(self, compilation: Compilation) -> None:
        """Mask the requests of the update laws set apart.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        compilation.drawn_apart = [
            (spec, self._masked_requests(spec, laws, declarations))
            for spec, laws in cast('Laws', compilation.drawn_apart)
        ]

    @staticmethod
    def _masked_requests(spec: KernelSpec, laws: list[BaseLaw], declarations: Declarations) -> list[BaseLaw]:
        """Return update laws that request topology changes, with probabilities that vanish on free rows."""
        mask = MaskRequestsPass._mask_of(spec, declarations)
        if mask is None:
            return laws
        return [
            UpdateLaw(law._get_targets()[0], MaskRequestsPass._masked_value(law.value, mask))
            if isinstance(law, UpdateLaw)
            else law
            for law in laws
        ]

    @staticmethod
    def _activity_symbol(spec: KernelSpec, group: str | None) -> Expression | None:
        """Return the symbol of the activity of the rows of a capacity group that a spec reads per member, or `None`.

        Args:
            spec:
                The kernel spec.

            group:
                The path of the group, or `None`.

        Returns:
            activity:
                The symbol, or `None`.
        """
        name = f'{group}.{_ACTIVE}'
        return next(
            (
                item
                for item, reference in spec.references.items()
                if reference.entry.name == name and reference.source is Source.MEMBER
            ),
            None,
        )

    @staticmethod
    def _mask_of(spec: KernelSpec, declarations: Declarations) -> Expression | None:
        """Return the activity of the rows a spec runs on, the edges of a changing layer or a capacity group's members.

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
            return MaskRequestsPass._activity_symbol(spec, name)
        if spec.group in declarations.capacity:
            return MaskRequestsPass._activity_symbol(spec, spec.group)
        return None

    @staticmethod
    def _masked_value(value: object, mask: Expression) -> object:
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
