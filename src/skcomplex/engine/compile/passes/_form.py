"""Infer the form of a problem from the classes of the laws of its components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable
from typing import cast

from ....errors import IncompatibleLawsError
from ....types import FormKind
from .._base import BasePass
from .._forms import _COMBINED, _FORMS
from .._laws import _FORMLESS_LAWS


class FormPass(BasePass):
    """Pass that infers the form of the problem from the law classes of its owners."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('kinds',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('form',)

    def _run(self, inputs: object = None) -> None:
        """Infer the form of the problem."""
        compilation = self.compilation_
        compilation.form = self._infer_form(cast('dict[str, set[type]]', compilation.kinds).values())

    @staticmethod
    def _infer_form(owners: Iterable[set[type]]) -> FormKind:
        """Return the form that the law classes of the components of a system determine.

        A combined class, such as a renewal, drops out of the classes when the rest are its companions and no
        component has laws of that class alone.

        Args:
            owners:
                The law classes of every component.

        Returns:
            form:
                The form.

        Raises:
            IncompatibleLawsError:
                No form takes the law classes.
        """
        kinds = FormPass._form_kinds(owners)
        if kinds not in _FORMS:
            found = ', '.join(sorted(kind.__name__ for kind in kinds))
            supported = '; '.join(
                ', '.join(sorted(kind.__name__ for kind in combination)) for combination in _FORMS if combination
            )
            error_msg = f'The system mixes the law kinds {found}. Supported combinations are: {supported}.'
            raise IncompatibleLawsError(error_msg)
        return _FORMS[kinds]

    @staticmethod
    def _form_kinds(owners: Iterable[set[type]]) -> frozenset[type]:
        """Return the law classes that decide the form of components with their own law classes.

        A combined class, such as a renewal, drops out of the classes when the rest are its companions and no component
        has laws of that class alone. A component with laws of that class alone next to components with other laws keeps
        it, so the system does not fit one form and runs as partitions.
        """
        owned = [{kind for kind in kinds if not issubclass(kind, _FORMLESS_LAWS)} for kinds in owners]
        kinds: set[type] = set().union(*owned)
        for kind, companions in _COMBINED.items():
            rest = kinds - {kind}
            if kind not in kinds or not rest:
                continue
            alone = any(item == {kind} for item in owned)
            if not alone and rest <= companions:
                kinds = rest
        return frozenset(kinds)
