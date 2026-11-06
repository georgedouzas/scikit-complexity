"""Define the patch map spec of the Sugarscape foragers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from ....core.specs import BaseSpec
from ._sugar_patch_view import SugarPatchView


@dataclass(frozen=True)
class _PatchMapSpec(BaseSpec):
    """Patches of a run by cell, which the foragers build once and keep in the storage of the system.

    Args:
        patches:
            The patch of every cell.
    """

    patches: Mapping[tuple[int, int], SugarPatchView]
