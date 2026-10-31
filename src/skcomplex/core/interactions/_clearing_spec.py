"""Define the clearing spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import MutableMapping
from dataclasses import dataclass

import numpy as np

from ...specs import BaseSpec
from ._offer import _Offer


@dataclass(frozen=True)
class ClearingSpec(BaseSpec):
    """Offers of one clearing of a matching, with the random generator and the storage that persists across clearings.

    Args:
        requesters:
            The offers of the requesters.

        providers:
            The offers of the providers.

        rng:
            The random generator, such as `model.rng`.

        storage:
            The storage of the matching that persists across clearings, such as `model.storage`.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.interactions import ClearingSpec
        >>> ClearingSpec([], [], np.random.default_rng(0), {}).check().requesters
        []
    """

    requesters: list[_Offer]
    providers: list[_Offer]
    rng: np.random.Generator
    storage: MutableMapping[str, object]
