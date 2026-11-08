"""Define the partner switch spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class PartnerSwitchSpec(BaseSpec):
    """Partner choice that samples providers and switches to a lower limit.

    The switch happens with some probability. The rule follows chapter 2 of Caiani et al. (2016). Every requester, in
    random order, samples `sample` providers that still offer something. When the lowest limit among them is below the
    limit of its current partner, the requester switches with probability `1 - exp(intensity * (p_new - p_old) /
    p_new)`, and otherwise stays. It matches with its partner at the partner's limit, and the requester's state
    `partner`, when it has one, holds the id of the partner across steps.

    Args:
        sample:
            The number of providers every requester samples.

        intensity:
            The intensity of switching.

        partner:
            The name of the requester state that holds the id of its partner.

    Examples:
        >>> from skcomplex.core.specs import PartnerSwitchSpec
        >>> PartnerSwitchSpec(sample=2, intensity=4.0).intensity
        4.0
    """

    sample: int = 3
    intensity: float = 1.0
    partner: str = 'partner'
