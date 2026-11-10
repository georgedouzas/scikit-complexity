"""Define the spec of a choice of partner that switches to a lower limit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ....core.specs import BaseSpec


@dataclass(frozen=True)
class PartnerSwitchSpec(BaseSpec):
    """A spec of a choice of partner that may switch to a cheaper provider.

    Every requester, in random order, samples `sample` providers that still offer something. When the lowest limit among
    them is below that of its current partner, it switches with the probability `1 - exp(intensity * (p_new - p_old) /
    p_new)`, and stays otherwise. It matches with its partner at the limit of the partner, and its state `partner` holds
    the id of the partner across steps. The rule follows Caiani et al. (2016, chapter 2).

    Unlike `RandomSearchSpec`, which meets new providers each time, it keeps a partner and switches with friction.
    Choose it when relations persist, such as customers loyal to a supplier.

    It appears in many fields. Households keep their shops, firms keep their banks, and workers keep their employers
    unless a better offer comes, in agent-based models of macroeconomics.

    Args:
        sample:
            The number of providers every requester samples.

        intensity:
            The intensity of switching.

        partner:
            The name of the requester state that holds the id of its partner.

    Examples:
        >>> from skcomplex.systems.general.matching import PartnerSwitchSpec
        >>> PartnerSwitchSpec(sample=2, intensity=4.0).intensity
        4.0
    """

    sample: int = 3
    intensity: float = 1.0
    partner: str = 'partner'
