"""Elementary flux modes and extreme pathways of a metabolic network, Raman's toy network by default."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ....core.spaces import StaticTime
from ....core.system import System
from ..._catalog import register_model
from ._flux_balance import ToyNetworkElement


@register_model(
    field='biology/systems_biology',
    tags=('elementary flux modes', 'extreme pathways', 'linear program', 'metabolism', 'static'),
    references=('Raman 2021', 'Schuster and Hilgetag 1994', 'Schilling et al. 2000', 'Gagneur and Klamt 2004'),
    defaults={'until': 0.0},
)
def elementary_flux_modes(knockouts: Sequence[str] = ()) -> System:
    """Return the toy network of Raman whose steady-state cone `ElementaryModeAnalysis` enumerates.

    Every steady-state flux of the network is a non-negative combination of the elementary modes of its cone (Raman
    2021, Section 9.8). The toy network has two modes, through `v1` and through `v2` and `v3`, both with growth yield
    `1/12` on `A`, so the bound 1000 on `v6` limits the growth to 50 whichever route carries the flux.

    Args:
        knockouts:
            The reactions to fix at zero.

    Returns:
        system:
            The system.

    Examples:
        >>> from skcomplex.analysis import ElementaryModeAnalysis
        >>> from skcomplex.models.biology.systems_biology import elementary_flux_modes
        >>> analysis = ElementaryModeAnalysis(normalize='cell.v_b1').prepare(elementary_flux_modes()).run()
        >>> analysis.modes_.shape[0]
        2
    """
    closed = {f'{bound}_{name}': 0.0 for name in knockouts for bound in ('lower', 'upper')}
    return System(elements=[ToyNetworkElement('cell').set_params(**closed)], time=StaticTime())
