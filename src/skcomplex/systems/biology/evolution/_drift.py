"""Evolution systems: allele frequencies under selection and genetic drift in the diffusion limit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.components.elements import GroupElement
from ....core.components.system import System
from ....core.engine.solvers import EulerMaruyamaSolver
from ..._catalog import register_system
from ._allele_frequency import AlleleFrequencyElement


@register_system(
    field='biology/evolution',
    tags=('population genetics', 'genetic drift', 'fixation', 'sde', 'diffusion'),
    references=('Kimura 1955', 'Bertin 2021'),
    defaults={'until': 10.0, 'dt': 0.05, 'solver': EulerMaruyamaSolver(dt=0.001)},
)
def kimura_diffusion(x: float = 0.1, *, Ns: float = 2.0, replicates: int = 500) -> System:
    """Return replicate populations whose allele frequencies diffuse by drift until the allele is lost or fixed.

    Each replicate follows `dx = N s x (1 - x) dt + sqrt(2 x (1 - x)) dW`, whose density obeys Kimura's diffusion
    equation `dp/dt = -N s d/dx [x (1 - x) p] + d^2/dx^2 [x (1 - x) p]`. Every replicate ends at loss or fixation, and
    an allele that starts at the frequency `x` fixes with the probability `(1 - exp(-N s x)) / (1 - exp(-N s))`, or `x`
    for a neutral allele.

    Args:
        x:
            The initial frequency of the allele.

        Ns:
            The scaled selection, the population size times the excess fitness.

        replicates:
            The number of independent replicate populations.

    Returns:
        system:
            The system, with the frequencies `alleles.x` of the replicates.

    Examples:
        >>> from skcomplex.systems.biology.evolution import kimura_diffusion
        >>> kimura_diffusion(replicates=10).elements[0].prepare().size_
        10
    """
    alleles = GroupElement('alleles', element=AlleleFrequencyElement, size=replicates, values={'x': x, 'Ns': Ns})
    return System(elements=[alleles])
