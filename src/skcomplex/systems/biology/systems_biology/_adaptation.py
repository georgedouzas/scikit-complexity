"""Systems biology systems: three-enzyme circuits that adapt to a step of their input."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.components.interactions import BaseInteraction
from ....core.components.system import System
from ....core.errors import ValidationError
from ..._catalog import register_system
from ._enzyme_node import EnzymeNodeElement
from ._enzyme_regulation import EnzymeRegulationInteraction
from ._enzyme_stimulus import EnzymeStimulusInteraction

NODES = ('A', 'B', 'C')
NEGATIVE_FEEDBACK_BUFFER = ((0, -1, 0), (0, 0, 1), (1, 0, 0))
RATES = ((0.419, 0.3, 5.061), (7.809, 9.355, 4.852), (0.735, 1.176, 1.928), (7.282, 0.182, 7.914))
CONSTANTS = ((0.1633, 2.5556, 0.0323), (0.0154, 0.3059, 0.0641), (4.3698, 0.0021, 0.3178), (0.0379, 0.0202, 1.802))
BASAL_RATES = (3.092, 0.712, 0.337)
BASAL_CONSTANTS = (0.143, 0.0031, 0.1074)


@register_system(
    field='biology/systems_biology',
    tags=('adaptation', 'network motifs', 'negative feedback', 'incoherent feedforward', 'ode'),
    references=('Ma et al. 2009', 'Raman 2021'),
    defaults={'until': 800.0, 'dt': 1.0},
)
def enzyme_adaptation_circuit(
    links: Sequence[Sequence[int]] = NEGATIVE_FEEDBACK_BUFFER,
    *,
    k: Sequence[Sequence[float]] = RATES,
    K: Sequence[Sequence[float]] = CONSTANTS,
    k_basal: Sequence[float] = BASAL_RATES,
    K_basal: Sequence[float] = BASAL_CONSTANTS,
    level: float = 0.5,
    change: float = 0.1,
    at: float = 300.0,
) -> System:
    """Return a circuit of three enzymes A, B, and C of Ma et al., whose output C may adapt to a step of the input.

    The input `I` activates A, and `links[i][j]`, 1, -1, or 0, says whether enzyme `j` activates, deactivates, or
    leaves enzyme `i`, with Michaelis-Menten kinetics of the rate `k[j][i]` and the constant `K[j][i]`, the last row of
    `k` and `K` for the input. A node with no activator gets a basal activating enzyme and one with no deactivator a
    basal deactivating enzyme, both of concentration 0.5 with the constants `k_basal` and `K_basal`. The input steps
    from `level` by `change` at the time `at`. Ma et al. enumerated the 16,038 topologies of three nodes whose input
    reaches C and found that adaptation, a sensitive transient followed by a precise return, needs a negative feedback
    loop with a buffer node or an incoherent feedforward loop with a proportioner node. The default is the negative
    feedback loop A to C to B, B deactivating A, with a parameter set that adapts: the buffer B works in saturation, so
    its steady state fixes the output.

    Args:
        links:
            The signs of the regulations, one row per regulated node and one column per regulating node.

        k:
            The catalytic rates, one row per regulating node and the input last, one column per regulated node.

        K:
            The Michaelis constants, in the layout of `k`.

        k_basal:
            The catalytic rates of the basal enzymes of every node.

        K_basal:
            The Michaelis constants of the basal enzymes of every node.

        level:
            The input before the step.

        change:
            The step of the input.

        at:
            The time of the step.

    Returns:
        system:
            The system, with the active fractions `A.X`, `B.X`, and `C.X`.

    Raises:
        ValidationError:
            The links are not a three by three matrix of 1, -1, and 0.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import enzyme_adaptation_circuit
        >>> [interaction.label for interaction in enzyme_adaptation_circuit().interactions][:3]
        ['input', 'B_A', 'C_B']
    """
    signs = np.asarray(links)
    if signs.shape != (3, 3) or not np.isin(signs, (-1, 0, 1)).all():
        error_msg = f'The links must be a 3 by 3 matrix of 1, -1, and 0, not {links!r}.'
        raise ValidationError(error_msg)
    rates, constants = np.asarray(k, dtype=float), np.asarray(K, dtype=float)
    interactions: list[BaseInteraction] = [
        EnzymeStimulusInteraction(
            'input',
            target='A',
            k=float(rates[3, 0]),
            K=float(constants[3, 0]),
            level=level,
            change=change,
            at=at,
        ),
    ]
    for target, source in zip(*np.nonzero(signs), strict=True):
        interactions.append(
            EnzymeRegulationInteraction(
                f'{NODES[source]}_{NODES[target]}',
                target=NODES[target],
                source=NODES[source],
                sign=int(signs[target, source]),
                k=float(rates[source, target]),
                K=float(constants[source, target]),
            ),
        )
    for index, node in enumerate(NODES):
        activated = index == 0 or (signs[index] > 0).any()
        for sign, present in ((1, activated), (-1, (signs[index] < 0).any())):
            if not present:
                interactions.append(
                    EnzymeStimulusInteraction(
                        f'basal_{"activation" if sign > 0 else "deactivation"}_{node}',
                        target=node,
                        sign=sign,
                        k=float(k_basal[index]),
                        K=float(K_basal[index]),
                    ),
                )
    return System(elements=[EnzymeNodeElement(node) for node in NODES], interactions=interactions)
