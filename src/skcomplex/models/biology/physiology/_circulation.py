"""Physiology models: lumped circulation of the blood."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.system import System
from ..._catalog import register_model
from ._chamber import ChamberElement
from ._resistive_flow import ResistiveFlowInteraction


@register_model(
    field='biology/physiology',
    tags=('cardiovascular', 'circulation', 'lumped model', 'hybrid'),
    references=('Suga and Sagawa 1974', 'Trzaska 2023'),
    defaults={'until': 8.0, 'dt': 0.02},
)
def cardiovascular(
    *,
    heart_rate: float = 75.0,
    E_max: float = 2.5,
    R_systemic: float = 1.0,
    C_arterial: float = 1.3,
    C_venous: float = 50.0,
    volume: float = 1500.0,
) -> System:
    """Return a closed lumped model of the systemic circulation driven by a left ventricle of varying elastance.

    Blood leaves the ventricle through the aortic valve into the compliant arteries, flows through the systemic
    resistance into the veins, and returns through the mitral valve. The ventricle stiffens in systole, so the model
    beats, the valves open and close, and the total volume of blood stays constant. With the defaults the arterial
    pressure swings between about 80 and 120 mmHg.

    Args:
        heart_rate:
            The number of beats per minute.

        E_max:
            The largest elastance of the ventricle in mmHg/ml.

        R_systemic:
            The systemic resistance in mmHg s/ml.

        C_arterial:
            The arterial compliance in ml/mmHg.

        C_venous:
            The venous compliance in ml/mmHg.

        volume:
            The total volume of blood in ml, split at the start among the ventricle, arteries, and veins.

    Returns:
        system:
            The model, with the volumes `ventricle.V`, `arteries.V`, and `veins.V`.

    Examples:
        >>> from skcomplex.models.biology.physiology import cardiovascular
        >>> sum(chamber.V for chamber in cardiovascular().elements)
        1500.0
    """
    period = 60.0 / heart_rate
    systole = 0.375 * period
    ventricle = ChamberElement(
        'ventricle',
        V=0.1 * volume,
        V0=10.0,
        E_min=0.08,
        E_max=E_max,
        period=period,
        systole=systole,
    )
    arteries = ChamberElement(
        'arteries',
        V=0.3 * volume,
        V0=250.0,
        E_min=1 / C_arterial,
        E_max=1 / C_arterial,
        period=period,
        systole=systole,
    )
    veins = ChamberElement(
        'veins',
        V=0.6 * volume,
        V0=500.0,
        E_min=1 / C_venous,
        E_max=1 / C_venous,
        period=period,
        systole=systole,
    )
    flows = [
        ResistiveFlowInteraction('aortic_valve', source='ventricle', target='arteries', R=0.02, valve=True),
        ResistiveFlowInteraction('systemic', source='arteries', target='veins', R=R_systemic),
        ResistiveFlowInteraction('mitral_valve', source='veins', target='ventricle', R=0.02, valve=True),
    ]
    return System(elements=[ventricle, arteries, veins], interactions=flows)
