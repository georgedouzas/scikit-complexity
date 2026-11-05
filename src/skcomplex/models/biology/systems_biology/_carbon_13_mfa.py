"""Carbon-13 metabolic flux analysis of the TCA example of Antoniewicz et al. (2007)."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

import numpy as np

from ....core.spaces import StaticTime
from ....core.system import System
from ....specs import DeclaredVariableSpec, VisualSpec
from ..._catalog import register_model
from ..common.metabolism import MetabolicNetworkElement, ReactionSpec
from ._labelling import LabellingAnalysis
from ._labelling_specs import AtomMapSpec, FragmentSpec, LabelMeasurementSpec, TracerSpec

BOUND = 300.0
DEVIATION = 0.002
TCA_REACTIONS = {
    'v1': ReactionSpec({'OAC': -1.0, 'Cit': 1.0}, upper=BOUND),
    'v2': ReactionSpec({'Cit': -1.0, 'AKG': 1.0}, upper=BOUND),
    'v3': ReactionSpec({'AKG': -1.0, 'Glu': 1.0}, upper=BOUND),
    'v4': ReactionSpec({'AKG': -1.0, 'Suc': 1.0}, upper=BOUND),
    'v5': ReactionSpec({'Suc': -1.0, 'Fum': 1.0}, upper=BOUND),
    'v6': ReactionSpec({'Fum': -1.0, 'OAC': 1.0}, lower=-BOUND, upper=BOUND),
    'v8': ReactionSpec({'OAC': 1.0}, upper=BOUND),
    'v9': ReactionSpec({'Glu': -1.0}, upper=BOUND),
}
"""The net reactions of the TCA example, with `v6` the fumarase whose exchange the labelling reads."""

TCA_ATOMS = (
    AtomMapSpec('v1', [('AcCoA', 'ab'), ('OAC', 'cdef')], [('Cit', 'fedbac')]),
    AtomMapSpec('v2', [('Cit', 'abcdef')], [('AKG', 'abcde')]),
    AtomMapSpec('v3', [('AKG', 'abcde')], [('Glu', 'abcde')]),
    AtomMapSpec('v4', [('AKG', 'abcde')], [('Suc', 'bcde')]),
    AtomMapSpec('v5', [('Suc', 'abcd')], [('Fum', 'abcd')]),
    AtomMapSpec('v6', [('Fum', 'abcd')], [('OAC', 'abcd')]),
    AtomMapSpec('v8', [('Asp', 'abcd')], [('OAC', 'abcd')]),
)
TCA_TRUE_FLUXES: dict[str, float | tuple[float, float]] = {
    'v1': 100.0,
    'v2': 100.0,
    'v3': 50.0,
    'v4': 50.0,
    'v5': 50.0,
    'v6': (125.0, 75.0),
    'v8': 50.0,
    'v9': 50.0,
}
TCA_NET_FLUXES = {'v1': 100.0, 'v2': 100.0, 'v3': 50.0, 'v4': 50.0, 'v5': 50.0, 'v6': 50.0, 'v8': 50.0, 'v9': 50.0}
TCA_SYMMETRIC = ('Suc', 'Fum')
TCA_OBJECTIVE: Mapping[str, float] = {}
GLUTAMATE = FragmentSpec('Glu', (1, 2, 3, 4, 5))
ASPARTATE = TracerSpec('Asp', {'0000': 1.0})
PUBLISHED_TRACERS = (TracerSpec('AcCoA', {'00': 0.5, '01': 0.25, '11': 0.25}), ASPARTATE)
PARALLEL_TRACERS = {
    'one': (TracerSpec('AcCoA', {'00': 0.5, '10': 0.5}), ASPARTATE),
    'two': (TracerSpec('AcCoA', {'00': 0.5, '11': 0.5}), ASPARTATE),
}


class TcaNetworkElement(MetabolicNetworkElement):
    """The TCA example of Antoniewicz et al. (2007).

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for the bars of its program.

        objective:
            The coefficient of every reaction in the objective, or `None` for `default_objective`.

        parsimonious:
            Whether to minimise the total absolute flux after the objective.

        infeasible:
            The infeasibility policy of the program.

        name:
            The name of the network.

        reactions:
            The reactions by name.

        default_objective:
            The objective of the network when `objective` is `None`.

        variables:
            The declared states and parameters that replace the ones of the reactions, by name, or `None` for none.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import carbon_13_mfa
        >>> carbon_13_mfa().elements[0].get_value('upper_v1')
        300.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        objective: Mapping[str, float] | None = None,
        parsimonious: bool = False,
        infeasible: str = 'raise',
        name: str | None = 'TcaNetwork',
        reactions: Mapping[str, ReactionSpec] = TCA_REACTIONS,
        default_objective: Mapping[str, float] = TCA_OBJECTIVE,
        variables: Mapping[str, DeclaredVariableSpec] | None = None,
    ) -> None:
        super().__init__(
            label,
            visual,
            objective=objective,
            parsimonious=parsimonious,
            infeasible=infeasible,
            name=name,
            reactions=reactions,
            default_objective=default_objective,
            variables=variables,
        )


@register_model(
    field='biology/systems_biology',
    tags=('carbon-13 flux analysis', 'isotope labelling', 'elementary metabolite units', 'metabolism', 'static'),
    references=('Raman 2021', 'Antoniewicz et al. 2007', 'Antoniewicz et al. 2006'),
    defaults={'until': 0.0},
)
def carbon_13_mfa(network: MetabolicNetworkElement | None = None) -> System:
    """Return a metabolic network that `LabellingAnalysis` fits to labelling data, the TCA example by default.

    Labelled atoms of a tracer spread through the network along its atom transitions, so the measured mass isotopomer
    distributions of its products carry the fluxes (Raman 2021, Section 9.7). `tca_labelling_analysis` gives the
    analysis of the TCA example of Antoniewicz et al. (2007) with data simulated from its known fluxes.

    Args:
        network:
            The metabolic network, or `None` for the TCA example.

    Returns:
        system:
            The system.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import carbon_13_mfa
        >>> carbon_13_mfa().elements[0].label
        'tca'
    """
    return System(elements=[TcaNetworkElement('tca') if network is None else network], time=StaticTime())


def tca_labelling_analysis(
    kind: str = 'estimation',
    *,
    noise: float = DEVIATION,
    random_state: int = 0,
    starts: int = 5,
    intervals: str = 'linear',
) -> LabellingAnalysis:
    """Return the labelling analysis of the TCA example, with data simulated from its known fluxes plus seeded noise.

    Args:
        kind:
            `'simulation'` for the published tracer at the true fluxes, or `'estimation'` for a fit to two parallel
            tracer experiments, 50% [1-13C] and 50% [1,2-13C] acetyl-CoA, with `v1` fixed at 100.

        noise:
            The standard deviation of the noise of the data.

        random_state:
            The seed of the noise and of the starts.

        starts:
            The number of feasible starts of the fit.

        intervals:
            `'linear'` or `'profile'`.

    Returns:
        analysis:
            The analysis, not checked yet.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import carbon_13_mfa, tca_labelling_analysis
        >>> analysis = tca_labelling_analysis('simulation').prepare(carbon_13_mfa()).run()
        >>> [round(float(item), 4) for item in analysis.distributions_['published']['Glu']]
        [0.3464, 0.2695, 0.2708, 0.0807, 0.0286, 0.0039]
    """
    if kind == 'simulation':
        return LabellingAnalysis(
            'simulation',
            atoms=TCA_ATOMS,
            tracers={'published': PUBLISHED_TRACERS},
            fragments=(GLUTAMATE,),
            fluxes=TCA_TRUE_FLUXES,
            symmetric=TCA_SYMMETRIC,
        )
    truth = LabellingAnalysis(
        'simulation',
        atoms=TCA_ATOMS,
        tracers=PARALLEL_TRACERS,
        fragments=(GLUTAMATE,),
        fluxes=TCA_TRUE_FLUXES,
        symmetric=TCA_SYMMETRIC,
    )
    simulated = truth.prepare(carbon_13_mfa()).run().distributions_
    rng = np.random.default_rng(random_state)
    measurements = []
    for experiment, values in simulated.items():
        clean = values[GLUTAMATE.label]
        noisy = clean + rng.normal(0.0, noise, clean.size) if noise else clean
        measurements.append(LabelMeasurementSpec(experiment, GLUTAMATE.label, noisy, DEVIATION))
    return LabellingAnalysis(
        'estimation',
        atoms=TCA_ATOMS,
        tracers=PARALLEL_TRACERS,
        fragments=(GLUTAMATE,),
        measurements=measurements,
        measured_fluxes={'v1': (100.0, 0.0)},
        symmetric=TCA_SYMMETRIC,
        starts=starts,
        intervals=intervals,
        random_state=random_state,
    )
