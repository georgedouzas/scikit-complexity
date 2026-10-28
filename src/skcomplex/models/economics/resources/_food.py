"""Food security models: the rice production of a country against the requirement of its growing population."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw, Sink, Source, TransferLaw
from ....core.laws.utils import lookup
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model

INPUT_EFFECT = ((0.0, 0.3), (0.25, 0.45), (0.5, 0.58), (0.75, 0.68), (1.0, 0.75), (1.5, 0.8))
LEARNING_EFFECT = ((0.0, 0.0), (0.25, 0.4), (0.5, 0.7), (0.75, 0.9), (1.0, 1.0))
TONNES_PER_KG = 1e-3


def _input_ratio(s: SimpleNamespace) -> Expression:
    """Return the input ratio, the fraction of the recommended inputs that the farmers apply."""
    return s.bio_fraction + (1 - s.bio_fraction) * (s.subsidy + (1 - s.subsidy) * s.farmer_inputs)


def _productivity(s: SimpleNamespace) -> Expression:
    """Return the productivity of the rice land."""
    return s.potential * lookup(_input_ratio(s), INPUT_EFFECT) * s.soil_quality * s.practice


def _production(s: SimpleNamespace) -> Expression:
    """Return the production of milled rice."""
    return s.milling_rate * _productivity(s) * s.rice_area * s.cropping_intensity


def _requirement(s: SimpleNamespace) -> Expression:
    """Return the rice the population requires."""
    return s.consumption * TONNES_PER_KG * s.population


def _self_sufficiency(s: SimpleNamespace) -> Expression:
    """Return the self-sufficiency, the production over the requirement."""
    return _production(s) / _requirement(s)


def _rice_laws(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the development and loss of rice land, the population, and the factors of the rice productivity."""
    chemical = 1 - s.bio_fraction
    return [
        TransferLaw(s.available_land, s.rice_area, s.available_land / s.conversion_time, label='development'),
        TransferLaw(s.rice_area, Sink('urban'), s.discard_rate * s.rice_area, label='discard'),
        TransferLaw(Source('births'), s.population, s.growth_rate * s.population, label='population_growth'),
        RateLaw(s.potential, s.research * (s.potential_limit - s.potential) / s.research_time),
        RateLaw(
            s.soil_quality,
            (1 - s.soil_quality) / s.regeneration_time
            - s.degradation * chemical * _input_ratio(s) * s.cropping_intensity * s.soil_quality,
        ),
        RateLaw(s.practice, lookup(s.training, LEARNING_EFFECT) * (1 - s.practice) / s.learning_time),
        RateLaw(s.training, (s.training_target - s.training) / s.extension_time),
        RateLaw(s.bio_fraction, (s.bio_target - s.bio_fraction) / s.transition_time),
        RateLaw(s.cropping_intensity, (s.intensity_target - s.cropping_intensity) / s.implementation_time),
    ]


class RiceSectorElement(Element):
    """Rice sector element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        available_land: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='land')] = 0.088,
        rice_area: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='land')] = 0.680647,
        population: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='people')] = 18.1024,
        potential: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 6.0,
        soil_quality: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.99,
        practice: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.61,
        training: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        bio_fraction: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        cropping_intensity: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        conversion_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 8.0,
        discard_rate: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0085,
        growth_rate: Annotated[object, ParameterSpec()] = 0.021,
        consumption: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 80.0,
        milling_rate: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.65,
        subsidy: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.67,
        farmer_inputs: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        research: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        potential_limit: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        research_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        regeneration_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        degradation: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.002,
        learning_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        training_target: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        extension_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        bio_target: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        transition_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        intensity_target: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        implementation_time: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.available_land = available_land
        self.rice_area = rice_area
        self.population = population
        self.potential = potential
        self.soil_quality = soil_quality
        self.practice = practice
        self.training = training
        self.bio_fraction = bio_fraction
        self.cropping_intensity = cropping_intensity
        self.conversion_time = conversion_time
        self.discard_rate = discard_rate
        self.growth_rate = growth_rate
        self.consumption = consumption
        self.milling_rate = milling_rate
        self.subsidy = subsidy
        self.farmer_inputs = farmer_inputs
        self.research = research
        self.potential_limit = potential_limit
        self.research_time = research_time
        self.regeneration_time = regeneration_time
        self.degradation = degradation
        self.learning_time = learning_time
        self.training_target = training_target
        self.extension_time = extension_time
        self.bio_target = bio_target
        self.transition_time = transition_time
        self.intensity_target = intensity_target
        self.implementation_time = implementation_time

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the rice sector element."""
        return _rice_laws(s)


@register_model(
    field='economics/resources',
    tags=('food security', 'agriculture', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 50.0, 'dt': 0.25},
    observables=lambda system: [
        ObservableSpec(name, expression, element='rice')
        for name, expression in (
            ('input_ratio', _input_ratio),
            ('productivity', _productivity),
            ('production', _production),
            ('requirement', _requirement),
            ('self_sufficiency', _self_sufficiency),
        )
    ],
)
def food_security(
    *,
    subsidy: float = 0.67,
    bio_target: float = 0.0,
    research: float = 0.0,
    training_target: float = 0.0,
    intensity_target: float = 1.0,
) -> System:
    """Return a food security system, the rice self-sufficiency of a country under subsidy, research, and extension.

    The land suitable for rice is developed into rice area over eight years, and the rice area is discarded for
    housing, roads, and industry at a small fraction per year. The production is the milled share of the
    productivity times the area times the cropping intensity. The productivity is the potential yield, which research
    raises towards a limit, times a table of the input ratio, the soil quality, which chemical fertiliser degrades and
    fallow regenerates, and the farming practice, which training and extension improve through a learning table. The
    input ratio is the share of the recommended fertiliser that the farmers apply, which the subsidy raises, and a
    target share of bio-fertiliser replaces the chemical one without degrading the soil. The requirement is the per
    capita consumption of a population growing exponentially, and the self-sufficiency level is the production over
    the requirement. Land is conserved, so the rice area peaks once the suitable land is developed and then declines,
    and without research and extension the self-sufficiency falls as the population grows. The structure and the
    initial values follow the food security model of Malaysia of Bala, Arshad and Noh (2017, Chapter 10), and the
    table functions and the rates are illustrative, with the land in millions of hectares, the population in
    millions, the productivity in tonnes of paddy per hectare, the rice in millions of tonnes, and time in years.

    Args:
        subsidy:
            The share of the recommended fertiliser that the subsidy pays for.

        bio_target:
            The target share of bio-fertiliser in the inputs.

        research:
            The funding of research for high-yielding varieties, relative to its normal level.

        training_target:
            The target coverage of the farmers by training and extension, from 0 to 1.

        intensity_target:
            The target cropping intensity, the crops per year on the rice area.

    Returns:
        system:
            The model, with the land stocks `rice.available_land` and `rice.rice_area`, the population
            `rice.population`, the factors `rice.potential`, `rice.soil_quality`, `rice.practice`, `rice.training`,
            `rice.bio_fraction`, and `rice.cropping_intensity`, and the flows `rice.development`, `rice.discard`, and
            `rice.population_growth` per year, over years. The observables of the model are `rice.input_ratio`,
            `rice.productivity`, `rice.production`, `rice.requirement`, and `rice.self_sufficiency`.

    Examples:
        >>> from skcomplex.models.economics.resources import food_security
        >>> food_security(subsidy=0.0).elements[0].subsidy
        0.0
    """
    sector = RiceSectorElement(
        'rice',
        subsidy=subsidy,
        bio_target=bio_target,
        research=research,
        training_target=training_target,
        intensity_target=intensity_target,
    )
    return System(elements=[sector])
