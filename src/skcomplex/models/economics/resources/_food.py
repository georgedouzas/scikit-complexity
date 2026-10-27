"""Food security models: the rice production of a country against the requirement of its growing population."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import BaseLaw, Expression, NumericDomain, parameter, state
from ....core.components import Element
from ....core.laws import RateLaw, Sink, Source, Symbols, TransferLaw, lookup
from ....core.specs import ObservableSpec
from ....core.system import System
from ....engine.runtime import register_model

INPUT_EFFECT = ((0.0, 0.3), (0.25, 0.45), (0.5, 0.58), (0.75, 0.68), (1.0, 0.75), (1.5, 0.8))
LEARNING_EFFECT = ((0.0, 0.0), (0.25, 0.4), (0.5, 0.7), (0.75, 0.9), (1.0, 1.0))
TONNES_PER_KG = 1e-3


def _input_ratio(s: Symbols) -> Expression:
    """Return the input ratio, the fraction of the recommended inputs that the farmers apply."""
    return s.bio_fraction + (1 - s.bio_fraction) * (s.subsidy + (1 - s.subsidy) * s.farmer_inputs)


def _productivity(s: Symbols) -> Expression:
    """Return the productivity of the rice land."""
    return s.potential * lookup(_input_ratio(s), INPUT_EFFECT) * s.soil_quality * s.practice


def _production(s: Symbols) -> Expression:
    """Return the production of milled rice."""
    return s.milling_rate * _productivity(s) * s.rice_area * s.cropping_intensity


def _requirement(s: Symbols) -> Expression:
    """Return the rice the population requires."""
    return s.consumption * TONNES_PER_KG * s.population


def _self_sufficiency(s: Symbols) -> Expression:
    """Return the self-sufficiency, the production over the requirement."""
    return _production(s) / _requirement(s)


def _rice_laws(s: Symbols) -> list[BaseLaw]:
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

    available_land = state(default=0.088, domain=NumericDomain.NONNEGATIVE, quantity='land')
    rice_area = state(default=0.680647, domain=NumericDomain.NONNEGATIVE, quantity='land')
    population = state(default=18.1024, domain=NumericDomain.NONNEGATIVE, quantity='people')
    potential = state(default=6.0, domain=NumericDomain.NONNEGATIVE)
    soil_quality = state(default=0.99, domain=NumericDomain.NONNEGATIVE)
    practice = state(default=0.61, domain=NumericDomain.NONNEGATIVE)
    training = state(default=0.0, domain=NumericDomain.NONNEGATIVE)
    bio_fraction = state(default=0.0, domain=NumericDomain.NONNEGATIVE)
    cropping_intensity = state(default=1.0, domain=NumericDomain.NONNEGATIVE)
    conversion_time = parameter(default=8.0, domain=NumericDomain.POSITIVE)
    discard_rate = parameter(default=0.0085, domain=NumericDomain.NONNEGATIVE)
    growth_rate = parameter(default=0.021, domain=NumericDomain.REAL)
    consumption = parameter(default=80.0, domain=NumericDomain.NONNEGATIVE)
    milling_rate = parameter(default=0.65, domain=NumericDomain.NONNEGATIVE)
    subsidy = parameter(default=0.67, domain=NumericDomain.NONNEGATIVE)
    farmer_inputs = parameter(default=0.3, domain=NumericDomain.NONNEGATIVE)
    research = parameter(default=0.0, domain=NumericDomain.NONNEGATIVE)
    potential_limit = parameter(default=10.0, domain=NumericDomain.POSITIVE)
    research_time = parameter(default=20.0, domain=NumericDomain.POSITIVE)
    regeneration_time = parameter(default=10.0, domain=NumericDomain.POSITIVE)
    degradation = parameter(default=0.002, domain=NumericDomain.NONNEGATIVE)
    learning_time = parameter(default=5.0, domain=NumericDomain.POSITIVE)
    training_target = parameter(default=0.0, domain=NumericDomain.NONNEGATIVE)
    extension_time = parameter(default=5.0, domain=NumericDomain.POSITIVE)
    bio_target = parameter(default=0.0, domain=NumericDomain.NONNEGATIVE)
    transition_time = parameter(default=10.0, domain=NumericDomain.POSITIVE)
    intensity_target = parameter(default=1.0, domain=NumericDomain.POSITIVE)
    implementation_time = parameter(default=10.0, domain=NumericDomain.POSITIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
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
