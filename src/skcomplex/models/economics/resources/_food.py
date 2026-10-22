"""Food security models: the rice production of a country against the requirement of its growing population."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.model import Auxiliary, Element, Law, Rate, Sink, Source, Symbols, System, Transfer, lookup
from ....core.runtime import register_model

INPUT_EFFECT = ((0.0, 0.3), (0.25, 0.45), (0.5, 0.58), (0.75, 0.68), (1.0, 0.75), (1.5, 0.8))
LEARNING_EFFECT = ((0.0, 0.0), (0.25, 0.4), (0.5, 0.7), (0.75, 0.9), (1.0, 1.0))
TONNES_PER_KG = 1e-3


def _rice_laws(s: Symbols) -> list[Law]:
    """Return the development and loss of rice land, the population, and the factors of the rice productivity."""
    chemical = 1 - s.bio_fraction
    input_ratio = Auxiliary('input_ratio', s.bio_fraction + chemical * (s.subsidy + (1 - s.subsidy) * s.farmer_inputs))
    productivity = Auxiliary(
        'productivity',
        s.potential * lookup(input_ratio.symbol, INPUT_EFFECT) * s.soil_quality * s.practice,
    )
    production = Auxiliary(
        'production',
        s.milling_rate * productivity.symbol * s.rice_area * s.cropping_intensity,
    )
    requirement = Auxiliary('requirement', s.consumption * TONNES_PER_KG * s.population)
    return [
        input_ratio,
        productivity,
        production,
        requirement,
        Auxiliary('self_sufficiency', production.symbol / requirement.symbol),
        Transfer(s.available_land, s.rice_area, s.available_land / s.conversion_time, label='development'),
        Transfer(s.rice_area, Sink('urban'), s.discard_rate * s.rice_area, label='discard'),
        Transfer(Source('births'), s.population, s.growth_rate * s.population, label='population_growth'),
        Rate(s.potential, s.research * (s.potential_limit - s.potential) / s.research_time),
        Rate(
            s.soil_quality,
            (1 - s.soil_quality) / s.regeneration_time
            - s.degradation * chemical * input_ratio.symbol * s.cropping_intensity * s.soil_quality,
        ),
        Rate(s.practice, lookup(s.training, LEARNING_EFFECT) * (1 - s.practice) / s.learning_time),
        Rate(s.training, (s.training_target - s.training) / s.extension_time),
        Rate(s.bio_fraction, (s.bio_target - s.bio_fraction) / s.transition_time),
        Rate(s.cropping_intensity, (s.intensity_target - s.cropping_intensity) / s.implementation_time),
    ]


RiceSector = Element.define(
    'RiceSector',
    states={
        'available_land': state(default=0.088, domain=Domain.NONNEGATIVE, quantity='land'),
        'rice_area': state(default=0.680647, domain=Domain.NONNEGATIVE, quantity='land'),
        'population': state(default=18.1024, domain=Domain.NONNEGATIVE, quantity='people'),
        'potential': state(default=6.0, domain=Domain.NONNEGATIVE),
        'soil_quality': state(default=0.99, domain=Domain.NONNEGATIVE),
        'practice': state(default=0.61, domain=Domain.NONNEGATIVE),
        'training': state(default=0.0, domain=Domain.NONNEGATIVE),
        'bio_fraction': state(default=0.0, domain=Domain.NONNEGATIVE),
        'cropping_intensity': state(default=1.0, domain=Domain.NONNEGATIVE),
    },
    parameters={
        'conversion_time': parameter(default=8.0, domain=Domain.POSITIVE),
        'discard_rate': parameter(default=0.0085, domain=Domain.NONNEGATIVE),
        'growth_rate': parameter(default=0.021, domain=Domain.REAL),
        'consumption': parameter(default=80.0, domain=Domain.NONNEGATIVE),
        'milling_rate': parameter(default=0.65, domain=Domain.NONNEGATIVE),
        'subsidy': parameter(default=0.67, domain=Domain.NONNEGATIVE),
        'farmer_inputs': parameter(default=0.3, domain=Domain.NONNEGATIVE),
        'research': parameter(default=0.0, domain=Domain.NONNEGATIVE),
        'potential_limit': parameter(default=10.0, domain=Domain.POSITIVE),
        'research_time': parameter(default=20.0, domain=Domain.POSITIVE),
        'regeneration_time': parameter(default=10.0, domain=Domain.POSITIVE),
        'degradation': parameter(default=0.002, domain=Domain.NONNEGATIVE),
        'learning_time': parameter(default=5.0, domain=Domain.POSITIVE),
        'training_target': parameter(default=0.0, domain=Domain.NONNEGATIVE),
        'extension_time': parameter(default=5.0, domain=Domain.POSITIVE),
        'bio_target': parameter(default=0.0, domain=Domain.NONNEGATIVE),
        'transition_time': parameter(default=10.0, domain=Domain.POSITIVE),
        'intensity_target': parameter(default=1.0, domain=Domain.POSITIVE),
        'implementation_time': parameter(default=10.0, domain=Domain.POSITIVE),
    },
    laws=_rice_laws,
)


@register_model(
    field='economics/resources',
    tags=('food security', 'agriculture', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 50.0, 'dt': 0.25},
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
            `rice.bio_fraction`, and `rice.cropping_intensity`, the auxiliaries `rice.input_ratio`,
            `rice.productivity`, `rice.production`, `rice.requirement`, and `rice.self_sufficiency`, and the flows
            `rice.development`, `rice.discard`, and `rice.population_growth` per year, over years.

    Examples:
        >>> from skcomplex.models.economics.resources import food_security
        >>> food_security(subsidy=0.0).elements[0].subsidy
        0.0
    """
    sector = RiceSector(
        'rice',
        subsidy=subsidy,
        bio_target=bio_target,
        research=research,
        training_target=training_target,
        intensity_target=intensity_target,
    )
    return System(components=[sector])
