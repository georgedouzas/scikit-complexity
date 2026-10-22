"""Process models: stocks of material that flows in, out, and between stages."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.model import Element, Flow, Interaction, Rate, Sink, Source, System, step, where
from ....core.runtime import register_model

INFLOW_START = 5.0

Tub = Element.define(
    'Tub',
    states={'water': state(default=50.0, domain=Domain.NONNEGATIVE, quantity='water')},
    parameters={
        'inflow': parameter(default=5.0, domain=Domain.NONNEGATIVE),
        'outflow': parameter(default=5.0, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Flow(Source(), s.water, s.inflow * step(s.t, INFLOW_START), label='faucet'),
        Flow(s.water, Sink(), where(s.water > 0, s.outflow, 0), label='drain'),
    ],
)

Stage = Element.define(
    'Stage',
    states={'m': state(default=0.0, domain=Domain.NONNEGATIVE)},
    parameters={'feed': parameter(default=0.0, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [Rate(s.m, s.feed)],
)

Transfer = Interaction.define(
    'Transfer',
    roles=['source', 'target'],
    parameters={'rate': parameter(default=0.5, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [Rate(s.source.m, -s.rate * s.source.m), Rate(s.target.m, s.rate * s.source.m)],
)


@register_model(
    field='engineering/processes',
    tags=('stocks and flows', 'system dynamics', 'ode'),
    references=('Meadows 2008',),
    defaults={'until': 10.0, 'dt': 0.5},
)
def bathtub(water: float = 50.0, *, inflow: float = 5.0, outflow: float = 5.0) -> System:
    """Return a bathtub, one stock of water with an inflow from the faucet and an outflow through the drain.

    The drain empties the tub at `outflow` gallons per minute while there is water, and the faucet opens to `inflow`
    gallons per minute after five minutes. With equal flows the level falls for five minutes and then stays constant.

    Args:
        water:
            The initial water in the tub, in gallons.

        inflow:
            The inflow after five minutes, in gallons per minute.

        outflow:
            The outflow, in gallons per minute.

    Returns:
        system:
            The model, with the water `tub.water` in gallons and the flows `tub.faucet` and `tub.drain` in gallons
            per minute, over minutes.

    Examples:
        >>> from skcomplex.models.engineering.processes import bathtub
        >>> bathtub(water=20.0).elements[0].water
        20.0
    """
    return System(components=[Tub('tub', water=water, inflow=inflow, outflow=outflow)])


@register_model(
    field='engineering/processes',
    tags=('separation', 'cascade', 'stocks and flows', 'ode'),
    references=('Barsky 2020',),
    defaults={'until': 50.0, 'dt': 0.5},
)
def cascade_classifier(
    n_stages: int = 10,
    *,
    k: float = 0.6,
    feed_stage: int | None = None,
    feed: float = 1.0,
) -> System:
    """Return a cascade classifier, a column of stages that separates a narrow size class of particles by an air flow.

    Material leaves every stage at unit rate, a fraction `k` upwards and `1 - k` downwards. The top stage releases
    into the fine product and the bottom stage into the coarse product. Material of one size class is fed into one
    stage at a constant rate. At steady state the share of the feed that ends in the fine product is
    `F_f = (1 - ((1 - k) / k)^(z + 1 - i)) / (1 - ((1 - k) / k)^(z + 1))` for `z` stages and feeding at stage `i`
    counted from the top, Barsky's fractional extraction.

    Args:
        n_stages:
            The number of stages `z`.

        k:
            The distribution coefficient, the fraction of material moving up from a stage.

        feed_stage:
            The stage that receives the feed, counted from 1 at the top, or `None` for the middle stage.

        feed:
            The feed rate.

    Returns:
        system:
            The model, with the material `stage_<i>.m` of each stage from the top, and the accumulated products
            `fine.m` and `coarse.m`.

    Examples:
        >>> from skcomplex.models.engineering.processes import cascade_classifier
        >>> len(cascade_classifier(n_stages=4).interactions)
        8
    """
    entry = (n_stages + 1) // 2 if feed_stage is None else feed_stage
    stages = [Stage(f'stage_{index}', feed=feed if index == entry else 0.0) for index in range(1, n_stages + 1)]
    labels = ['fine', *(stage.label for stage in stages), 'coarse']
    interactions = [
        Transfer(f'up_{index}', source=labels[index], target=labels[index - 1], rate=k)
        for index in range(1, n_stages + 1)
    ] + [
        Transfer(f'down_{index}', source=labels[index], target=labels[index + 1], rate=1 - k)
        for index in range(1, n_stages + 1)
    ]
    return System(components=[Stage('fine'), *stages, Stage('coarse')], interactions=interactions)
