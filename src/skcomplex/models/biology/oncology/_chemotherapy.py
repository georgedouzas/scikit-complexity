"""Oncology models: tumour, immune, and normal cells under chemotherapy, with resistance, delays, and metastasis."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.interactions import BaseInteraction, RateInteraction, ReactionInteraction
from ....core.laws.utils import clip, delayed
from ....core.system import System
from ..._catalog import register_model
from ..common.kinetics import MichaelisMentenInteraction
from ..common.pharmacokinetics import DosingInteraction
from ._metastasis_chemotherapy import MetastasisChemotherapyElement
from ._multidrug_resistance import MultidrugResistanceElement
from ._resistance_switching import ResistanceSwitchingElement
from ._resting_hunting_chemotherapy import RestingHuntingChemotherapyElement
from ._tumour_immune_chemotherapy import TumourImmuneChemotherapyElement

BOOK = 'Padmanabhan, Meskin, and Al Moustafa 2021'


DRUGS = (1, 2, 3)


RESISTANCE_CLASSES = {
    (): 'S',
    (1,): 'R1',
    (2,): 'R2',
    (3,): 'R3',
    (1, 2): 'R12',
    (1, 3): 'R13',
    (2, 3): 'R23',
    (1, 2, 3): 'R123',
}


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'ode', 'tumour-immune', 'competition'),
    references=(BOOK, 'de Pillis and Radunskaya 2003'),
    defaults={'until': 150.0, 'dt': 0.5},
)
def de_pillis_radunskaya(
    E: float = 0.15,
    A: float = 0.25,
    N: float = 1.0,
    *,
    u: float = 0.0,
    s: float = 0.33,
    r1: float = 1.5,
    a2: float = 0.3,
) -> System:
    """Return the de Pillis-Radunskaya model of immune, tumour, and normal cells under a constant drug infusion.

    The cells are normalised so that the carrying capacity of normal cells is 1, and time is in days. Untreated, a
    tumour that starts at 0.25 escapes the immune cells and settles at the high equilibrium where it outnumbers the
    normal cells. A constant infusion `u = 1` clears it, while the normal cells stay above 0.7. The parameters are those
    of Table 3.3 of Padmanabhan et al. The book's initial tumour of `1e-5` lies in the basin of the tumour-free
    equilibrium, which is stable since `r1 < c2 s / d1 + c3`, so the defaults are the initial conditions of de Pillis
    and Radunskaya 2003, `E = 0.15`, `A = 0.25`, `N = 1`. The book prints no infusion schedule, so the infusion is a
    constant rate `u` per day, untreated by default.

    Args:
        E:
            The initial immune cells.

        A:
            The initial tumour cells.

        N:
            The initial normal cells.

        u:
            The constant infusion rate of the drug.

        s:
            The influx of immune cells.

        r1:
            The growth rate of the tumour.

        a2:
            The fraction of tumour cells the drug kills.

    Returns:
        system:
            The model, with the cells `patient.E`, `patient.A`, `patient.N` and the drug `patient.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import de_pillis_radunskaya
        >>> de_pillis_radunskaya(u=1.0).elements[0].u
        1.0
    """
    return System(elements=[TumourImmuneChemotherapyElement('patient', E=E, A=A, N=N, u=u, s=s, r1=r1, a2=a2)])


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'ode', 'drug resistance', 'toxicity'),
    references=(BOOK, 'Tse et al. 2007'),
    defaults={'until': 300.0, 'dt': 1.0},
)
def multidrug_resistance(
    S: float = 4.605e11,
    *,
    r: float = 0.0099,
    u1: float = 10.0,
    u2: float = 10.0,
    u3: float = 10.0,
) -> System:
    """Return a tumour of eight populations resistant to subsets of three drugs, given by constant infusions.

    Each drug kills the populations sensitive to it once its concentration exceeds a threshold, so the drugs clear the
    sensitive cells while the cells resistant to all three keep growing. Time is in days, the cells are counted, and the
    drugs are in dosage units D. The parameters are those of Table 3.4 of Padmanabhan et al. The book prints the
    triple-resistant population without its own growth, `dR123/dt = r (m1 R23 + m2 R13 + m3 R12)`, against its
    assumption that all populations grow at the same rate, so the model adds the growth `r R123`. The book prints no
    infusion schedule, so the drugs are constant infusions whose concentrations settle within the book's range of 0 to
    50.

    Args:
        S:
            The initial sensitive cells, all cells resistant to any drug starting at 0.

        r:
            The growth rate of every population.

        u1:
            The infusion rate of drug 1.

        u2:
            The infusion rate of drug 2.

        u3:
            The infusion rate of drug 3.

    Returns:
        system:
            The model, with the populations `tumour.S`, `tumour.R1` to `tumour.R123`, the drugs `tumour.U1` to
            `tumour.U3`, and their toxicities `tumour.TX1` to `tumour.TX3`.

    Examples:
        >>> from skcomplex.models.biology.oncology import multidrug_resistance
        >>> multidrug_resistance(u2=0.0).elements[0].u2
        0.0
    """
    tumour = MultidrugResistanceElement('tumour', S=S, r=r, u1=u1, u2=u2, u3=u3)

    def growth(name: str) -> RateInteraction:
        """Return the division of the cells of a resistance class."""
        return RateInteraction(
            f'growth_{name}',
            target='tumour',
            state=name,
            rate=lambda s: s.target.r * getattr(s.target, name),
        )

    def mutation(name: str, drug: int, mutant: str) -> ReactionInteraction:
        """Return the offspring of a resistance class that gain the resistance to one more drug."""
        return ReactionInteraction(
            f'mutation_{name}_{drug}',
            target='tumour',
            reactants={name: 1},
            products={mutant: 1},
            propensity=lambda s: s.target.r * getattr(s.target, f'm{drug}') * getattr(s.target, name),
        )

    def kill(name: str, drug: int) -> ReactionInteraction:
        """Return the kill of a resistance class by a drug it is sensitive to, above the threshold."""
        return ReactionInteraction(
            f'kill_{name}_{drug}',
            target='tumour',
            reactants={name: 1},
            modifiers={f'U{drug}': 1},
            propensity=lambda s: getattr(s.target, f'a{drug}')
            * clip(getattr(s.target, f'U{drug}') - s.target.Uth, 0)
            * getattr(s.target, name),
        )

    def pharmacokinetics(drug: int) -> list[BaseInteraction]:
        """Return the infusion and clearance of a drug, and the build-up and recovery of its toxicity."""
        return [
            DosingInteraction(
                f'infusion_{drug}',
                target='tumour',
                state=f'U{drug}',
                dose=lambda s: getattr(s.target, f'u{drug}'),
            ),
            ReactionInteraction(
                f'clearance_{drug}',
                target='tumour',
                reactants={f'U{drug}': 1},
                propensity=lambda s: getattr(s.target, f'k{drug}') * getattr(s.target, f'U{drug}'),
            ),
            RateInteraction(
                f'toxicity_{drug}',
                target='tumour',
                state=f'TX{drug}',
                rate=lambda s: getattr(s.target, f'U{drug}'),
            ),
            ReactionInteraction(
                f'recovery_{drug}',
                target='tumour',
                reactants={f'TX{drug}': 1},
                propensity=lambda s: getattr(s.target, f'eta{drug}') * getattr(s.target, f'TX{drug}'),
            ),
        ]

    growths = [growth(name) for name in RESISTANCE_CLASSES.values()]
    sensitivities = [
        (resistances, name, drug)
        for resistances, name in RESISTANCE_CLASSES.items()
        for drug in DRUGS
        if drug not in resistances
    ]
    mutations = [
        mutation(name, drug, RESISTANCE_CLASSES[tuple(sorted((*resistances, drug)))])
        for resistances, name, drug in sensitivities
    ]
    kills = [kill(name, drug) for _, name, drug in sensitivities]
    drugs = [interaction for drug in DRUGS for interaction in pharmacokinetics(drug)]
    return System(elements=[tumour], interactions=[*growths, *mutations, *kills, *drugs])


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'ode', 'drug resistance', 'phenotype switching'),
    references=(BOOK, 'Greene, Sanchez-Tapia, and Sontag 2018', 'Greene, Gevertz, and Sontag 2019'),
    defaults={'until': 400.0, 'dt': 1.0},
)
def drug_resistance_switching(
    S: float = 0.01,
    *,
    u: float = 1.0,
    m1: float = 1e-6,
    m2: float = 0.0,
    alpha_m: float = 0.01,
    alpha_p: float = 0.0,
    m5: float = 0.0,
    r2: float = 0.2,
    r3: float = 0.2,
) -> System:
    """Return sensitive cells and cells resistant by mutation or by a reversible phenotype switch under a dose.

    The drug kills the sensitive cells, so the resistant cells, which grow slower, take the tumour over. Resistance is
    acquired spontaneously at the rates `m1` and `m2` and induced by the drug at the rates `alpha_m` and `alpha_p` per
    unit of dose, and switched cells return to sensitivity at the rate `m5`. Time and the cells are dimensionless, the
    carrying capacity being 1. The book gives values only for the two-population model of Greene, Sanchez-Tapia, and
    Sontag, its Table 3.5, which is this model without switched cells. The defaults are that model, with the
    induced rate `alpha_m = 0.01` of the table and an effective dose of 1, so switching is off until `m2`, `alpha_p`,
    or `m5` is set.

    Args:
        S:
            The initial sensitive cells, the resistant cells starting at 0.

        u:
            The effective dose.

        m1:
            The rate of spontaneous resistance by mutation.

        m2:
            The rate of spontaneous resistance by phenotype switching.

        alpha_m:
            The rate per unit of dose of resistance induced by mutation.

        alpha_p:
            The rate per unit of dose of resistance induced by phenotype switching.

        m5:
            The rate at which switched cells become sensitive again.

        r2:
            The growth rate of mutated cells relative to sensitive cells.

        r3:
            The growth rate of switched cells relative to sensitive cells.

    Returns:
        system:
            The model, with the cells `tumour.S`, `tumour.Rm`, and `tumour.Rp`.

    Examples:
        >>> from skcomplex.models.biology.oncology import drug_resistance_switching
        >>> drug_resistance_switching(alpha_m=0.0).elements[0].alpha_m
        0.0
    """
    tumour = ResistanceSwitchingElement(
        'tumour',
        S=S,
        u=u,
        m1=m1,
        m2=m2,
        alpha_m=alpha_m,
        alpha_p=alpha_p,
        m5=m5,
        r2=r2,
        r3=r3,
    )
    return System(elements=[tumour])


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'dde', 'delay', 'tumour-immune'),
    references=(BOOK, 'Banerjee and Sarkar 2008', 'Borges et al. 2014'),
    defaults={'until': 1500.0, 'dt': 5.0},
)
def delayed_resting_hunting_chemotherapy(
    E: float = 0.01,
    Q: float = 0.48,
    A: float = 0.18,
    *,
    u: float = 0.0,
    tau: float = 45.6,
    a: float = 0.001,
) -> System:
    """Return the tumour-immune model of resting and hunting cells with a conversion delay, under a drug infusion.

    Resting immune cells become hunting cells after the delay `tau`, and hunting cells kill the tumour. Time and the
    cells are dimensionless. The parameters and initial conditions are those of Table 3.6 of Padmanabhan et al., and
    the infusion is a constant rate `u`, untreated by default, within the book's range of 0 to 0.03.

    Args:
        E:
            The initial hunting cells.

        Q:
            The initial resting cells, also their history.

        A:
            The initial tumour cells.

        u:
            The constant infusion rate of the drug.

        tau:
            The delay of the conversion of resting cells to hunting cells.

        a:
            The kill rate of the drug on every population.

    Returns:
        system:
            The model, with the cells `patient.E`, `patient.Q`, `patient.A` and the drug `patient.U`.

    Examples:
        >>> from skcomplex.models.biology.oncology import delayed_resting_hunting_chemotherapy
        >>> delayed_resting_hunting_chemotherapy(tau=10.0).elements[0].tau
        10.0
    """
    patient = RestingHuntingChemotherapyElement('patient', E=E, Q=Q, A=A, u=u, tau=tau, a=a)
    drug_effects = [
        interaction
        for name in ('E', 'Q', 'A')
        for interaction in (
            MichaelisMentenInteraction(
                f'drug_kill_{name}',
                target='patient',
                reactants={name: 1},
                substrate=name,
                vmax=lambda s: s.target.a,
                km=lambda s: s.target.bd,
                modifiers={'U': 1},
            ),
            MichaelisMentenInteraction(
                f'drug_uptake_{name}',
                target='patient',
                reactants={'U': 1},
                substrate=name,
                vmax=lambda s: s.target.g,
                km=lambda s: s.target.bd,
                modifiers={'U': 1},
            ),
        )
    ]
    interactions = [
        ReactionInteraction(
            'recruitment',
            target='patient',
            reactants={'Q': 1},
            products={'E': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.m1 * s.target.E * delayed(s.target.Q, s.target.tau),
        ),
        ReactionInteraction(
            'effector_death',
            target='patient',
            reactants={'E': 1},
            propensity=lambda s: s.target.d1 * s.target.E,
        ),
        ReactionInteraction(
            'effector_inactivation',
            target='patient',
            reactants={'E': 1},
            modifiers={'A': 1},
            propensity=lambda s: s.target.c1 * s.target.A * s.target.E,
        ),
        RateInteraction(
            'resting_growth',
            target='patient',
            state='Q',
            rate=lambda s: s.target.r1 * s.target.Q * (1 - s.target.b1 * s.target.Q),
        ),
        RateInteraction(
            'tumour_growth',
            target='patient',
            state='A',
            rate=lambda s: s.target.r2 * s.target.A * (1 - s.target.b2 * s.target.A),
        ),
        ReactionInteraction(
            'tumour_kill',
            target='patient',
            reactants={'A': 1},
            modifiers={'E': 1},
            propensity=lambda s: s.target.c2 * s.target.A * s.target.E,
        ),
        DosingInteraction('infusion', target='patient', state='U', dose=lambda s: s.target.u),
        ReactionInteraction(
            'clearance',
            target='patient',
            reactants={'U': 1},
            propensity=lambda s: s.target.k * s.target.U,
        ),
        *drug_effects,
    ]
    return System(elements=[patient], interactions=interactions)


@register_model(
    field='biology/oncology',
    tags=('chemotherapy', 'dde', 'delay', 'metastasis'),
    references=(BOOK, 'Pinho, Freedman, and Nani 2002'),
    defaults={'until': 10.0, 'dt': 0.05},
)
def delayed_metastasis(
    Np: float = 1460.0,
    Ap: float = 165.0,
    Ns: float = 1750.0,
    As: float = 0.0,
    *,
    up: float = 0.0,
    us: float = 0.0,
    tau: float = 1.0,
) -> System:
    """Return normal and tumour cells at a primary and a secondary site, with a delayed migration of tumour cells.

    Untreated, the faster tumour cells outcompete the normal cells at the primary site, and the cells that migrate seed
    a tumour that does the same at the secondary site. The parameters are those of Table 3.7 of Padmanabhan et al.,
    with time in days, and the initial cells are at the upper ends of the book's ranges with the secondary site free
    of tumour. The book prints the kill at the secondary site with the infusion `us` in place of the concentration
    `Us`, so the model uses `Us` as at the primary site. The infusions are constant rates, untreated by default, within
    the book's range of 0 to 650.

    Args:
        Np:
            The initial normal cells at the primary site.

        Ap:
            The initial tumour cells at the primary site, also their history.

        Ns:
            The initial normal cells at the secondary site.

        As:
            The initial tumour cells at the secondary site.

        up:
            The infusion rate of the drug at the primary site.

        us:
            The infusion rate of the drug at the secondary site.

        tau:
            The delay of the migration.

    Returns:
        system:
            The model, with the cells `patient.Np`, `patient.Ap`, `patient.Ns`, `patient.As` and the drugs
            `patient.Up` and `patient.Us`.

    Examples:
        >>> from skcomplex.models.biology.oncology import delayed_metastasis
        >>> delayed_metastasis(up=650.0).elements[0].up
        650.0
    """
    patient = MetastasisChemotherapyElement('patient', Np=Np, Ap=Ap, Ns=Ns, As=As, up=up, us=us, tau=tau)

    def population(cells: str, other: str, drug: str, index: int, uptake: str) -> list[BaseInteraction]:
        """Return the growth, the competition, the drug kill, and the drug uptake of the cells of a site."""
        return [
            RateInteraction(
                f'growth_{cells}',
                target='patient',
                state=cells,
                rate=lambda s: getattr(s.target, f'r{index}')
                * getattr(s.target, cells)
                * (1 - getattr(s.target, f'b{index}') * getattr(s.target, cells)),
            ),
            ReactionInteraction(
                f'competition_{cells}',
                target='patient',
                reactants={cells: 1},
                modifiers={other: 1},
                propensity=lambda s: getattr(s.target, f'c{index}')
                * getattr(s.target, cells)
                * getattr(s.target, other),
            ),
            MichaelisMentenInteraction(
                f'drug_kill_{cells}',
                target='patient',
                reactants={cells: 1},
                substrate=cells,
                vmax=lambda s: getattr(s.target, f'a{index}'),
                km=lambda s: s.target.bc,
                modifiers={drug: 1},
            ),
            MichaelisMentenInteraction(
                f'drug_uptake_{cells}',
                target='patient',
                reactants={drug: 1},
                substrate=cells,
                vmax=lambda s: getattr(s.target, uptake),
                km=lambda s: s.target.bc,
                modifiers={drug: 1},
            ),
        ]

    def drug(name: str, dose: str) -> list[BaseInteraction]:
        """Return the infusion and the clearance of the drug at a site."""
        return [
            DosingInteraction(f'infusion_{name}', target='patient', state=name, dose=lambda s: getattr(s.target, dose)),
            ReactionInteraction(
                f'clearance_{name}',
                target='patient',
                reactants={name: 1},
                propensity=lambda s: s.target.k * getattr(s.target, name),
            ),
        ]

    interactions = [
        *population('Np', 'Ap', 'Up', 1, 'g1'),
        *population('Ap', 'Np', 'Up', 2, 'g2'),
        *population('Ns', 'As', 'Us', 3, 'g3'),
        *population('As', 'Ns', 'Us', 4, 'g4'),
        *drug('Up', 'up'),
        *drug('Us', 'us'),
    ]
    interactions += [
        ReactionInteraction(
            'emigration',
            target='patient',
            reactants={'Ap': 1},
            propensity=lambda s: s.target.delta * s.target.Ap,
        ),
        RateInteraction(
            'arrival',
            target='patient',
            state='As',
            rate=lambda s: s.target.epsilon * s.target.delta * delayed(s.target.Ap, s.target.tau),
        ),
    ]
    return System(elements=[patient], interactions=interactions)
