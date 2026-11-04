"""Convert metabolic network elements to and from COBRApy models, and read and write their standard formats."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import os
import sys
from pathlib import Path
from types import ModuleType

from .....errors import ValidationError
from ...._extras import import_extra
from ._metabolic_network import MetabolicNetworkElement, metabolic_network_type
from ._reaction_spec import ReactionSpec

SOLVER = 'hybrid'
BUNDLED_MODEL = 'textbook'
DEFAULT_COMPARTMENT = 'c'
BIGG_URL = 'https://bigg.ucsd.edu/static/models/'


def load_extra(owner: str) -> ModuleType:
    """Return COBRApy, the first library of the extra `metabolism`.

    Args:
        owner:
            The name of the class, function, or model that needs the extra, for the error.

    Returns:
        cobra:
            The module.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.models.biology.common.metabolism import load_extra
        >>> load_extra('example').__name__
        'cobra'
    """
    if sys.platform == 'linux':
        # Once GAP has run, loading libSBML the usual way binds it to a library of GAP, and its parser crashes.
        flags = sys.getdlopenflags()
        sys.setdlopenflags(flags | os.RTLD_DEEPBIND)
        try:
            import_extra('libsbml', owner, 'metabolism')
        finally:
            sys.setdlopenflags(flags)
    return import_extra('cobra', owner, 'metabolism')


def from_cobra(model: object, label: str = 'network', *, needed_by: str = 'from_cobra') -> MetabolicNetworkElement:
    """Return a metabolic network element of a COBRApy model, with its bounds, rules, and objective.

    Args:
        model:
            The `cobra.Model`.

        label:
            The label of the element.

        needed_by:
            The name of what needs the extra, for the error.

    Returns:
        element:
            The element, of a type built for the model.

    Raises:
        MissingExtraError:
            The extra is not installed.

        ValidationError:
            The model is not a `cobra.Model`.

    Examples:
        >>> import cobra
        >>> from skcomplex.models.biology.common.metabolism import from_cobra
        >>> model = cobra.Model('one')
        >>> reaction = cobra.Reaction('r', upper_bound=5.0)
        >>> reaction.add_metabolites({cobra.Metabolite('a', compartment='c'): -1.0})
        >>> model.add_reactions([reaction])
        >>> from_cobra(model, 'cell').upper_r
        5.0
    """
    cobra = load_extra(needed_by)
    if not isinstance(model, cobra.Model):
        error_msg = f'from_cobra converts a `cobra.Model`, not a {type(model).__name__}.'
        raise ValidationError(error_msg)
    reactions = {
        reaction.id: ReactionSpec(
            {metabolite.id: float(value) for metabolite, value in reaction.metabolites.items()},
            float(reaction.lower_bound),
            float(reaction.upper_bound),
            reaction.gene_reaction_rule,
        )
        for reaction in model.reactions
    }
    objective = {
        reaction.id: float(reaction.objective_coefficient)
        for reaction in model.reactions
        if reaction.objective_coefficient != 0
    }
    name = ''.join(part.capitalize() for part in str(model.id or 'network').split('_'))
    element_type = metabolic_network_type(f'{MetabolicNetworkElement._identifier(name)}Element', reactions, objective)
    return element_type(label)


def to_cobra(element: MetabolicNetworkElement, *, needed_by: str = 'to_cobra') -> object:
    """Return the COBRApy model of a metabolic network element, with its current bounds, rules, and objective.

    A switch that is off knocks its gene out, and the model solves with the `hybrid` interface, which reaches HiGHS.

    Args:
        element:
            The element.

        needed_by:
            The name of what needs the extra, for the error.

    Returns:
        model:
            The `cobra.Model`.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.models.biology.common.metabolism import ReactionSpec, metabolic_network_type, to_cobra
        >>> reactions = {'r': ReactionSpec({'a': -1.0}, upper=3.0, rule='g')}
        >>> model = to_cobra(metabolic_network_type('OneElement', reactions, {'r': 1.0})('cell'))
        >>> model.reactions.r.upper_bound, model.reactions.r.gene_reaction_rule
        (3.0, 'g')
    """
    cobra = load_extra(needed_by)
    model = cobra.Model(element.label)
    metabolites: dict[str, object] = {}
    for spec in element.reactions.values():
        for name in spec.coefficients:
            compartment = name.rpartition('_')[2] if '_' in name else DEFAULT_COMPARTMENT
            metabolites.setdefault(name, cobra.Metabolite(name, compartment=compartment))
    reactions = []
    for name, spec in element.reactions.items():
        identifier = element._identifier(name)
        reaction = cobra.Reaction(name)
        reaction.bounds = (getattr(element, f'lower_{identifier}'), getattr(element, f'upper_{identifier}'))
        reaction.add_metabolites({metabolites[item]: float(value) for item, value in spec.coefficients.items()})
        reaction.gene_reaction_rule = spec.rule
        reactions.append(reaction)
    model.add_reactions(reactions)
    model.solver = SOLVER
    weights = element.default_objective if element.objective is None else element.objective
    model.objective = {model.reactions.get_by_id(name): float(value) for name, value in weights.items()}
    for gene in type(element).genes_of():
        if getattr(element, element.gene_name(gene)) < 1:
            model.genes.get_by_id(gene).knock_out()
    return model


def load_network(
    source: str | Path = BUNDLED_MODEL,
    label: str = 'network',
    *,
    needed_by: str = 'load_network',
) -> MetabolicNetworkElement:
    """Return the metabolic network element of an SBML or BiGG JSON file, or of a BiGG identifier.

    Args:
        source:
            The path of a `.xml`, `.sbml`, or `.json` file, a BiGG identifier, which COBRApy downloads, or `'textbook'`
            for the E. coli core model it bundles.

        label:
            The label of the element.

        needed_by:
            The name of what needs the extra, for the error.

    Returns:
        element:
            The element.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.models.biology.common.metabolism import load_network
        >>> len(load_network('textbook', 'cell').reactions)
        95
    """
    cobra = load_extra(needed_by)
    path = Path(source)
    if path.suffix == '.json':
        return from_cobra(cobra.io.load_json_model(str(path)), label)
    if path.suffix in {'.xml', '.sbml'}:
        return from_cobra(cobra.io.read_sbml_model(str(path)), label)
    if str(source) == BUNDLED_MODEL:
        return from_cobra(cobra.io.load_model(BUNDLED_MODEL), label)
    base = cobra.io.web.load.BiGGModels

    def initialize(self: object) -> None:
        super(base, self).__init__(url=BIGG_URL)

    repository = type('SecureBiGGModels', (base,), {'__init__': initialize})
    return from_cobra(cobra.io.load_model(str(source), repositories=[repository()]), label)


def save_network(element: MetabolicNetworkElement, path: str | Path) -> None:
    """Write a metabolic network element to an SBML or BiGG JSON file, by the suffix of the path.

    Args:
        element:
            The element.

        path:
            The path of a `.xml`, `.sbml`, or `.json` file.

    Raises:
        MissingExtraError:
            The extra is not installed.

        ValidationError:
            The suffix names no format.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from skcomplex.models.biology.common.metabolism import ReactionSpec, metabolic_network_type, save_network
        >>> network = metabolic_network_type('SavedElement', {'r': ReactionSpec({'a': -1.0})}, {'r': 1.0})('cell')
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     save_network(network, Path(folder) / 'network.json')
        ...     (Path(folder) / 'network.json').exists()
        True
    """
    cobra = load_extra('save_network')
    target = Path(path)
    if target.suffix == '.json':
        cobra.io.save_json_model(to_cobra(element), str(target))
    elif target.suffix in {'.xml', '.sbml'}:
        cobra.io.write_sbml_model(to_cobra(element), str(target))
    else:
        error_msg = f'The path {target} has the suffix {target.suffix!r}. Use `.xml`, `.sbml`, or `.json`.'
        raise ValidationError(error_msg)
