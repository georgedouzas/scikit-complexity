"""Convert flux balances of metabolic networks to and from COBRApy models.

The module also reads and writes their standard formats.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from .....core.components.elements import Element
from .....core.errors import ValidationError
from ...._extras import import_extra
from ._flux_balance import FluxBalanceInteraction
from ._reaction_spec import ReactionSpec

SOLVER = 'hybrid'
BUNDLED_SYSTEM = 'textbook'
DEFAULT_COMPARTMENT = 'c'


def load_extra(owner: str) -> ModuleType:
    """Return COBRApy, the first library of the extra `metabolism`.

    Args:
        owner:
            The name of the class, function, or system that needs the extra, for the error.

    Returns:
        cobra:
            The module.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import load_extra
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


def from_cobra(
    system: object,
    cell: str = 'network',
    *,
    label: str = 'fba',
    needed_by: str = 'from_cobra',
) -> FluxBalanceInteraction:
    """Return the flux balance of a COBRApy model.

    The interaction keeps the reactions, bounds, rules, and objective of the model, and `get_values` gives the values
    of its cell.

    Args:
        system:
            The `cobra.Model`.

        cell:
            The path of the cell whose values hold the fluxes, the bounds, and the switches.

        label:
            The label of the interaction.

        needed_by:
            The name of what needs the extra, for the error.

    Returns:
        network:
            The flux balance of the model.

    Raises:
        MissingExtraError:
            The extra is not installed.

        ValidationError:
            The system is not a `cobra.Model`.

    Examples:
        >>> import cobra
        >>> from skcomplex.systems.biology.common.metabolism import from_cobra
        >>> system = cobra.Model('one')
        >>> reaction = cobra.Reaction('r', upper_bound=5.0)
        >>> reaction.add_metabolites({cobra.Metabolite('a', compartment='c'): -1.0})
        >>> system.add_reactions([reaction])
        >>> from_cobra(system, 'cell').get_values()['upper_r']
        5.0
    """
    cobra = load_extra(needed_by)
    if not isinstance(system, cobra.Model):
        error_msg = f'from_cobra converts a `cobra.Model`, not a {type(system).__name__}.'
        raise ValidationError(error_msg)
    reactions = {
        reaction.id: ReactionSpec(
            {metabolite.id: float(value) for metabolite, value in reaction.metabolites.items()},
            float(reaction.lower_bound),
            float(reaction.upper_bound),
            reaction.gene_reaction_rule,
        )
        for reaction in system.reactions
    }
    objective = {
        reaction.id: float(reaction.objective_coefficient)
        for reaction in system.reactions
        if reaction.objective_coefficient != 0
    }
    name = ''.join(part.capitalize() for part in str(system.id or 'network').split('_'))
    return FluxBalanceInteraction(
        label,
        cell=cell,
        reactions=reactions,
        objective=objective,
        name=FluxBalanceInteraction._get_identifier(name),
    )


def to_cobra(network: FluxBalanceInteraction, cell: Element | None = None, *, needed_by: str = 'to_cobra') -> object:
    """Return the COBRApy model of a flux balance.

    The model keeps the reactions, rules, and objective of the network and the bounds of its cell.

    A switch of the cell that is off knocks its gene out, and the system solves with the `hybrid` interface, which
    reaches HiGHS.

    Args:
        network:
            The flux balance.

        cell:
            The cell whose values hold the bounds and the switches, or `None` for the values the network starts with.

        needed_by:
            The name of what needs the extra, for the error.

    Returns:
        system:
            The `cobra.Model`.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import FluxBalanceInteraction, ReactionSpec, to_cobra
        >>> reactions = {'r': ReactionSpec({'a': -1.0}, upper=3.0, rule='g')}
        >>> system = to_cobra(FluxBalanceInteraction('fba', reactions=reactions, objective={'r': 1.0}))
        >>> system.reactions.r.upper_bound, system.reactions.r.gene_reaction_rule
        (3.0, 'g')
    """
    cobra = load_extra(needed_by)
    given = dict(cell.get_params()['values'] or {}) if cell is not None else {}
    values = network.get_values() | {name: float(np.asarray(value)) for name, value in given.items()}
    system = cobra.Model(network.cell if cell is None else cell.label)
    metabolites: dict[str, object] = {}
    for spec in network.reactions.values():
        for name in spec.coefficients:
            compartment = name.rpartition('_')[2] if '_' in name else DEFAULT_COMPARTMENT
            metabolites.setdefault(name, cobra.Metabolite(name, compartment=compartment))
    reactions = []
    for name, spec in network.reactions.items():
        identifier = network._get_identifier(name)
        reaction = cobra.Reaction(name)
        reaction.bounds = (values[f'lower_{identifier}'], values[f'upper_{identifier}'])
        reaction.add_metabolites({metabolites[item]: float(value) for item, value in spec.coefficients.items()})
        reaction.gene_reaction_rule = spec.rule
        reactions.append(reaction)
    system.add_reactions(reactions)
    system.solver = SOLVER
    system.objective = {system.reactions.get_by_id(name): float(value) for name, value in network.objective.items()}
    for gene in network.get_genes_of():
        if values[network.get_gene_name(gene)] < 1:
            system.genes.get_by_id(gene).knock_out()
    return system


def load_network(
    source: str | Path = BUNDLED_SYSTEM,
    cell: str = 'network',
    *,
    label: str = 'fba',
    needed_by: str = 'load_network',
) -> FluxBalanceInteraction:
    """Return the flux balance of a network in a file or of a BiGG identifier.

    The file is an SBML or BiGG JSON file.

    Args:
        source:
            The path of a `.xml`, `.sbml`, or `.json` file, a BiGG identifier, which COBRApy downloads, or `'textbook'`
            for the E. coli core system it bundles.

        cell:
            The path of the cell whose values hold the fluxes, the bounds, and the switches.

        label:
            The label of the interaction.

        needed_by:
            The name of what needs the extra, for the error.

    Returns:
        network:
            The flux balance of the network.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.systems.biology.common.metabolism import load_network
        >>> len(load_network('textbook', 'cell').reactions)
        95
    """
    cobra = load_extra(needed_by)
    path = Path(source)
    if path.suffix == '.json':
        return from_cobra(cobra.io.load_json_model(str(path)), cell, label=label)
    if path.suffix in {'.xml', '.sbml'}:
        return from_cobra(cobra.io.read_sbml_model(str(path)), cell, label=label)
    if str(source) == BUNDLED_SYSTEM:
        return from_cobra(cobra.io.load_model(BUNDLED_SYSTEM), cell, label=label)
    repository = cobra.io.web.load.BiGGModels()
    repository._url = repository.url.copy_with(scheme='https')
    return from_cobra(cobra.io.load_model(str(source), repositories=[repository]), cell, label=label)


def save_network(network: FluxBalanceInteraction, path: str | Path, cell: Element | None = None) -> None:
    """Write a flux balance to an SBML or BiGG JSON file.

    The format follows the suffix of the path.

    Args:
        network:
            The flux balance.

        path:
            The path of a `.xml`, `.sbml`, or `.json` file.

        cell:
            The cell whose values hold the bounds and the switches, or `None` for the values the network starts with.

    Raises:
        MissingExtraError:
            The extra is not installed.

        ValidationError:
            The suffix names no format.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from skcomplex.systems.biology.common.metabolism import FluxBalanceInteraction, ReactionSpec, save_network
        >>> network = FluxBalanceInteraction('fba', reactions={'r': ReactionSpec({'a': -1.0})}, objective={'r': 1.0})
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     save_network(network, Path(folder) / 'network.json')
        ...     (Path(folder) / 'network.json').exists()
        True
    """
    cobra = load_extra('save_network')
    target = Path(path)
    if target.suffix == '.json':
        cobra.io.save_json_model(to_cobra(network, cell), str(target))
    elif target.suffix in {'.xml', '.sbml'}:
        cobra.io.write_sbml_model(to_cobra(network, cell), str(target))
    else:
        error_msg = f'The path {target} has the suffix {target.suffix!r}. Use `.xml`, `.sbml`, or `.json`.'
        raise ValidationError(error_msg)
