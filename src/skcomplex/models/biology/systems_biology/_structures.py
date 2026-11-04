"""Read and write molecular structures and trajectories through Biopython and MDTraj."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import gzip
from importlib.resources import files
from pathlib import Path
from typing import TextIO

import numpy as np

from ....errors import ValidationError
from ..._extras import import_extra
from ._structure_spec import StructureSpec
from .types import BiopythonStructureProtocol

STRUCTURE_MODELS = 'fuzzy_oil_drop or early_stage_folding_intermediate'
BUNDLED = ('1L2Y', '1UBQ', '5PTI')
NM_TO_ANGSTROM = 10.0
POLYMER = ' '
FULL_NAME = 4
SUFFIX_PDB = '.pdb'
SUFFIXES_MMCIF = ('.cif', '.mmcif')


def _read_models(handle: TextIO | str, *, is_pdb: bool) -> StructureSpec:
    """Return the polymer atoms of a file read by Biopython's parser, the most occupied of alternate locations."""
    pdb = import_extra('Bio.PDB', STRUCTURE_MODELS, 'structures')
    parser = pdb.PDBParser(QUIET=True) if is_pdb else pdb.MMCIFParser(QUIET=True)
    structure = parser.get_structure('structure', handle)
    coordinates, atoms = [], []
    for model in structure:
        selected = [
            (atom, residue, chain)
            for chain in model
            for residue in chain
            if residue.id[0] == POLYMER
            for atom in residue
        ]
        coordinates.append([atom.coord for atom, _, _ in selected])
        atoms = selected
    return StructureSpec(
        np.asarray(coordinates, dtype=float),
        tuple(atom.get_name() for atom, _, _ in atoms),
        tuple(residue.get_resname() for _, residue, _ in atoms),
        tuple(int(residue.id[1]) for _, residue, _ in atoms),
        tuple(str(chain.id) for _, _, chain in atoms),
    ).check()


def read_structure(source: str | Path) -> StructureSpec:
    """Return the polymer atoms of every model of a bundled entry, or of a PDB or an mmCIF file.

    Args:
        source:
            The name of a bundled entry, `'1L2Y'`, `'1UBQ'`, or `'5PTI'`, or the path of a `.pdb`, `.cif`, or `.mmcif`
            file.

    Returns:
        structure:
            The structure.

    Raises:
        MissingExtraError:
            The extra `structures` is not installed.

        ValidationError:
            The source is no bundled entry and no file of a known format.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import read_structure
        >>> read_structure('1L2Y').coordinates.shape
        (38, 304, 3)
    """
    if source in BUNDLED:
        with gzip.open(str(files(__package__) / f'{source}.cif.gz'), 'rt') as handle:
            return _read_models(handle, is_pdb=False)
    path = Path(source)
    if path.suffix == SUFFIX_PDB or path.suffix in SUFFIXES_MMCIF:
        return _read_models(str(path), is_pdb=path.suffix == SUFFIX_PDB)
    error_msg = f'{source!r} is no bundled entry, {", ".join(BUNDLED)}, and no {SUFFIX_PDB} or .cif file.'
    raise ValidationError(error_msg)


def read_biopython_structure(
    structure: StructureSpec,
    models: tuple[int, ...] | None = None,
) -> BiopythonStructureProtocol:
    """Return a structure as an object of Biopython.

    Args:
        structure:
            The structure.

        models:
            The indices of the models to keep, or `None` for every model.

    Returns:
        object:
            The `Bio.PDB.Structure` of the atoms in their order.

    Raises:
        MissingExtraError:
            The extra `structures` is not installed.

    Examples:
        >>> from skcomplex.models.biology.systems_biology import read_biopython_structure, read_structure
        >>> len(list(read_biopython_structure(read_structure('1L2Y'), (0, 1)).get_chains()))
        2
    """
    pdb = import_extra('Bio.PDB', STRUCTURE_MODELS, 'structures')
    builder = pdb.StructureBuilder.StructureBuilder()
    builder.init_structure('structure')
    chosen = range(len(structure.coordinates)) if models is None else models
    for serial, index in enumerate(chosen):
        builder.init_model(serial, serial + 1)
        chain, number = None, None
        builder.init_seg(' ')
        for atom, (name, residue, position, owner) in enumerate(
            zip(
                structure.atom_names,
                structure.residue_names,
                structure.residue_numbers,
                structure.chain_names,
                strict=True,
            ),
        ):
            if owner != chain:
                chain, number = owner, None
                builder.init_chain(owner)
            if position != number:
                number = position
                builder.init_residue(residue, POLYMER, position, POLYMER)
            builder.init_atom(
                name,
                structure.coordinates[index, atom].astype(np.float32),
                0.0,
                1.0,
                POLYMER,
                name if len(name) == FULL_NAME else f' {name:<3}',
                atom + 1,
                name[0],
            )
    return builder.get_structure()


def write_structure(structure: StructureSpec, path: str | Path) -> None:
    """Write every model of a structure to a `.pdb`, `.cif`, or `.mmcif` file.

    Args:
        structure:
            The structure.

        path:
            The path of the file, whose suffix gives the format.

    Raises:
        MissingExtraError:
            The extra `structures` is not installed.

        ValidationError:
            The suffix of the path names no known format.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from skcomplex.models.biology.systems_biology import read_structure, write_structure
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     write_structure(read_structure('1UBQ'), Path(folder) / 'copy.cif')
        ...     read_structure(Path(folder) / 'copy.cif').coordinates.shape
        (1, 602, 3)
    """
    target = Path(path)
    pdb = import_extra('Bio.PDB', STRUCTURE_MODELS, 'structures')
    if target.suffix != SUFFIX_PDB and target.suffix not in SUFFIXES_MMCIF:
        error_msg = f'The structure file {target} has no suffix of a known format, {SUFFIX_PDB} or .cif.'
        raise ValidationError(error_msg)
    writer = pdb.PDBIO() if target.suffix == SUFFIX_PDB else pdb.MMCIFIO()
    writer.set_structure(read_biopython_structure(structure))
    writer.save(str(target))


def read_trajectory(path: str | Path, topology: str | Path) -> StructureSpec:
    """Return the frames of a molecular dynamics trajectory as the models of a structure, through MDTraj.

    Args:
        path:
            The path of a DCD, XTC, TRR, NetCDF, or HDF5 trajectory.

        topology:
            The path of a file that gives the atoms, such as a PDB file.

    Returns:
        structure:
            The structure, with one model per frame, in ångström.

    Raises:
        MissingExtraError:
            The extra `trajectories` is not installed.
    """
    trajectory = import_extra('mdtraj', STRUCTURE_MODELS, 'trajectories').load(str(path), top=str(topology))
    atoms = list(trajectory.topology.atoms)
    return StructureSpec(
        np.asarray(trajectory.xyz, dtype=float) * NM_TO_ANGSTROM,
        tuple(atom.name for atom in atoms),
        tuple(atom.residue.name for atom in atoms),
        tuple(int(atom.residue.resSeq) for atom in atoms),
        tuple(str(atom.residue.chain.chain_id or atom.residue.chain.index) for atom in atoms),
    ).check()
