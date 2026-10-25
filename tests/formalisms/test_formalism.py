"""Test the declaration and registration of formalisms, and the laws a system may hold."""

import json
import os
import subprocess  # nosec B404: the tests run the interpreter on code they write
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import ClassVar

import pytest

from skcomplex.core.base import (
    BaseLaw,
    LawClass,
    LawKind,
    NumericDomain,
    RegistryError,
    Requires,
    Trait,
    is_registered,
    register_laws,
    state,
)
from skcomplex.core.components import ElementComponent
from skcomplex.core.laws import Symbols
from skcomplex.core.system import System
from skcomplex.engine.compile import validate
from skcomplex.formalisms import Formalism, register

SOURCE = Path(str(files('skcomplex'))).parent
ROOT = Path(str(files('tests'))).parent


def _run(code: str) -> object:
    """Return the JSON a fresh interpreter prints after running code that imports the outside formalism."""
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(SOURCE), str(ROOT)])}
    result = subprocess.run(  # noqa: S603  # nosec B603: runs the test interpreter on code the test writes
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    return json.loads(result.stdout.splitlines()[-1])


@dataclass(frozen=True)
class StrayLaw(BaseLaw):
    """Law of a class that no formalism registers."""

    law_kind: ClassVar[LawKind] = LawKind.RATE
    time_class: ClassVar[LawClass] = LawClass.CONTINUOUS
    traits: ClassVar[frozenset[Trait]] = frozenset({Trait.DECLARABLE})
    requires: ClassVar[Requires] = Requires()

    target: object
    value: object


@dataclass(frozen=True)
class UndeclaredLaw(BaseLaw):
    """Law class without a kind."""


class DriftingComponent(ElementComponent):
    """Element whose law is of an unregistered class."""

    x: float = state(default=0.0, domain=NumericDomain.REAL)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return a stray law."""
        return [StrayLaw(s.x, 1)]


def test_registered_formalisms_list_contributions():
    """Test that a registered formalism is listed with its law kinds, its form, and its solvers."""
    found = _run(
        'import json\n'
        'from tests.formalisms import outside\n'
        'from skcomplex.formalisms import registered_formalisms\n'
        "(item,) = [item for item in registered_formalisms() if item.name == 'outside']\n"
        'print(json.dumps({\n'
        "    'kinds': [law.law_kind.name for law in item.laws],\n"
        "    'forms': [form.name for form in item.forms],\n"
        "    'solvers': sorted(item.solvers),\n"
        '}))\n',
    )
    assert found == {'kinds': ['DECAY'], 'forms': ['DECAYING'], 'solvers': ['exact-decay']}


def test_register_adds_contributions_to_their_registries():
    """Test that registering a formalism adds its laws and solvers, and that registering it again changes nothing."""
    found = _run(
        'import json\n'
        'from tests.formalisms import outside\n'
        'from skcomplex.formalisms import register, registered_formalisms\n'
        'from skcomplex.core.base import law_classes\n'
        'from skcomplex.engine.compile import SOLVERS\n'
        'register(outside.FORMALISM)\n'
        'print(json.dumps({\n'
        "    'law': outside.DecayLaw in law_classes(),\n"
        "    'solver': SOLVERS.resolve((outside.DECAYING, 'auto')) is outside.ExactDecaySolver,\n"
        "    'count': [item.name for item in registered_formalisms()].count('outside'),\n"
        '}))\n',
    )
    assert found == {'law': True, 'solver': True, 'count': 1}


def test_register_rejects_another_formalism_of_a_registered_name():
    """Test that a second formalism under a registered name is rejected."""
    found = _run(
        'import json\n'
        'from tests.formalisms import outside\n'
        'from skcomplex.core.base import RegistryError\n'
        'from skcomplex.formalisms import Formalism, register\n'
        'try:\n'
        "    register(Formalism('outside'))\n"
        'except RegistryError as error:\n'
        '    print(json.dumps(str(error)))\n',
    )
    assert 'outside' in str(found)
    assert 'Another formalism' in str(found)


def test_law_from_outside_formalism_validates_compiles_and_simulates():
    """Test that a law of an outside formalism validates, compiles with its pass, and simulates with its solver."""
    found = _run(
        'import json\n'
        'from tests.formalisms.outside import DECAYING, ExactDecaySolver, NucleusComponent\n'
        'from skcomplex.engine.compile import compile_system, validate\n'
        'from skcomplex.engine.runtime import Simulation\n'
        'from skcomplex.core.system import System\n'
        "system = System(components=[NucleusComponent('a', n=2.0, k=0.5)])\n"
        'simulation = Simulation(system, dt=0.5).run(until=2)\n'
        'print(json.dumps({\n'
        "    'valid': validate(system).ok,\n"
        "    'form': compile_system(system).form is DECAYING,\n"
        "    'solver': type(simulation.stepper_) is ExactDecaySolver,\n"
        "    'final': float(simulation.trajectory_.values('a.n')[-1]),\n"
        '}))\n',
    )
    assert found == {'valid': True, 'form': True, 'solver': True, 'final': pytest.approx(2 * 0.36787944117144233)}


@pytest.mark.slow
def test_register_of_fields_registers_its_dependency():
    """Test that importing the fields formalism registers both fields and the lattices it builds on."""
    found = _run(
        'import json\n'
        'import skcomplex.formalisms.fields\n'
        'from skcomplex.formalisms import registered_formalisms\n'
        'print(json.dumps([item.name for item in registered_formalisms()]))\n',
    )
    assert found == ['lattices', 'fields']


def test_register_rejects_unregistered_dependency():
    """Test that a formalism whose dependency is not registered is rejected and names the dependency."""
    with pytest.raises(RegistryError, match="'absent'"):
        register(Formalism('dependent', depends=('absent',)))


def test_register_rejects_cycle():
    """Test that a formalism that depends on itself is rejected as a cycle."""
    with pytest.raises(RegistryError, match='cycle'):
        register(Formalism('looped', depends=('looped',)))


def test_register_laws_rejects_classes_without_declarations():
    """Test that a class that is not a law class, or a law class without a kind, is not registered."""
    with pytest.raises(RegistryError, match='not a law class'):
        register_laws(int)
    with pytest.raises(RegistryError, match='law_kind'):
        register_laws(UndeclaredLaw)
    assert not is_registered(UndeclaredLaw)


def test_validate_reports_unregistered_law():
    """Test that validation names a law class that no formalism registered and says a formalism must register it."""
    report = validate(System(components=[DriftingComponent('d')]))
    (issue,) = [issue for issue in report.issues if issue.code == 'law-unregistered']
    assert 'StrayLaw' in issue.message
    assert 'formalism must register' in issue.message
    assert issue.path == 'd'
