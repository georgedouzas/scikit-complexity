## Why

The user asked for two rules across the code: "hoist constants at the top using only PUBLIC names, no blank line
between them and comments" and "prefer to have methods or static methods in classes instead of private functions".
The constitution and `package-layout` already state both in part, but the code does not follow them. Many module
constants carry a leading underscore, sit after classes, or are split by blank lines and attribute docstrings. Many
private module functions serve one class or several classes of their module, so a reader of a class cannot see
everything it does, and a refactor of a base cannot see which helpers its children share. The symbolic functions of
the library are defined in two ways, and many methods begin with a noun although the constitution asks for a verb.
Nothing checks these rules, so they drift. This change makes the code follow them and adds tests that keep them true.

## What Changes

- In every Python file of `src`, `tests`, and `docs`, the module constants sit in one block right after the imports,
  before the type aliases and the first definition. The block holds no blank line and no comment, and no constant
  carries an attribute docstring.
- Every module constant has a public `UPPER_CASE` name. Private constants are renamed. A rename that clashes with a
  name the module already holds takes a more specific name. No package re-exports a renamed constant, so the public
  surface does not grow.
- A module value built from a class or function of its own module, such as a symbolic function instance or a table
  built from a class of its module, is a derived value. It sits in one block right after the last definition it
  reads, with the same naming and spacing rules.
- Every symbolic function the library defines is a private `BuiltinFunction` subclass, as `where` and `stop_gradient`
  already are, so its print, evaluation, and derivative hooks are methods. Unknown functions of equations keep Sage's
  `function(name)`.
- A function that a class reads in `src`, directly or through other private functions, becomes a method of that class:
  a private method, a `@staticmethod` when it reads no instance state, and a `@classmethod` only when it needs the
  class. Functions that several classes read have one owner and the others call it through the owner.
- Every method of `src` begins with a verb, as the constitution asks. Moved helpers and older methods whose names
  begin with a noun take verb names, such as `_noise` becoming `_get_noise`.
- Three kinds of module functions stay. Numba-compiled kernels stay module functions, because compiled code calls
  them by their global name. Private helpers that only public functions read stay, in the modules whose public API is
  functions, such as analyses, experiments, law utilities, and `simulate`. Plain functions that a derived value reads,
  such as the entries of a dispatch table, stay beside that value.
- A public function of a private module that no package re-exports is a helper too. It becomes a method of the class
  that reads it, or moves to the `_utils` module of its package when classes of several modules read it.
- The constitution states the exceptions above, so the rules and the code agree.
- Tests in `tests/test_sources.py` check every file against these rules and name each file, line, and name that
  breaks them.
- Behaviour does not change. Tests that import a moved private function call it through its owning class instead.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `package-layout`: module constants form one public block after the imports in the sources, tests, and documents,
  classes own the functions they read, methods begin with a verb, symbolic functions are `BuiltinFunction` subclasses,
  the three kinds of module functions that stay are named, and source tests check all of it.

## Impact

- Code: every module of `src/skcomplex` that holds a private module function, a private or misplaced constant, a gap
  in its constant block, a symbolic function built with Sage's factory, or a method whose name begins with a noun. The
  heaviest packages are `core/analysis`, `core/engine` (viz, solvers, compile, problems, backends), `core/experiments`,
  `core/specs`, `core/components/laws/utils`, and the system modules of every field.
- Tests: `tests/test_sources.py` gains the checks. Test modules with misplaced or split constants are reordered. Tests
  that import a moved private function or a renamed private constant change the import only.
- Documents: the gallery examples, the gallery tools and hooks, and the generators follow the constants rule.
  `openspec/constitution.md` names the exceptions. `CONTRIBUTING.md` points to the rules where it describes module
  layout.
- Performance: Numba kernels are unchanged. A static method call costs the same as a module function call.
- Dependencies: it lands after `add-system-package`, `slim-core-elements-and-spaces`, `slim-core-interactions`,
  `slim-core-laws`, `rename-core-systems`, and `move-recipes-to-systems`, because those changes move, merge, and remove
  many of the same modules. It measures its scope again on the code they leave.
- Users: none. Only private names change, and no public name is added or removed. Expressions print as before.
