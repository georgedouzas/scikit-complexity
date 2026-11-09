## Why

The user asked for two rules across the code: "hoist constants at the top using only PUBLIC names, no blank line
between them and comments" and "prefer to have methods or static methods in classes instead of private functions".
The constitution and `package-layout` already state both in part, but the code does not follow them. Many module
constants carry a leading underscore, sit after classes, or are split by blank lines and attribute docstrings. Many
private module functions serve one class or several classes of their module, so a reader of a class cannot see
everything it does, and a refactor of a base cannot see which helpers its children share. Nothing checks either rule,
so they drift. This change makes the code follow them and adds a test that keeps them true.

## What Changes

- Every module constant sits in one block right after the imports, before the type aliases and the first definition.
  The block holds no blank line and no comment, and no constant carries an attribute docstring.
- Every module constant has a public `UPPER_CASE` name. Private constants are renamed. A rename that clashes with a
  name the module already holds takes a more specific name. No package re-exports a renamed constant, so the public
  surface does not grow.
- A module value built from a class or function of its own module, such as a Sage symbolic function with a print hook
  or a dialect built from a spec class of its module, is a derived value. It sits in one block right after the last
  definition it reads, with the same naming and spacing rules.
- A function that a class reads, directly or through other private functions, becomes a method of that class: a
  private method, a `@staticmethod` when it reads no instance state, and a `@classmethod` only when it needs the
  class. Functions that several classes read have one owner and the others call it through the owner.
- Three kinds of module functions stay. Numba-compiled kernels stay module functions, because compiled code calls
  them by their global name. Private helpers of public functions stay in the modules whose public API is functions,
  such as analyses, experiments, law utilities, and `simulate`, when no class reads them. Hooks that a derived value
  hands to a library stay beside that value.
- A public function of a private module that no package re-exports is a helper too. It becomes a method of the class
  that reads it, or moves to the `_utils` module of its package when classes of several modules read it.
- The constitution states the exceptions above, so the rules and the code agree.
- A test in `tests/test_sources.py` checks every module of `src/skcomplex` against these rules and names each module,
  line, and name that breaks them.
- Behaviour does not change. Tests that import a moved private function call it through its owning class instead.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `package-layout`: module constants form one public block after the imports, classes own the functions they read,
  the three kinds of module functions that stay are named, and a source test checks both rules.

## Impact

- Code: every module of `src/skcomplex` that holds a private module function, a private or misplaced constant, or a
  gap in its constant block. The heaviest packages are `core/analysis`, `core/engine` (viz, solvers, compile,
  problems, backends), `core/experiments`, `core/specs`, `core/components/laws/utils`, and the system modules of every
  field.
- Tests: `tests/test_sources.py` gains the checks. Tests that import a moved private function or a renamed private
  constant change the import only.
- Documents: `openspec/constitution.md` names the exceptions. `CONTRIBUTING.md` points to the rules where it describes
  module layout.
- Performance: Numba kernels are unchanged. A static method call costs the same as a module function call.
- Dependencies: it lands after `add-system-package`, `slim-core-elements-and-spaces`, `slim-core-interactions`, and
  `slim-core-laws`, because those changes move, merge, and remove many of the same modules. It measures its scope again
  on the code they leave.
- Users: none. Only private names change, and no public name is added or removed.
