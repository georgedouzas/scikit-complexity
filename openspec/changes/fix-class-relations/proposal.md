# Proposal

## Why

Several classes outside the bases derive from a class they are not; they derive from it only to fit a slot:

- `InvariantQuantity`, `ObjectiveQuantity`, and `AuxiliaryQuantity` are laws only so they fit the `laws` list of a component.
- `AggregateQuantity` and `WienerProcess` are elements only so a role can bind them.
- `RewritingLaw` is an interaction, but it acts on a single component through one role, the way a law does.
- Five group solvers derive from `WordProblem`, and an experiment's schedule is an `Observer`.

Each of these reads as one thing and is another.

## What Changes

- **BREAKING** A *quantity* is a named expression of a system's states and parameters. `ObservableQuantity`, `AuxiliaryQuantity`,
  `InvariantQuantity`, `ObjectiveQuantity`, and `AggregateQuantity` are quantities that differ in what the framework does with them:

  | Quantity | What the framework does with it |
  |---|---|
  | observable | records it |
  | auxiliary | substitutes it into laws |
  | invariant | checks that it stays constant |
  | objective | optimizes it |
  | aggregate | gives it to other partitions |

  They derive from a new `BaseQuantity` in `core/base`. A component *has* quantities, declared next to its
  laws. None of them is a law or an element any more.
- **BREAKING** `WienerProcess` stops being an element. It stays bound to an interaction through a role, as the noise
  processes the interaction draws.
- **BREAKING** `RewritingLaw` and `RewriteLaw` become laws of the rewriting formalism that a word, patch, or host graph
  has.
- The five group solvers become implementations of the word problem, named after it, such as `FreeWordProblem`.
- The schedule of an experiment becomes a member of the experiment that the run consults at every frame, not an
  observer.

## Capabilities

### New Capabilities

### Modified Capabilities
- `system-modeling`: quantities are declared next to laws, and `AggregateQuantity` and `WienerProcess` are no longer elements.
- `rewriting-systems`: rewriting rules are laws of the rewritten component.

## Impact

- **Code:**
  - about 34 modules use `InvariantQuantity`, `ObjectiveQuantity`, or `AuxiliaryQuantity`;
  - about 26 modules use `AggregateQuantity` or `WienerProcess`;
  - the rewriting formalism;
  - the groups formalism;
  - `engine/experiments`.
- **Models:** every model that declares one of these quantities in its laws moves it to the quantity declaration.
- **Depends on:** `gather-base-classes`, which adds the bases this change derives from.
