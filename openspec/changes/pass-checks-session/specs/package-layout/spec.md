## ADDED Requirements

### Requirement: The code imports what it annotates
No module SHALL import inside an `if TYPE_CHECKING:` block, and the import contracts SHALL check type-checking imports
like any other. A spec SHALL live in the lowest package that can import every type its fields name.

#### Scenario: No hidden imports
- **WHEN** a test searches the source for `TYPE_CHECKING`
- **THEN** it finds none, and lint-imports passes with type-checking imports checked

### Requirement: Findings are fixed, not suppressed
The source, the tests, and the documentation SHALL hold no `noqa`, `type: ignore`, or `nosec` comment, and the checks
session SHALL pass: black, ruff, lint-imports, mypy, bandit, interrogate, pydoclint, and deptry.

#### Scenario: No suppression comments
- **WHEN** a test searches the source, the tests, and the examples for `noqa`, `type: ignore`, and `nosec`
- **THEN** it finds none
