# Scripts

`check_conventions.py` checks the conventions of the sources, the tests, and the documentation, such as the placement
rule, the docstring style, and the test files that mirror the modules. Run it from the root of the repository with
`pdm run python openspec/scripts/check_conventions.py`, or through `pdm checks conventions`. It prints each finding as
`file:line: message` and exits with 1 when it finds one.
