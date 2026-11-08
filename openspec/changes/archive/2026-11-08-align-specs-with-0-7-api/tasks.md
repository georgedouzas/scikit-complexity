# Tasks

## 1. Delta specs

- [x] 1.1 Write the delta specs of every modified capability and verify `openspec validate align-specs-with-0-7-api` passes
- [x] 1.2 Check every modified requirement against the code in `src/skcomplex` and verify each named function, class, and argument exists

## Workflow follow-up

- Archive the change with `openspec archive align-specs-with-0-7-api -y` and verify `openspec validate --specs` passes.
- Grep `openspec/specs` for the removed names and verify none remains.
