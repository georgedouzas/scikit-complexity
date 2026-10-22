# Design

## Context

The inventory records 537 entries, each tagged after `redesign-system-model` with its time kind, space kind, component
kinds, interaction bindings, and law classes, and with its priority. Citation counts per source show which books
were mined: Bertin is cited 109 times, while eight books are cited four times or fewer.

## Goals / Non-Goals

**Goals:**
- Make the coverage of every source explicit, so a thin citation count means a thin source and not a skipped one.
- Find primitives the core lacks before more models of covered kinds are added.

**Non-Goals:**
- Implementing the surveyed models; their entries go pending or blocked on the changes that implement them.

## Decisions

### Survey record

The design records, per source, each chapter or section read, the models found, and the tags of each in the system
vocabulary: time kind, space kind, component kinds, interaction bindings, and law classes. A section with no model is recorded as such, with a reason such as "conceptual" or
"covered by an existing entry".

### Priorities

New entries take the inventory's rule: `core` when they exercise a category or a combination of categories that no
implemented model exercises, or do not map to the vocabulary, and `example` when they repeat a covered pattern. An entry
that does not map is blocked on the change that adds the general primitive it needs; if no change adds it, the survey
proposes one. A field-specific need, such as a named potential, a pairing algorithm, or a learning heuristic, is not a
missing primitive: the entry maps to the general primitive, and the realization is part of implementing the model in
`skcomplex.models.<field>`.

## Risks / Trade-offs

- [Conceptual books yield few models] → The survey records the sections it read, so the low count is explained.
