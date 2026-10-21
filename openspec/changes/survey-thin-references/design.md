# Design

## Context

The inventory records 525 models with their primitives and priorities. Citation counts per source show which books
were mined: Bertin is cited 109 times, while eight books are cited four times or fewer.

## Goals / Non-Goals

**Goals:**
- Make the coverage of every source explicit, so a thin citation count means a thin source and not a skipped one.
- Find primitives the core lacks before more models of covered kinds are added.

**Non-Goals:**
- Implementing the surveyed models; their entries go pending or blocked on the changes that implement them.

## Decisions

### Survey record

The design records, per source, each chapter or section read, the models found, and the primitives each exercises,
using the inventory vocabulary. A section with no model is recorded as such, with a reason such as "conceptual" or
"covered by an existing entry".

### Priorities

New entries take the inventory's rule: `core` when they exercise a primitive or a combination of primitives that no
implemented model exercises, or need a missing primitive, and `example` when they repeat a covered pattern. Entries that
need a missing primitive are blocked on the change that adds it; if no change adds it, the survey proposes one.

## Risks / Trade-offs

- [Conceptual books yield few models] → The survey records the sections it read, so the low count is explained.
