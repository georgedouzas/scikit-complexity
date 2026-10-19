"""Scaffold, check, and run the gallery examples from the command line."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import argparse
import os
import sys
from pathlib import Path

from gallery_tools import (
    BACKENDS,
    DEFAULT_BACKEND,
    Budget,
    GalleryBuild,
    check_coverage,
    find_examples,
    run_examples,
    scaffold_examples,
    summarize,
)

DOCS = Path(__file__).parent.parent
EXAMPLES = DOCS / 'examples'
GALLERY = DOCS / 'generated' / 'gallery'
INVENTORY = DOCS / 'overview' / 'inventory.toml'


def _scaffold(arguments: argparse.Namespace) -> int:
    """Write the scaffolds of the models without an example."""
    written = scaffold_examples(EXAMPLES, INVENTORY, overwrite=arguments.overwrite)
    print(f'Wrote {len(written)} examples.')  # noqa: T201
    return 0


def _coverage(arguments: argparse.Namespace) -> int:
    """Report the models without an example and the orphan examples."""
    coverage = check_coverage(EXAMPLES)
    for name in coverage.missing:
        print(f'Missing example: {name}')  # noqa: T201
    for name in coverage.orphans:
        print(f'Orphan example: {name}')  # noqa: T201
    examples = find_examples(EXAMPLES)
    scaffolded = sum(example.scaffolded for example in examples)
    print(f'{len(examples)} examples, {len(examples) - scaffolded} curated, {scaffolded} scaffolded.')  # noqa: T201
    return 1 if arguments.strict and not coverage.complete else 0


def _run(arguments: argparse.Namespace) -> int:
    """Run the selected examples in parallel and capture their screenshots into the cache."""
    build = GalleryBuild(EXAMPLES, GALLERY, run_all=arguments.all, backend=arguments.backend)
    examples = build.selected()
    if arguments.names:
        examples = [example for example in examples if example.model in arguments.names]
    reports = run_examples(examples, build, Budget(), arguments.jobs)
    print(summarize(reports))  # noqa: T201
    return 1 if any(report.error for report in reports) else 0


def main() -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(prog='gallery_tools', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    scaffold = commands.add_parser('scaffold', help='write the scaffolds of the models without an example')
    scaffold.add_argument('--overwrite', action='store_true', help='rewrite the examples that are still scaffolds')
    scaffold.set_defaults(handler=_scaffold)
    coverage = commands.add_parser('coverage', help='report the models without an example and the orphans')
    coverage.add_argument('--strict', action='store_true', help='fail when the coverage is not complete')
    coverage.set_defaults(handler=_coverage)
    run = commands.add_parser('run', help='run examples and capture their screenshots into the cache')
    run.add_argument('names', nargs='*', help='the models to run, all selected ones by default')
    run.add_argument('--backend', choices=BACKENDS, default=DEFAULT_BACKEND, help='the capture backend')
    run.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2), help='the number of processes')
    run.add_argument('--all', action='store_true', help='run the scaffolded examples too')
    run.set_defaults(handler=_run)
    arguments = parser.parse_args()
    return int(arguments.handler(arguments))


if __name__ == '__main__':
    sys.exit(main())
