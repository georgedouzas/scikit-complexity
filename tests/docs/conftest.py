"""Configure the tests of the documentation tooling, which lives in the documentation folder."""

import sys
from pathlib import Path

DOCS = Path(__file__).parent.parent.parent / 'docs'
sys.path.insert(0, str(DOCS))
