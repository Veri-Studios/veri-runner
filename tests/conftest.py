"""Make `veri_runner` importable from a plain checkout (no install needed).

Also works when this repo is vendored as a submodule inside the Veri
monorepo: the path inserted is this repo's root, wherever it is mounted.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
