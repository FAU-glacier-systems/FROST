import os
import sys

# The `frost` package isn't pip-installed anywhere (no `pip install -e .` step
# in this project's workflow) - it's only importable because the repo root is
# on sys.path. `python -m pytest` gets that for free (a side effect of `-m`),
# but plain `pytest` does not, so make it explicit here.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
