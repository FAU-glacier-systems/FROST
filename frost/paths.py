# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""Data folders of the repository, independent of the working directory."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / 'data' / 'raw'
DATA_RESULTS = REPO_ROOT / 'data' / 'results'
