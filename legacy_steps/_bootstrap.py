from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root_on_path():
    project_root = Path(__file__).resolve().parents[1]
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

