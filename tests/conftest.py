import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Applies root to all test files for /metrics that don't call run_python, unless specifically overridden
sys.path.insert(0, str(REPO_ROOT / "metrics"))


@pytest.fixture
def run_python():
    def _run_python(script_path, *args):
        path = REPO_ROOT / script_path
        return subprocess.run([sys.executable, str(path), *map(str, args)], capture_output=True, text=True)
    return _run_python