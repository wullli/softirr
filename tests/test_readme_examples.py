
from __future__ import annotations

import re
from pathlib import Path

import pytest

_README_PATH = Path(__file__).parents[1] / "README.md"
_PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _extract_python_blocks() -> list[str]:
    text = _README_PATH.read_text(encoding="utf-8")
    return _PYTHON_BLOCK_RE.findall(text)


_blocks = _extract_python_blocks()


def test_readme_has_python_blocks():
    assert _blocks, "README.md should contain at least one ```python code block."


@pytest.mark.parametrize("block", _blocks, ids=range(len(_blocks)))
def test_readme_python_block_runs(block: str):
    exec(compile(block, "README.md", "exec"), {"__name__": "__main__"})
