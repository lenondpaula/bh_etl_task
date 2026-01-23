import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pytest
from etl_engine import normalize_text


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Bairro São José", "BAIRRO SAO JOSE"),
        ("  centro ", "CENTRO"),
        (None, None),
        ("moço bairro", "MOCO BAIRRO"),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected
