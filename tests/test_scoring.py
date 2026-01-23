import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from etl_engine import minmax_scale_series


def test_apetite_investidor_range():
    # simulate sample data
    pontos = pd.Series([10, 50, 100, 25, 75], dtype=float)
    renda = pd.Series([2000, 5000, 15000, 8000, 3000], dtype=float)

    score_mob = minmax_scale_series(pontos).fillna(0.0)
    score_renda = minmax_scale_series(renda).fillna(0.0)

    apetite = (score_mob * 0.4) + (score_renda * 0.6)

    assert np.all(apetite >= 0.0) and np.all(apetite <= 1.0)
