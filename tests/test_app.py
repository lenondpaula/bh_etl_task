import os
from pathlib import Path

import pytest


@pytest.mark.skip(reason="UI smoke test example; enable when CI supports streamlit.testing.v1")
def test_streamlit_app_smoke():
    # Ensure data exists before running UI test
    from src.generate_mock_data import main as generate_mock
    from src.etl_engine import run_etl

    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(exist_ok=True)

    generate_mock(count=5, seed=7)
    run_etl(
        csv_path=str(data_dir / "dados_economicos.csv"),
        geojson_path=str(data_dir / "bairros_data.geojson"),
        out_path=str(data_dir / "bh_final_data.geojson"),
    )

    # Basic UI test using streamlit.testing.v1
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run()

    # Verify title rendered and at least one chart present
    assert any("BH Strategic Navigator" in str(el) for el in at.title), "Title not rendered"
    assert len(at.get_delta("plotly_chart")) + len(at.get_delta("pydeck_chart")) >= 1, "No charts rendered"
