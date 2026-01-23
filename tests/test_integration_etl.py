import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import polars as pl
from generate_mock_data import main as generate_mock
from etl_engine import run_etl


def test_etl_generates_parquet_with_required_columns(tmp_path):
    # Ensure data directory exists in project root
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    # Generate synthetic source data
    generate_mock(count=10, seed=123)

    # Run ETL producing geojson and parquet
    run_etl(
        csv_path=str(data_dir / "dados_economicos.csv"),
        geojson_path=str(data_dir / "bairros_data.geojson"),
        out_path=str(data_dir / "bh_final_data.geojson"),
    )

    parquet_path = data_dir / "data_final.parquet"
    assert parquet_path.exists(), "Parquet file was not created by ETL"

    df = pl.read_parquet(str(parquet_path))
    required = {
        "Nome_Bairro",
        "Renda_Media",
        "Qtd_Empresas",
        "Qtd_Pontos_Onibus",
        "Apetite_Investidor",
        "Classificacao",
    }
    assert required.issubset(set(df.columns)), f"Missing required columns: {required - set(df.columns)}"
