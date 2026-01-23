#!/usr/bin/env python3
"""ETL demonstrativo para o projeto BH Strategic Navigator.

Fluxo:
 - Ler `data/dados_economicos.csv` com Polars
 - Ler `data/bairros_data.geojson` com GeoPandas
 - Normalizar nomes (remover acentos, caixa alta)
 - Converter Polars -> Pandas e realizar merge
 - Criar scores normalizados e indicador final
 - Classificar oportunidades e salvar `data/bh_final_data.geojson`
"""
from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import MinMaxScaler


def normalize_text(s: str | None) -> str | None:
    """Normaliza texto removendo acentos e colocando em caixa alta.

    Args:
        s: Texto de entrada ou None.

    Returns:
        str | None: Texto normalizado (sem acentos, uppercase, sem espaços extremos) ou None se entrada for None.
    """
    if s is None:
        return None
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.upper().strip()


def minmax_scale_series(ser: pd.Series) -> pd.Series:
    """Aplica normalização Min-Max em uma série numérica com tolerância a NaN.

    Args:
        ser: Série pandas com valores numéricos e possivelmente NaN.

    Returns:
        pd.Series: Série normalizada no intervalo [0, 1]. Quando todos os valores são NaN ou constantes, retorna zeros.
    """
    arr = ser.to_numpy(dtype=float)
    if np.all(np.isnan(arr)):
        return pd.Series(np.zeros_like(arr), index=ser.index)
    arr_no_nan = arr[~np.isnan(arr)]
    if arr_no_nan.size == 0 or np.nanmax(arr) == np.nanmin(arr):
        return pd.Series(np.zeros_like(arr), index=ser.index)
    scaler = MinMaxScaler()
    arr2 = np.nan_to_num(arr, nan=0.0)
    scaled = scaler.fit_transform(arr2.reshape(-1, 1)).ravel()
    return pd.Series(scaled, index=ser.index)


def run_etl(csv_path: str = "data/dados_economicos.csv", geojson_path: str = "data/bairros_data.geojson", out_path: str = "data/bh_final_data.geojson") -> None:
    """Executa o pipeline ETL: leitura, normalização, merge, scoring e saída.

    Args:
        csv_path: Caminho para o CSV de dados econômicos.
        geojson_path: Caminho para o GeoJSON de bairros (geometria).
        out_path: Caminho de saída para o GeoJSON final com atributos calculados.

    Returns:
        None: Salva GeoJSON e Parquet em disco na pasta `data/`.
    """
    log = logging.getLogger("etl_engine")
    log.info("Lendo CSV com Polars: %s", csv_path)
    pl_df = pl.read_csv(csv_path)

    log.info("Lendo GeoJSON com GeoPandas: %s", geojson_path)
    gdf = gpd.read_file(geojson_path)

    # Normalizar nomes em GeoDataFrame
    log.info("Normalizando nomes em GeoDataFrame")
    gdf["Nome_Bairro_NORM"] = gdf["Nome_Bairro"].apply(normalize_text)
    # Remover colunas duplicadas antes do merge para evitar sufixos (_x/_y)
    dup_cols = ["Renda_Media", "Qtd_Empresas", "Qtd_Pontos_Onibus"]
    gdf = gdf.drop(columns=[c for c in dup_cols if c in gdf.columns])

    # Converter Polars -> Pandas e normalizar nomes
    log.info("Convertendo Polars -> Pandas e normalizando nomes")
    df_econ = pl_df.to_pandas()
    df_econ["Nome_Bairro_NORM"] = df_econ["Nome_Bairro"].apply(normalize_text)

    # Selecionar colunas de interesse no CSV antes do merge
    cols = [c for c in ["id_bairro", "Nome_Bairro_NORM", "Renda_Media", "Qtd_Empresas", "Qtd_Pontos_Onibus"] if c in df_econ.columns]
    df_econ_sel = df_econ[cols]

    log.info("Realizando merge entre geometria e dados econômicos")
    merged = gdf.merge(df_econ_sel, on="Nome_Bairro_NORM", how="left")

    # Garantir tipos numéricos
    for col in ["Renda_Media", "Qtd_Empresas", "Qtd_Pontos_Onibus"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
        else:
            merged[col] = np.nan

    # Engenharia de atributos: normalizações
    log.info("Calculando scores normalizados")
    merged["Score_Mobilidade"] = minmax_scale_series(merged["Qtd_Pontos_Onibus"]).fillna(0.0)
    merged["Score_Renda"] = minmax_scale_series(merged["Renda_Media"]).fillna(0.0)
    merged["Saturacao_Comercial"] = minmax_scale_series(merged["Qtd_Empresas"]).fillna(0.0)

    # Apetite_Investidor
    merged["Apetite_Investidor"] = (merged["Score_Mobilidade"] * 0.4) + (merged["Score_Renda"] * 0.6)

    # Classificação (vetorizado)
    merged["Classificacao"] = "REGULAR"
    mask_ouro = (merged["Apetite_Investidor"] > 0.7) & (merged["Saturacao_Comercial"] < 0.4)
    merged.loc[mask_ouro, "Classificacao"] = "OPORTUNIDADE DE OURO"
    mask_saturado = (merged["Saturacao_Comercial"] >= 0.7)
    merged.loc[mask_saturado, "Classificacao"] = "SATURADO"

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Salvando resultado final em %s", out_path)
    merged.to_file(out_path, driver="GeoJSON")

    # Salvar também versão parquet (sem geometria) para integração/consumo tabular
    try:
        parquet_path = Path(out_path).with_name("data_final.parquet")
        cols_out = [
            "id_bairro",
            "Nome_Bairro",
            "Renda_Media",
            "Qtd_Empresas",
            "Qtd_Pontos_Onibus",
            "Apetite_Investidor",
            "Classificacao",
        ]
        df_out = merged[[c for c in cols_out if c in merged.columns]].copy()
        pl_df = pl.DataFrame(df_out)
        pl_df.write_parquet(str(parquet_path))
        log.info("Parquet salvo em %s", parquet_path)
    except Exception as e:
        log.warning("Falha ao salvar parquet: %s", e)

    log.info("ETL concluído. Registros processados: %d", len(merged))
    # liberar memória de objetos grandes
    del merged


def main() -> None:
    """Entry point do módulo ETL: configura logging e executa o ETL padrão."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_etl()


if __name__ == "__main__":
    main()
