#!/usr/bin/env python3
"""Gera dados sintéticos geoespaciais para o MVP 'BH Strategic Navigator'.

Cria 50 polígonos aproximados (bairros) ao redor de coordenadas de Belo Horizonte
e salva `data/bairros_data.geojson` e `data/dados_economicos.csv`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
from faker import Faker
from shapely.geometry import Polygon


def generate_square(center_lon: float, center_lat: float, size_deg: float) -> Polygon:
    """Gera um polígono quadrado centrado em uma coordenada.

    Args:
        center_lon: Longitude do centro do polígono (EPSG:4326).
        center_lat: Latitude do centro do polígono (EPSG:4326).
        size_deg: Tamanho do lado do quadrado em graus.

    Returns:
        Polygon: Polígono quadrado com vértices definidos em coordenadas geográficas.
    """
    half = size_deg / 2.0
    return Polygon([
        (center_lon - half, center_lat - half),
        (center_lon - half, center_lat + half),
        (center_lon + half, center_lat + half),
        (center_lon + half, center_lat - half),
        (center_lon - half, center_lat - half),
    ])


def main(count: int = 50, seed: int = 42) -> None:
    """Gera dados sintéticos e salva GeoJSON e CSV.

    Args:
        count: Número de bairros (polígonos) a gerar.
        seed: Semente de aleatoriedade para reprodutibilidade.

    Returns:
        None: Salva arquivos em disco na pasta `data/`.

    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("generate_mock_data")

    faker = Faker("pt_BR")
    rng = np.random.RandomState(seed)

    base_lat = -19.9
    base_lon = -43.9

    geoms: List[Polygon] = []
    ids: List[int] = []
    nomes: List[str] = []
    renda_media: List[int] = []
    qtd_empresas: List[int] = []
    qtd_pontos_onibus: List[int] = []

    for i in range(count):
        # pequenos deslocamentos em graus (~ up to ~5 km)
        lat_offset = rng.uniform(-0.05, 0.05)
        lon_offset = rng.uniform(-0.05, 0.05)
        center_lat = base_lat + lat_offset
        center_lon = base_lon + lon_offset

        # tamanho do polígono (em graus) — varia para simular bairros maiores/menores
        size = float(rng.uniform(0.002, 0.02))

        geoms.append(generate_square(center_lon, center_lat, size))
        ids.append(i + 1)
        # Nome do bairro sintético: combina cidade fake com índice para evitar duplicatas
        nomes.append(f"{faker.city()} {i+1}")
        renda_media.append(int(rng.uniform(2000, 15000)))
        qtd_empresas.append(int(rng.randint(10, 501)))
        qtd_pontos_onibus.append(int(rng.randint(5, 101)))

    gdf = gpd.GeoDataFrame(
        {
            "id_bairro": ids,
            "Nome_Bairro": nomes,
            "Renda_Media": renda_media,
            "Qtd_Empresas": qtd_empresas,
            "Qtd_Pontos_Onibus": qtd_pontos_onibus,
        },
        geometry=geoms,
        crs="EPSG:4326",
    )

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    geojson_path = out_dir / "bairros_data.geojson"
    csv_path = out_dir / "dados_economicos.csv"

    log.info("Salvando GeoJSON em %s", geojson_path)
    gdf.to_file(geojson_path, driver="GeoJSON")

    log.info("Salvando CSV econômico em %s", csv_path)
    gdf[["id_bairro", "Nome_Bairro", "Renda_Media", "Qtd_Empresas", "Qtd_Pontos_Onibus"]].to_csv(
        csv_path, index=False
    )

    log.info("Geração concluída: %d bairros gerados", count)


if __name__ == "__main__":
    main()
