import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pydeck as pdk
import plotly.express as px
import streamlit as st
import pandas as pd


def choose_name_column(gdf):
    """Seleciona a coluna de nome do bairro disponível.

    Args:
        gdf: DataFrame/GeoDataFrame com colunas de nome possíveis.

    Returns:
        str | None: Nome da coluna encontrada ou None se nenhuma disponível.
    """
    for c in ["Nome_Bairro", "Nome_Bairro_x", "Nome_Bairro_y"]:
        if c in gdf.columns:
            return c
    return None


def color_from_score(score: float) -> tuple:
    """Converte um score [0,1] em uma cor RGB.

    Args:
        score: Valor normalizado entre 0 e 1.

    Returns:
        tuple: Tripla (R, G, B) indicando a cor.
    """
    r = int(255 * (1 - score))
    g = int(200 * score)
    b = 50
    return (r, g, b)


def polygon_to_coords(poly):
    """Extrai coordenadas de um Polygon/MultiPolygon em formato [lon, lat].

    Args:
        poly: Geometria shapely (Polygon ou MultiPolygon).

    Returns:
        list[list[float]]: Lista de coordenadas [lon, lat] do contorno externo.
    """
    try:
        exterior = poly.exterior.coords[:]
        return [[lon, lat] for lon, lat in exterior]
    except Exception:
        try:
            for p in poly.geoms:
                exterior = p.exterior.coords[:]
                return [[lon, lat] for lon, lat in exterior]
        except Exception:
            return []


def main():
    """Inicializa a aplicação Streamlit com mapa 3D e análise de clusters.

    Aplica estilo corporativo e minimalista via CSS embutido e organiza o layout.
    """
    st.set_page_config(layout="wide", page_title="BH Strategic Navigator - Site Selection AI")

    # Tema corporativo minimalista (cores neutras + acento azul)
    st.markdown(
        """
        <style>
        /* Background e tipografia */
        .stApp { background-color: #f7f8fa; color: #0b1f3b; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }
        .css-18e3th9 { padding-top: 1rem; }

        /* Conteúdos principais (cards) */
        .stContainer, .stCard, .streamlit-expanderHeader { background-color: transparent }
        .stMetric > div { background-color: #ffffff !important; border-radius: 10px !important; box-shadow: 0 4px 10px rgba(11,31,59,0.06) !important; padding: 10px !important; }

        /* Títulos */
        .stMarkdown h1 { color: #0B5FFF; font-weight: 600; }
        .stMarkdown h2 { color: #0b2b59; }

        /* Esconder menu do Streamlit para visual mais limpo */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("# BH Strategic Navigator - Site Selection AI")
    st.markdown("#### Seleção de locais — análise de oportunidade comercial (MVP)")

    # Footer corporativo customizado (exibe contato e licença)
    footer_md = """
    <div style='position: fixed; bottom: 8px; left: 16px; width: calc(100% - 32px); text-align: center; font-size:12px; color:#34415a;'>
      © 2026 Lenon de Paula &nbsp;&nbsp; | &nbsp;&nbsp; 📧 lenondpaula@gmail.com &nbsp;&nbsp; | &nbsp;&nbsp; 📱 +55 (55) 98135-9099
      <div style='margin-top:4px; font-size:11px; color:#707b8c;'>Licensed under the PolyForm Noncommercial License 1.0.0 — see /LICENSE</div>
    </div>
    """
    st.markdown(footer_md, unsafe_allow_html=True)

    data_path = Path("data/bh_final_data.geojson")
    if not data_path.exists():
        st.error("Arquivo data/bh_final_data.geojson não encontrado. Execute o ETL primeiro.")
        return

    @st.cache_data(show_spinner=False)
    def load_final_gdf(path_str: str) -> pd.DataFrame:
        """Carrega e prepara o GeoDataFrame final com colunas de estilo.

        Args:
            path_str: Caminho para o GeoJSON final.

        Returns:
            pd.DataFrame: GeoDataFrame com colunas calculadas para visualização.
        """
        gdf_local = gpd.read_file(path_str)
        # Ensure numeric columns exist and are filled
        for col in ["Qtd_Empresas", "Apetite_Investidor", "Saturacao_Comercial", "Renda_Media"]:
            if col not in gdf_local.columns:
                gdf_local[col] = 0
        # Prepare style columns
        gdf_local["_coords"] = gdf_local.geometry.apply(polygon_to_coords)
        gdf_local["Apetite_Investidor"] = gdf_local["Apetite_Investidor"].fillna(0).astype(float)
        gdf_local["Saturacao_Comercial"] = gdf_local["Saturacao_Comercial"].fillna(0).astype(float)
        max_emp_local = max(1, int(gdf_local["Qtd_Empresas"].max()))
        gdf_local["elevation"] = gdf_local["Qtd_Empresas"].fillna(0).astype(float) / max_emp_local * 3000
        gdf_local[["r", "g", "b"]] = gdf_local.apply(
            lambda row: pd_color_tuple(row["Apetite_Investidor"]), axis=1, result_type="expand"
        )
        return gdf_local

    gdf = load_final_gdf(str(data_path))

    name_col = choose_name_column(gdf)
    if name_col is None:
        st.error("Coluna de nome do bairro não encontrada no GeoDataFrame.")
        return

    # gdf já contém colunas de estilo via cache

    # Layout: map left, decision panel right
    tab1, tab2 = st.tabs(["Map", "Cluster Analysis"])

    with tab1:
        col_map, col_panel = st.columns([3, 1])

        with col_map:
            # pydeck expects records with polygon coordinates and color/elevation
            records = gdf.assign(
                coordinates=gdf["_coords"],
                fill_color=gdf[["r", "g", "b"]].values.tolist(),
                elevation=gdf["elevation"].astype(float),
            ).to_dict(orient="records")

            polygon_layer = pdk.Layer(
                "PolygonLayer",
                data=records,
                pickable=True,
                stroked=True,
                filled=True,
                extruded=True,
                wireframe=True,
                get_polygon="coordinates",
                get_fill_color="fill_color",
                get_line_color=[80, 80, 80],
                get_elevation="elevation",
            )

            tooltip = {
                "html": "<b>{%s}</b><br/>Renda: {%s}<br/>Empresas: {%s}<br/>Classificação: {%s}" % (name_col, "Renda_Media", "Qtd_Empresas", "Classificacao"),
                "style": {"backgroundColor": "#F0F0F0", "color": "#000000"},
            }

            deck = pdk.Deck(layers=[polygon_layer], initial_view_state=pdk.ViewState(latitude=-19.9, longitude=-43.9, zoom=11, pitch=45), tooltip=tooltip)
            st.pydeck_chart(deck)

        with col_panel:
            # Metrics
            top_idx = int(gdf["Apetite_Investidor"].idxmax())
            top_bairro = gdf.loc[top_idx, name_col]
            mean_renda = float(gdf["Renda_Media"].dropna().mean())

            st.metric("Bairro com Maior Potencial", top_bairro)
            st.metric("Média de Renda da Região", f"R$ {mean_renda:,.0f}")

    with tab2:
        st.header("Cluster Analysis")

        fig = px.scatter(
            gdf,
            x="Saturacao_Comercial",
            y="Apetite_Investidor",
            hover_name=name_col,
            size_max=12,
            color="Classificacao",
            labels={"Saturacao_Comercial": "Saturação Comercial", "Apetite_Investidor": "Apetite Investidor"},
        )

        # Highlight ideal quadrant: Saturacao_Comercial < 0.4 and Apetite_Investidor > 0.7
        fig.add_shape(type="rect", x0=0, x1=0.4, y0=0.7, y1=1.0, fillcolor="rgba(0,255,0,0.08)", line_width=0)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)


def pd_color_tuple(score: float):
    """Helper para gerar tupla RGB a partir de score normalizado.

    Args:
        score: Valor normalizado entre 0 e 1.

    Returns:
        tuple: Tripla (R, G, B).
    """
    r = int(255 * (1 - score))
    g = int(200 * (score))
    b = 50
    return (r, g, b)


if __name__ == "__main__":
    main()
