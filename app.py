import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pydeck as pdk
import altair as alt
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
    """Converte um score [0,1] em uma cor RGB com thresholds claros.

    Args:
        score: Valor normalizado entre 0 e 1.

    Returns:
        tuple: Tripla (R, G, B) indicando a cor.
    """
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return (200, 100, 0)
    if score > 0.7:
        return (0, 255, 0)
    if score >= 0.4:
        return (255, 200, 0)
    return (200, 0, 0)


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
        .stApp, [data-testid='stSidebar'] { background-color: #FFFFFF !important; color: #31333F !important; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }
        .css-18e3th9 { padding-top: 1rem; }

        /* Conteúdos principais (cards) */
        .stApp, .stSidebar, .stContainer, .stCard { background-color: #FFFFFF !important; color: #31333F !important; }
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
    with st.expander("🧭 Guia Rápido de Navegação e Uso", expanded=True):
        st.markdown("**Visualização 3D:** Altura = Densidade de Empresas; Cor = Apetite do Investidor (Verde = Alta Oportunidade / Blue Ocean).")
        st.markdown("**Sidebar:** Filtros por setor e indicadores em tempo real.")
        st.markdown("**Cluster Analysis:** Bairros são agrupados por perfis socioeconômicos similares; clusters de alto apetite sinalizam zonas premium para investimento.")
        st.markdown("**Controles:** Botão direito do mouse + arrastar para girar/inclinar a visão 3D.")
    st.markdown("#### Seleção de locais — análise de oportunidade comercial (MVP)")

    # Sidebar: About/README expander
    try:
        readme_text = Path("README_APP.md").read_text(encoding="utf8")
    except Exception:
        readme_text = "README_APP.md não encontrado."
    st.sidebar.expander("📄 Sobre o Projeto (README)", expanded=False).markdown(readme_text, unsafe_allow_html=True)


    # Footer corporativo customizado (exibe contato e licença)
    footer_md = """
    <div style='position: fixed; bottom: 8px; left: 16px; width: calc(100% - 32px); text-align: center; font-size:12px; color:#34415a;'>
      © 2026 Lenon de Paula &nbsp;&nbsp; | &nbsp;&nbsp; 📧 lenondpaula@gmail.com &nbsp;&nbsp; | &nbsp;&nbsp; 📱 +55 (55) 98135-9099
      <div style='margin-top:4px; font-size:11px; color:#707b8c;'>Licensed under the PolyForm Noncommercial License 1.0.0 — see /LICENSE</div>
    </div>
    """
    st.markdown(footer_md, unsafe_allow_html=True)

    data_path = Path("data/bh_final_data.parquet")
    if not data_path.exists():
        st.error("Arquivo data/bh_final_data.parquet não encontrado. Execute o ETL primeiro.")
        return

    @st.cache_data(show_spinner=False)
    def load_final_gdf(path_str: str) -> pd.DataFrame:
        """Carrega e prepara o GeoDataFrame final com colunas de estilo.

        Args:
            path_str: Caminho para o GeoJSON final.

        Returns:
            pd.DataFrame: GeoDataFrame com colunas calculadas para visualização.
        """
        import shapely.wkt as wkt
        if str(path_str).endswith('.parquet'):
            df = pd.read_parquet(path_str)
            if 'geometry_wkt' in df.columns:
                df['geometry'] = df['geometry_wkt'].apply(lambda s: wkt.loads(s) if pd.notna(s) else None)
            gdf_local = gpd.GeoDataFrame(df, geometry='geometry')
        else:
            gdf_local = gpd.read_file(path_str)
        # Ensure numeric columns exist and are filled
        for col in ["Qtd_Empresas", "Apetite_Investidor", "Saturacao_Comercial", "Renda_Media"]:
            if col not in gdf_local.columns:
                gdf_local[col] = 0
        # Prepare style columns
        gdf_local["_coords"] = gdf_local.geometry.apply(polygon_to_coords)
        # Convert geometry to WKT string to avoid pyarrow serialization errors in Streamlit
        if "geometry" in gdf_local.columns:
            gdf_local["geometry"] = gdf_local["geometry"].apply(lambda g: g.wkt if g is not None else None)
        gdf_local["Apetite_Investidor"] = gdf_local["Apetite_Investidor"].fillna(0).astype(float)
        gdf_local["Saturacao_Comercial"] = gdf_local["Saturacao_Comercial"].fillna(0).astype(float)
        # elevation: use Densidade_Comercial normalized to [0, 1000]
        dens = gdf_local.get("Densidade_Comercial", pd.Series(0)).fillna(0).astype(float)
        max_dens = max(1.0, float(dens.max()))
        # elevation: log-scale and cap to avoid extreme towers; keeps other bairros visible
        raw = ((dens / max_dens) * 1000).astype(float)
        gdf_local["elevation"] = (np.log1p(raw) * 150).clip(0, 400).astype(float)
        # fill_color: gradient yellow -> dark green based on Apetite_Investidor
        def _fill(s: float):
            sc = float(s) if (s is not None and not np.isnan(s)) else 0.0
            # thresholds: >0.7 green, 0.4-0.7 yellow/orange, <0.4 red
            if sc > 0.7:
                return [0, 255, 0, 160]
            if sc >= 0.4:
                return [255, 200, 0, 160]
            return [200, 0, 0, 160]
        gdf_local["fill_color"] = gdf_local["Apetite_Investidor"].apply(_fill)
        return gdf_local

    gdf = load_final_gdf(str(data_path))

    # Sidebar: legenda e filtro
    st.sidebar.header("Legenda")
    st.sidebar.markdown(
        "**Apetite = 0.4 * Mobilidade + 0.6 * Renda**\n\n- Mobilidade: pontos de ônibus por bairro (normalizado)\n- Renda: renda média por bairro (normalizada)",
        unsafe_allow_html=True,
    )

    name_col = choose_name_column(gdf)
    if name_col is None:
        st.error("Coluna de nome do bairro não encontrada no GeoDataFrame.")
        return

    bairros_options = sorted(list(gdf[name_col].dropna().unique()))

    # Sidebar: filter to Centro-Sul only. If region column exists, prefer that; otherwise use hardcoded list
    def _normalize_simple(s: str) -> str:
        import unicodedata

        if not s:
            return ""
        s2 = str(s).upper()
        s2 = "".join(c for c in unicodedata.normalize("NFKD", s2) if not unicodedata.combining(c))
        s2 = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s2)
        s2 = " ".join(s2.split())
        return s2

    region_col = next((c for c in gdf.columns if "REG" in c.upper()), None)
    centro_options = []
    if region_col:
        try:
            centro_options = sorted(list(gdf[gdf[region_col].str.contains("CENTRO", case=False, na=False)][name_col].dropna().unique()))
        except Exception:
            centro_options = []
    if not centro_options:
        hardcoded = ["Savassi", "Centro", "Lourdes", "Belvedere", "Sion", "Anchieta", "Cruzeiro", "Santo Agostinho", "Funcionários"]
        hard_norm = set(_normalize_simple(x) for x in hardcoded)
        centro_options = [b for b in bairros_options if (_normalize_simple(b) in hard_norm) or any(h in _normalize_simple(b) for h in hard_norm)]
        centro_options = sorted(list(dict.fromkeys(centro_options)))
    if not centro_options:
        centro_options = bairros_options[:9]

    selected = st.sidebar.multiselect("Selecionar bairros (Centro-Sul)", options=centro_options, default=centro_options[:5])
    if selected:
        gdf = gdf[gdf[name_col].isin(selected)].copy()

    # métricas rápidas na sidebar
    total_empresas = int(gdf["Qtd_Empresas"].sum())
    bairro_maior_renda = (
        gdf.loc[gdf["Renda_Media"].idxmax(), name_col] if gdf["Renda_Media"].notna().any() else "-"
    )
    bairro_maior_mobilidade = (
        gdf.loc[gdf["Score_Mobilidade"].idxmax(), name_col] if gdf["Score_Mobilidade"].notna().any() else "-"
    )
    st.sidebar.metric("Total de Empresas", f"{total_empresas:,}")
    st.sidebar.metric("Maior Renda (bairro)", f"{bairro_maior_renda}")
    st.sidebar.metric("Maior Mobilidade (bairro)", f"{bairro_maior_mobilidade}")

    # gdf já contém colunas de estilo via cache

    # Layout: map left, decision panel right
    tab1, tab2 = st.tabs(["Map", "Cluster Analysis"])

    with tab1:
        col_map, col_panel = st.columns([3, 1])

        with col_map:
            # pydeck expects records with polygon coordinates and color/elevation
            records = []
            for _, row in gdf.iterrows():
                coords = row.get("_coords", []) or []
                try:
                    coords = [[float(lon), float(lat)] for lon, lat in coords if lon is not None and lat is not None]
                except Exception:
                    coords = []
                fill_color = row.get("fill_color", [255, 255, 0, 180])
                fill_color = [int(x) for x in list(fill_color)]
                elevation = float(row.get("elevation", 0.0) or 0.0)
                rec = {
                    "coordinates": coords,
                    "fill_color": fill_color,
                    "elevation": elevation,
                    name_col: row.get(name_col),
                    "Renda_Media": float(row.get("Renda_Media", 0) or 0.0),
                    "Qtd_Empresas": int(row.get("Qtd_Empresas", 0) or 0),
                    "Classificacao": str(row.get("Classificacao", "")),
                }
                records.append(rec)

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
                elevation_scale=1,
                elevation_range=[0, 400],
            )

            tooltip = {
                "html": "<b>{%s}</b><br/>Renda: {%s}<br/>Empresas: {%s}<br/>Classificação: {%s}" % (name_col, "Renda_Media", "Qtd_Empresas", "Classificacao"),
                "style": {"backgroundColor": "#F0F0F0", "color": "#000000"},
            }

            df_table = gdf.drop(columns=["geometry"], errors="ignore").copy()

            deck = pdk.Deck(
                layers=[polygon_layer],
                initial_view_state=pdk.ViewState(latitude=-19.933, longitude=-43.935, zoom=13.5, pitch=45),
                tooltip=tooltip,
            )
            try:
                st.pydeck_chart(deck)
            except Exception as e:
                st.warning(f"Falha ao renderizar mapa 3D: {e}. Mostrando tabela resumo.")
                st.dataframe(gdf[[name_col, "Renda_Media", "Qtd_Empresas", "Classificacao"]])

        with col_panel:
            st.header("Indicadores")
            st.metric("Total de Empresas", f"{total_empresas:,}")
            st.markdown(f"**Bairro com maior renda:** {bairro_maior_renda}")
            st.markdown(f"**Bairro com maior mobilidade:** {bairro_maior_mobilidade}")
            st.markdown("---")
            st.markdown("Selecione bairros no painel lateral para comparar.")

    with tab2:
        st.header("Cluster Analysis")

        # Scatter plot (Altair) — saturação vs apetite
        try:
            scatter = alt.Chart(gdf.reset_index()).mark_circle(size=80).encode(
                x=alt.X("Saturacao_Comercial:Q", title="Saturação Comercial"),
                y=alt.Y("Apetite_Investidor:Q", title="Apetite Investidor"),
                color=alt.Color("Classificacao:N"),
                tooltip=[alt.Tooltip(name_col + ":N"), alt.Tooltip("Saturacao_Comercial:Q"), alt.Tooltip("Apetite_Investidor:Q"), alt.Tooltip("Qtd_Empresas:Q")],
            ).interactive().properties(height=420)
            st.altair_chart(scatter, use_container_width=True)
        except Exception as e:
            st.warning(f"Falha ao renderizar scatter (Altair): {e}")
            st.dataframe(gdf[[name_col, "Saturacao_Comercial", "Apetite_Investidor", "Qtd_Empresas"]])

        # Load IQVU / Score_Renda.csv and show bar chart
        @st.cache_data
        def load_iqvu(path: str) -> pd.DataFrame:
            try:
                df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8", on_bad_lines="skip")
                name_col_score = next((c for c in df.columns if "NOME" in c.upper()), None)
                iqvu_col = next((c for c in df.columns if "IQVU" in c.upper()), None)
                if name_col_score and iqvu_col:
                    df = df[[name_col_score, iqvu_col]].copy()
                    df[iqvu_col] = df[iqvu_col].astype(str).str.replace(",", ".", regex=False)
                    df[iqvu_col] = pd.to_numeric(df[iqvu_col], errors="coerce")
                    df[iqvu_col] = df[iqvu_col].fillna(0)
                    df = (
                        df.groupby(name_col_score)[iqvu_col]
                        .mean()
                        .reset_index()
                        .sort_values(iqvu_col, ascending=False)
                    )

                    df.columns = ["NOME", "IQVU"]
                    return df
            except Exception:
                return pd.DataFrame()
            return pd.DataFrame()


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
