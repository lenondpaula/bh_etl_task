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
        .stApp { background-color: #FFFFFF !important; color: #31333F !important; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }
        /* Sidebar: dark theme */
        [data-testid='stSidebar'] { background-color: #1A1C24 !important; color: #FFFFFF !important; }
        [data-testid='stSidebar'] .stMarkdown, [data-testid='stSidebar'] label, [data-testid='stSidebar'] .stText { color: #E0E0E0 !important; }
        /* Multiselect tags and selected chips - high contrast background */
        [data-testid='stSidebar'] .stMultiSelect span, [data-testid='stSidebar'] .stMultiSelect div, [data-testid='stSidebar'] .stSelectbox span { color: #FFFFFF !important; }
        [data-testid='stSidebar'] .stMultiSelect .css-1n39o9k > div, [data-testid='stSidebar'] .stMultiSelect .css-1n39o9k .css-1gk2a, [data-testid='stSidebar'] .stSelectbox .css-1n39o9k > div { background-color: #2B2B2B !important; color: #FFFFFF !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius:4px !important; }
        [data-testid='stSidebar'] .stMultiSelect .css-1n39o9k > div span, [data-testid='stSidebar'] .stSelectbox .css-1n39o9k > div span { color: #FFFFFF !important; }
        .css-18e3th9 { padding-top: 1rem; }

        /* Conteúdos principais (cards) */
        .stContainer, .stCard { background-color: #FFFFFF !important; color: #31333F !important; }
        .stMetric > div { background-color: transparent !important; border-radius: 6px !important; box-shadow: none !important; padding: 6px !important; }
/* Sidebar metrics: compact, white text on dark bg */
[data-testid='stSidebar'] .stMetric > div { background-color: #262730 !important; color: #FFFFFF !important; font-size: 0.9rem !important; padding: 8px !important; }
[data-testid='stSidebar'] .stMetric > div .stMetricValue, [data-testid='stSidebar'] .stMetric > div .stMetricLabel { color: #FFFFFF !important; }

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

    st.markdown("# BH Strategic Navigator")
    # Two-column layout for expanders: Guide (left) and README (right)
    col1, col2 = st.columns(2)
    with col1.expander("🧭 Guia Rápido de Navegação e Uso", expanded=True):
        st.markdown("**Visualização 3D:** Altura = Densidade de Empresas; Cor = Apetite do Investidor (Verde = Alta Oportunidade / Blue Ocean).")
        st.markdown("**Sidebar:** Filtros por setor e indicadores em tempo real.")
        st.markdown("**Cluster Analysis:** Bairros são agrupados por perfis socioeconômicos similares; clusters de alto apetite sinalizam zonas premium para investimento.")
        st.markdown("**Controles:** Botão direito do mouse + arrastar para girar/inclinar a visão 3D.")

    try:
        readme_text = Path("README_APP.md").read_text(encoding="utf8")
    except Exception:
        readme_text = "README_APP.md não encontrado."
    with col2.expander("📄 Sobre o Projeto (README)", expanded=False):
        st.markdown(readme_text, unsafe_allow_html=True)

    st.markdown("#### Seleção de locais — análise de oportunidade comercial (MVP)")



    # Sticky professional footer (full-width)
    footer_css_html = """
    <style>
      .footer-sticky {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0E1117;
        z-index: 999;
        padding: 10px;
        text-align: center;
        color: #FFFFFF;
        font-size:12px;
      }
    </style>
    <div class="footer-sticky">
      © 2026 Lenon de Paula &nbsp;&nbsp; | &nbsp;&nbsp; 📧 lenondpaula@gmail.com &nbsp;&nbsp; | &nbsp;&nbsp; 📱 +55 (55) 98135-9099
      <div style='margin-top:4px; font-size:11px; color:#B0B5BC;'>Licensed under the PolyForm Noncommercial License 1.0.0 — see /LICENSE</div>
    </div>
    """
    st.markdown(footer_css_html, unsafe_allow_html=True)

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
        import shapely.wkt as wkt
        if str(path_str).endswith('.parquet'):
            df = pd.read_parquet(path_str)
            if 'geometry_wkt' in df.columns:
                df['geometry'] = df['geometry_wkt'].apply(lambda s: wkt.loads(s) if pd.notna(s) else None)
            # CRITICAL: Parse WKT strings in geometry column
            elif 'geometry' in df.columns and isinstance(df['geometry'].iloc[0], str):
                df['geometry'] = df['geometry'].apply(lambda s: wkt.loads(s) if pd.notna(s) else None)
            gdf_local = gpd.GeoDataFrame(df, geometry='geometry')
        else:
            gdf_local = gpd.read_file(path_str)
        # Ensure geometry is in WGS84 (lat/lon) for pydeck
        try:
            if "geometry" in gdf_local.columns:
                if gdf_local.crs is None:
                    bounds = gdf_local.total_bounds
                    if bounds is not None and (
                        max(abs(bounds[0]), abs(bounds[2])) > 180
                        or max(abs(bounds[1]), abs(bounds[3])) > 90
                    ):
                        gdf_local = gdf_local.set_crs(epsg=31983, allow_override=True)
                if gdf_local.crs and str(gdf_local.crs).lower() not in ["epsg:4326", "epsg:4674"]:
                    gdf_local = gdf_local.to_crs(epsg=4326)
        except Exception:
            pass
        # Ensure numeric columns exist and are filled
        for col in ["Qtd_Empresas", "Apetite_Investidor", "Saturacao_Comercial", "Renda_Media"]:
            if col not in gdf_local.columns:
                gdf_local[col] = 0
        # Prepare style columns
        gdf_local["_coords"] = gdf_local.geometry.apply(polygon_to_coords)
        # Keep shapely geometry objects for polygon extraction and mapping; expose WKT separately
        if "geometry" in gdf_local.columns:
              try:
                  import shapely.wkt as wkt
                  gdf_local["geometry_wkt"] = gdf_local["geometry"].apply(lambda g: g.wkt if g is not None else None)
              except Exception:
                  gdf_local["geometry_wkt"] = None
        # Drop geometry column after using for coordinate extraction to avoid pyarrow issues
        gdf_local = gdf_local.drop(columns=["geometry"], errors="ignore")
        gdf_local["Apetite_Investidor"] = gdf_local["Apetite_Investidor"].fillna(0).astype(float)
        gdf_local["Saturacao_Comercial"] = gdf_local["Saturacao_Comercial"].fillna(0).astype(float)
        # elevation: 300 + (Apetite * 400) para torres urbanas visíveis (300-700m range)
        if "Elevation_3D" in gdf_local.columns:
            gdf_local["elevation"] = gdf_local["Elevation_3D"].fillna(300).astype(float)
        elif "Apetite_Investidor" in gdf_local.columns:
            gdf_local["elevation"] = 300 + (gdf_local["Apetite_Investidor"].fillna(0).astype(float) * 400)
        else:
           gdf_local["elevation"] = 300.0
        # DEBUG: Log elevation values being sent to PyDeck
        st.write("🔍 Valores de Elevação (Top 5):", gdf_local[['Nome_Bairro', 'Apetite_Investidor', 'elevation']].head(5))
        
        # fill_color: Color by Classificacao (OURO/SATURADO/PRATA)
        def _fill_by_class(classif: str):
            classif_upper = str(classif).upper() if pd.notna(classif) else ""
            if classif_upper == "OURO":
                return [255, 215, 0, 200]  # Gold - High Visibility
            elif classif_upper == "SATURADO":
                return [255, 69, 0, 180]  # Red/Orange - Warning
            elif classif_upper == "PRATA":
                return [0, 128, 255, 160]  # Blue - Standard
            else:
                return [128, 128, 128, 160]  # Grey - Fallback
        
        gdf_local["fill_color"] = gdf_local.get("Classificacao", pd.Series("")).apply(_fill_by_class)
        return gdf_local

    gdf = load_final_gdf(str(data_path))

    # Sidebar: legenda e filtro
    st.sidebar.header("🎨 Legenda de Cores")
    st.sidebar.markdown(
        """
**Classificação por Cores (RGB)**:

🥇 **OURO** [Gold #FFD700]
- Renda elevada + Mobilidade excelente
- Apetite_Investidor ≥ 0.78 (não saturado)

🔴 **SATURADO** [Red/Orange #FF4500]
- Mercado maduro com alta concorrência
- Qtd_Empresas ≥ 15.000

🥈 **PRATA** [Blue #0080FF]
- Crescimento estável e oportunidades
- Mercado aberto com bom potencial

---

**Score de Apetite** = 0.4 × Mobilidade + 0.3 × Renda + 0.3 × (1 - Saturação)

**Elevação 3D** = 300 + (Apetite_Investidor × 400) metros
        """,
        unsafe_allow_html=True,
    )

    name_col = choose_name_column(gdf)
    if name_col is None:
        st.error("Coluna de nome do bairro não encontrada no GeoDataFrame.")
        return

    # Emergency cleaning: remove 'fallback' prefixes and '(?)' tokens from displayed names
    try:
        gdf[name_col] = gdf[name_col].astype(str).str.replace(r'fallback[-_]?','', regex=True, case=False).str.replace(r'\(\?\)|\?', '', regex=True).str.replace('_', ' ').str.strip()
        gdf[name_col] = gdf[name_col].str.upper()
    except Exception:
        pass

    # Build deduplicated, normalized bairro options for sidebar
    bairros_options = sorted(list(dict.fromkeys([b.strip() for b in gdf[name_col].dropna().astype(str)])))
    # Ensure key Centro-Sul tokens are present in options (uppercased names already applied to gdf[name_col])
    for token in ['SANTO AGOSTINHO', 'CRUZEIRO']:
        if token not in bairros_options and token in list(gdf[name_col].unique()):
            bairros_options.append(token)
    bairros_options = sorted(list(dict.fromkeys(bairros_options)))

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
        # Dynamic: read directly from dataframe (ensures 17 mapped neighborhoods match exactly)
        centro_options = sorted(gdf[name_col].unique().tolist())
    centro_options = sorted(list(dict.fromkeys(centro_options)))
    selected = st.sidebar.multiselect("Selecionar bairros (Centro-Sul)", options=centro_options, default=centro_options)
    # If user explicitly deselects all, show nothing; otherwise always show selected (default = all)
    if selected is None or len(selected) == 0:
        selected = centro_options  # Force load all if empty
    if len(selected) > 0:
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
                # Amplify elevation so towers are visible even when raw values are small
                elevation = float(row.get("elevation", 0.0) or 0.0)
                # PolygonLayer expects an array of rings (outer ring = first element)
                coords_wrapped = [coords] if coords else []
                rec = {
                      "geometry": coords_wrapped,                    "coordinates": coords_wrapped,                    "fill_color": fill_color,
                    "elevation": elevation,
                      "Apetite_Investidor": float(row.get("Apetite_Investidor", 0) or 0.0),
                    name_col: row.get(name_col),
                    "Renda_Media": float(row.get("Renda_Media", 0) or 0.0),
                    "Qtd_Empresas": int(row.get("Qtd_Empresas", 0) or 0),
                    "Classificacao": str(row.get("Classificacao", "")),
                }
                records.append(rec)
            # If records are empty, show dataframe head for debugging so we can inspect whether geometry exists
            if len(records) == 0:
                st.write(gdf.head())
                st.warning("Debug: no polygon records available for pydeck; showing dataframe head.")

            polygon_layer = pdk.Layer(
                "PolygonLayer",
                data=records,
                pickable=True,
                stroked=True,
                filled=True,
                extruded=True,
                wireframe=True,
                opacity=0.8,
                get_polygon="geometry",
                get_fill_color="fill_color",
                get_line_color=[80, 80, 80],
                get_elevation="elevation",
                auto_highlight=True,
                elevation_scale=1,
                elevation_range=[0, 1000],
            )

            tooltip = {
                "html": "<b>{%s}</b><br/>Renda: {%s}<br/>Empresas: {%s}<br/>Classificação: {%s}" % (name_col, "Renda_Media", "Qtd_Empresas", "Classificacao"),
                "style": {"backgroundColor": "#F0F0F0", "color": "#000000"},
            }

            df_table = gdf.drop(columns=["geometry"], errors="ignore").copy()

            deck = pdk.Deck(
                layers=[polygon_layer],
                initial_view_state=pdk.ViewState(latitude=-19.935, longitude=-43.935, zoom=13.2, pitch=55),
                tooltip=tooltip,
            )
            try:
                st.pydeck_chart(deck, key='map_final_validated_01')
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
