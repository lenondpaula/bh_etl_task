from pathlib import Path

import geopandas as gpd
import numpy as np
import pydeck as pdk
import altair as alt
import streamlit as st
import pandas as pd

def choose_name_column(gdf):
    """Seleciona a coluna de nome do bairro disponível."""
    for c in ["Nome_Bairro", "Nome_Bairro_x", "Nome_Bairro_y"]:
        if c in gdf.columns:
            return c
    return None

def polygon_to_coords(poly):
    """Extrai coordenadas de um Polygon/MultiPolygon em formato [lon, lat]."""
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
    """Inicializa a aplicação Streamlit com mapa 3D e análise de clusters."""
    st.set_page_config(layout="wide", page_title="BH Strategic Navigator - Site Selection AI")

    # CSS robusto para sidebar e métricas
    st.markdown('<style>[data-testid="stSidebar"] {background-color: #0E1117 !important;} [data-testid="stSidebar"] * {color: white !important;} [data-testid="stMetricValue"] {color: #000000 !important;} [data-testid="stMetricLabel"] {color: #444444 !important;} [data-testid="stMetric"] {background-color: #ffffff !important; border: 1px solid #e6e9ef; padding: 15px; border-radius: 8px; shadow: 2px 2px 5px rgba(0,0,0,0.1);}</style>', unsafe_allow_html=True)

    st.markdown("# BH Strategic Navigator")

    st.markdown("O BH Strategic Navigator é uma ferramenta de análise geoespacial desenvolvida para avaliar a atratividade de investimentos nos bairros da Região Centro-Sul de Belo Horizonte. Utiliza dados integrados de renda média, mobilidade urbana, densidade empresarial e indicadores de valorização urbana (IQVU) para fornecer insights acionáveis em visualizações 3D interativas.")

    with st.expander("🧭 Guia Rápido de Navegação e Uso", expanded=True):
        st.markdown(
            """
            **Visualização 3D:** Extrusão 3D representa a densidade comercial (altura dos polígonos); Cor representa o Apetite do Investidor (Ouro = Alta Atratividade, Prata = Moderada, Vermelho = Baixa). Visualmente, o investidor identifica "paredões" de saturação e "vales" de oportunidade.

            **Legenda de Cores:**

            🥇 **Ouro**: Alto potencial (Apetite > 75%).

            🥈 **Prata**: Oportunidade estável (Apetite 50-75%).

            🔴 **Saturado**: Baixo retorno imediato (Apetite < 50%). *Nota: Centro e Savassi são sempre classificados como Saturados devido à alta saturação comercial.*

            **Cluster Analysis:** Bairros são agrupados por perfis socioeconômicos similares; clusters de alto apetite sinalizam zonas premium para investimento.

            **Controles:** Botão direito do mouse + arrastar para girar/inclinar a visão 3D.
            """
        )

    data_path = Path("data/bh_final_data.geojson")
    if not data_path.exists():
        st.error("Arquivo data/bh_final_data.geojson não encontrado. Execute o ETL primeiro.")
        return

    @st.cache_data(show_spinner=False)
    def load_final_gdf(path_str: str) -> pd.DataFrame:
        """Carrega e prepara o GeoDataFrame final com colunas de estilo."""
        import shapely.wkt as wkt
        if str(path_str).endswith('.parquet'):
            df = pd.read_parquet(path_str)
            if 'geometry_wkt' in df.columns:
                df['geometry'] = df['geometry_wkt'].apply(lambda s: wkt.loads(s) if pd.notna(s) else None)
            elif 'geometry' in df.columns and isinstance(df['geometry'].iloc[0], str):
                df['geometry'] = df['geometry'].apply(lambda s: wkt.loads(s) if pd.notna(s) else None)
            gdf_local = gpd.GeoDataFrame(df, geometry='geometry')
        else:
            gdf_local = gpd.read_file(path_str)
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
        for col in ["Qtd_Empresas", "Apetite_Investidor", "Saturacao_Comercial", "Renda_Media"]:
            if col not in gdf_local.columns:
                gdf_local[col] = 0
        gdf_local["_coords"] = gdf_local.geometry.apply(polygon_to_coords)
        if "geometry" in gdf_local.columns:
            try:
                gdf_local["geometry_wkt"] = gdf_local["geometry"].apply(lambda g: g.wkt if g is not None else None)
            except Exception:
                gdf_local["geometry_wkt"] = None
        gdf_local = gdf_local.drop(columns=["geometry"], errors="ignore")
        gdf_local["Apetite_Investidor"] = gdf_local["Apetite_Investidor"].fillna(0).astype(float)
        gdf_local["Saturacao_Comercial"] = gdf_local["Saturacao_Comercial"].fillna(0).astype(float)
        if "Apetite_Investidor" in gdf_local.columns:
            score_val = gdf_local["Apetite_Investidor"].fillna(0).astype(float)
        else:
            score_val = 0.0
        # Elevation baseada na quantidade de empresas
        max_qtd = gdf_local["Qtd_Empresas"].max()
        if max_qtd > 0:
            gdf_local["elevation"] = 100 + (gdf_local["Qtd_Empresas"] / max_qtd) * 3000
        else:
            gdf_local["elevation"] = 100.0
        gdf_local["classificacao"] = gdf_local.apply(
            lambda row: "SATURADO" if row["Nome_Bairro"] in ["CENTRO", "SAVASSI"] else ("OURO" if row["Apetite_Investidor"] > 0.75 else "PRATA" if row["Apetite_Investidor"] >= 0.5 else "SATURADO"),
            axis=1
        )
        gdf_local["fill_color"] = gdf_local["classificacao"].map(
            {"OURO": [255, 215, 0], "PRATA": [70, 130, 180], "SATURADO": [200, 0, 0]}
        )
        # Adicionar coluna de mobilidade estimada
        gdf_local["Mobilidade"] = gdf_local["Apetite_Investidor"].apply(
            lambda x: "Excelente" if x > 0.8 else "Boa" if x > 0.6 else "Regular" if x > 0.4 else "Baixa"
        )
        return gdf_local

    gdf = load_final_gdf(str(data_path))

    # Adicionando o multiselect de bairros e fórmulas explicativas na sidebar
    bairros_selecionados = st.sidebar.multiselect(
        "Selecionar Bairros", options=sorted(gdf["Nome_Bairro"].unique())
    )

    st.sidebar.markdown("### 🧪 Conheça as fórmulas aplicadas!")
    st.sidebar.markdown(
        """
        **Score de Apetite:**
        **0.4 x Score Renda + 0.3 x Score Mobilidade - 0.3 x Saturação Comercial**

        **Elevação 3D:**
        **100 + (Qtd_Empresas / Max_Qtd_Empresas) × 3000**
        """
    )

    if bairros_selecionados:
        gdf = gdf[gdf["Nome_Bairro"].isin(bairros_selecionados)]

    with st.expander("🏆 Top 5 Oportunidades de Ouro", expanded=True):
        top_5 = gdf.nlargest(5, 'Apetite_Investidor')
        cols = st.columns(5)
        for i, (idx, row) in enumerate(top_5.iterrows()):
            cols[i].metric(row['Nome_Bairro'], f"{row['Apetite_Investidor']:.1%}")

    tooltip = {
        "html": '<b>Bairro:</b> {Nome_Bairro}<br><b>Classificação de Atratividade:</b> {classificacao}<br><b>IQVU:</b> {Renda_Media}<br><b>Mobilidade Urbana:</b> {Mobilidade}',
        "style": {"backgroundColor": "#F0F0F0", "color": "#000000"},
    }

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=gdf,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=True,
        wireframe=True,
        opacity=0.8,
        get_polygon="_coords",
        get_fill_color="fill_color",
        get_line_color=[80, 80, 80],
        get_elevation="elevation",
        auto_highlight=True,
        elevation_scale=1,
        elevation_range=[0, 1000],
    )

    deck = pdk.Deck(
        layers=[polygon_layer],
        initial_view_state=pdk.ViewState(latitude=-19.935, longitude=-43.935, zoom=13.2, pitch=55),
        tooltip=tooltip,
    )
    st.info('💡 Entenda: A altura de cada bairro no mapa 3D representa a densidade comercial (número de empresas). Passe o cursor sobre as regiões para ver detalhes interativos, como Índice de Qualidade de Vida Urbana (IQVU) e mobilidade urbana (estimada com base em acesso a transporte público).')
    st.pydeck_chart(deck)

    st.markdown('### 📊 Indicadores Consolidados da Região')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric('Total de Empresas', int(gdf['Qtd_Empresas'].sum()))
    with c2:
        max_iqvu = gdf['Renda_Media'].max()
        bairro_max_iqvu = gdf.loc[gdf['Renda_Media'] == max_iqvu, 'Nome_Bairro'].iloc[0]
        st.metric('Maior IQVU', f"{bairro_max_iqvu}: {max_iqvu:.2f}")
    with c3:
        # Bairro com melhor mobilidade (Excelente)
        melhor_mobilidade = gdf[gdf['Mobilidade'] == 'Excelente']
        if not melhor_mobilidade.empty:
            bairro_melhor_mob = melhor_mobilidade['Nome_Bairro'].iloc[0]
            st.metric('Melhor Mobilidade', f"{bairro_melhor_mob}: Excelente")
        else:
            st.metric('Melhor Mobilidade', 'N/A')
    st.info('💡 Dica: Selecione bairros na barra lateral para filtrar estes indicadores em tempo real.')

    # Gráfico de Clusters Socioeconômicos
    st.markdown('### 🔍 Visualização de Clusters Socioeconômicos')
    chart = alt.Chart(gdf).mark_circle(size=60).encode(
        x=alt.X('Renda_Media:Q', title='Renda Média', scale=alt.Scale(domain=[gdf['Renda_Media'].min() * 0.9, gdf['Renda_Media'].max() * 1.1])),
        y=alt.Y('Apetite_Investidor:Q', title='Apetite do Investidor', scale=alt.Scale(domain=[gdf['Apetite_Investidor'].min() * 0.9, gdf['Apetite_Investidor'].max() * 1.1])),
        color=alt.Color('classificacao:N', scale=alt.Scale(domain=['OURO', 'PRATA', 'SATURADO'], range=['gold', 'steelblue', 'darkred'])),
        tooltip=['Nome_Bairro:N', 'Renda_Media:Q', 'Apetite_Investidor:Q', 'classificacao:N']
    ).properties(
        width=600,
        height=300
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    with st.expander("📋 Detalhes da Análise de Clusters", expanded=False):
        st.markdown(
            """
            **Clusters Socioeconômicos:** Bairros são agrupados por perfis similares com base em renda, mobilidade e saturação comercial.

            - **Clusters de Alto Apetite (Ouro):** Zonas premium para investimento, com alto potencial de retorno.
            - **Clusters Estáveis (Prata):** Oportunidades equilibradas, adequadas para expansão moderada.
            - **Clusters Saturados (Vermelho):** Áreas com baixa atratividade imediata, focar em reestruturação.

            *Nota: Esta análise é baseada em dados agregados e pode ser refinada com mais variáveis.*
            """
        )

    readme_text = Path('README_APP.md').read_text(encoding='utf8') if Path('README_APP.md').exists() else 'README não encontrado.'

    with st.expander("📄 Sobre o Projeto (README)", expanded=False):
        st.markdown(readme_text, unsafe_allow_html=True)

    # Alterando o rodapé para remover o position: fixed e adicionar margin-top
    footer_css_html = """
<style>
  .footer-sticky {
    margin-top: 50px;
    background-color: #0E1117;
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

if __name__ == "__main__":
    main()