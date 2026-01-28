
# Guia para Agentes de IA — BH Strategic Navigator

## Visão Geral
Projeto MVP para análise de oportunidades de investimento por bairro em BH, combinando pipeline ETL (Polars, Pandas, GeoPandas) e dashboard Streamlit. Foco em arquitetura prática, baixo consumo de memória e integração de dados geográficos/econômicos.

## Componentes e Fluxo
- **ETL**: [src/etl_engine.py](src/etl_engine.py) — Orquestra leitura (Polars), processamento (Pandas), geometria (GeoPandas), normalização (`normalize_text`), merge e scoring. Entrada: `data/dados_economicos.csv`, `data/bairros_data.geojson`. Saída: `data/bh_final_data.geojson`, `data/data_final.parquet`.
- **Dashboard**: [app.py](app.py) — Visualização interativa (PyDeck, Plotly), abas Mapa 3D e Scatter, integração direta com GeoDataFrames.
- **Mock Data**: [src/generate_mock_data.py](src/generate_mock_data.py) — Geração de fixtures sintéticos para testes/demos.

## Padrões e Convenções
- **Normalização de nomes**: Sempre use `normalize_text` antes de joins entre CSV e GeoJSON (`Nome_Bairro_NORM`).
- **Scores**: Use `minmax_scale_series` para normalizar [0,1] e combine scores via operações vetorizadas (ex: `Apetite_Investidor`).
- **Classificação**: Regras diretas em Pandas, sem classes formais. Exemplo: "OPORTUNIDADE DE OURO", "REGULAR", "SATURADO".
- **Logging**: Use `logging.getLogger("etl_engine")` para etapas críticas.
- **Tratamento de NaN**: Preencha NaN com 0 antes de escalar; após merge, preencha campos faltantes com 0 ou NaN conforme contexto.

## Workflows Essenciais
- **Setup**: Instale dependências em blocos (leve → pesado) para evitar OOM:
  - Leve: `pip install polars pytest faker black flake8`
  - Médio: `pip install numpy pandas scikit-learn`
  - Pesado: `pip install geopandas fiona rtree shapely` (swap ativo)
  - UI: `pip install streamlit plotly pydeck`
- **Geração de dados de teste**: `python3 src/generate_mock_data.py`
- **Executar ETL**: `python3 src/etl_engine.py`
- **Dashboard**: `streamlit run app.py` (porta 8501)
- **Testes**: `pytest -v` ou `pytest tests/ --cov=src`

## Boas Práticas Específicas
- Priorize Polars para leitura de CSVs grandes (streaming/chunks).
- Limite threads: `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1`.
- Monitore swap/memória (`free -h`, `top`).
- Antes de rodar Streamlit, confirme Security Group liberando porta 8501.

## Exemplos de Código
```python
gdf["Nome_Bairro_NORM"] = gdf["Nome_Bairro"].apply(normalize_text)
df_econ["Nome_Bairro_NORM"] = df_econ["Nome_Bairro"].apply(normalize_text)
merged = gdf.merge(df_econ, on="Nome_Bairro_NORM", how="left")
apetite = (0.4 * Score_Renda + 0.3 * Score_Mobilidade - 0.3 * Saturacao_Comercial).clip(0, 1)
```

## Referências Rápidas
- [README.md](README.md): Setup e uso
- [ARQUITETURA_PROPOSTA.md](ARQUITETURA_PROPOSTA.md): Visão futura (não implementar)
- `data/`: Dados de entrada/saída
- `tests/`: Testes de integração

**Este documento reflete a arquitetura real e deve ser seguido por agentes de IA.**
