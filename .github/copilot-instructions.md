# Instruções para Agentes de IA - BH Strategic Navigator

## Visão Geral do Projeto

**BH Strategic Navigator** (bh_etl_task) é um MVP que combina um pipeline ETL funcional com dashboard interativo Streamlit para análise de oportunidades de investimento por bairro em Belo Horizonte. Diferente da estrutura aspiracional anterior, o projeto atual é **produção-ready** com arquitetura prática e testada.

## Arquitetura Atual (Real)

### Componentes Principais

1. **ETL Engine** ([src/etl_engine.py](src/etl_engine.py))
   - Núcleo funcional que orquestra o pipeline: Polars (leitura CSV) → Pandas (processamento) → GeoPandas (geometria) → Saída GeoJSON + Parquet
   - Funções criticas: `run_etl()` (orquestrador), `normalize_text()` (limpeza), `minmax_scale_series()` (normalização min-max)
   - Entrada: `data/dados_economicos.csv` + `data/bairros_data.geojson`
   - Saída: `data/bh_final_data.geojson` + `data/data_final.parquet`

2. **Geração de Dados Sintéticos** ([src/generate_mock_data.py](src/generate_mock_data.py))
   - Cria dados de teste com Faker + GeoPandas (geometrias quadradas para bairros)
   - Usado em testes e demos; parametrizável (count, seed)

3. **Dashboard Streamlit** ([app.py](app.py))
   - Visualização interativa com 2 abas: Mapa 3D (PyDeck) + Scatter (Plotly)
   - Integração direta com GeoDataFrames e operações NumPy/Pandas
   - Estilo corporativo minimalista (CSS customizado)

### Fluxo de Dados

```
CSV (economia) ──┐
                 ├─→ Polars/Pandas ──→ normalize_text() ──→ merge() ──→ 
GeoJSON (geo) ───┘                      minmax_scale() ─→ classify()

                                         ↓
                            GeoJSON final + Parquet
                                         ↓
                                    Streamlit App
```

## Padrões e Convenções Reais

### Estrutura de Diretórios

```
.
├── src/
│   ├── etl_engine.py          # Pipeline ETL principal
│   ├── generate_mock_data.py  # Geração de fixtures
│   └── __pycache__/
├── data/                       # Entrada/saída de dados
│   ├── bairros_data.geojson
│   ├── dados_economicos.csv
│   ├── bh_final_data.geojson  # Saída ETL
│   └── data_final.parquet     # Saída ETL
├── tests/                      # Testes integração
│   ├── test_integration_etl.py # Teste principal de output
│   ├── test_app.py            # Smoke test Streamlit
│   ├── test_text_normalization.py
│   └── test_scoring.py
├── app.py                      # Dashboard Streamlit
├── requirements.txt            # Runtime (Streamlit, Polars, GeoPandas, etc)
├── requirements-dev.txt        # Dev (pytest, pytest-cov)
├── Dockerfile
├── ARQUITETURA_PROPOSTA.md     # Visão futura: Data Lakehouse na AWS
└── README.md
```

### Convenções de Código Reais (Não Aspiracionais)

- **Linguagem**: Python 3.8+
- **Type Hints**: Usadas em funções públicas e privadas (`str | None`, `pd.Series`, etc.)
- **Estilo**: PEP 8 com quebras pragmáticas (ver `app.py` para CSS inline)
- **Normalização**: Função `normalize_text()` trata acentos e uniformização de nomes (padrão para join entre CSV e GeoJSON)
- **Escalas e Scores**: Min-Max (0-1) aplicada com `minmax_scale_series()` permitindo NaN; resulta em scores: `Apetite_Investidor`, `Score_Renda`, `Score_Mobilidade`
- **Classificação**: Regras vetorizadas direto em Pandas (sem classes formais): "REGULAR", "OPORTUNIDADE DE OURO", "SATURADO"

### Logging

- `logging` padrão do Python com `getLogger("etl_engine")`
- Logs informativos para etapas críticas (leitura, normalização, merge, output)
- Não há rastreamento de ID de execução (pode ser adicionado)

## Workflows Críticos

### Setup Desenvolvimento
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Gerar Dados de Teste
```bash
python3 src/generate_mock_data.py
```
Cria `data/bairros_data.geojson` + `data/dados_economicos.csv` (50 bairros por padrão)

### Executar ETL
```bash
python3 src/etl_engine.py
```
Lê os CSVs/GeoJSONs e gera `data/bh_final_data.geojson` + `data/data_final.parquet`

### Executar Dashboard
```bash
streamlit run app.py
```
Abre em http://localhost:8501

### Testes
```bash
pytest -v                           # Todos os testes
pytest tests/test_integration_etl.py -v  # Apenas integração
pytest tests/ --cov=src             # Com cobertura
```

## Dependências Principais

- **streamlit** (1.0+): Dashboard interativo
- **polars**: Leitura eficiente de CSV
- **geopandas**: Processamento de geometrias; merge com CSVs
- **pandas**: Transformação e merge de dataframes
- **numpy**: Operações numéricas (escalas, NaN handling)
- **scikit-learn**: MinMaxScaler para normalização
- **pydeck**: Mapas 3D interativos
- **plotly**: Gráficos scatter e análises
- **faker** (opcional): Geração de dados de teste
- **pytest**, **pytest-cov** (dev): Testes unitários/integração

## Padrões Específicos do Projeto

### 1. Join entre Dados Geográficos e Econômicos
O projeto executa merge entre GeoDataFrame (bairros com geometria) e DataFrame econômico (CSV) usando **normalização de nomes**:
- GeoPandas carrega GeoJSON com coluna "Nome_Bairro"
- CSV lido com Polars é convertido para Pandas
- Ambos criam coluna `Nome_Bairro_NORM` = `normalize_text(Nome_Bairro)`
- Merge é feito em `Nome_Bairro_NORM` com `how="left"` para garantir todas as geometrias

**Padrão de Código:**
```python
gdf["Nome_Bairro_NORM"] = gdf["Nome_Bairro"].apply(normalize_text)
df_econ["Nome_Bairro_NORM"] = df_econ["Nome_Bairro"].apply(normalize_text)
merged = gdf.merge(df_econ[...], on="Nome_Bairro_NORM", how="left")
```

### 2. Tratamento de Dados Inválidos
- NaN em séries numéricas: `minmax_scale_series()` converte para 0 antes de escalar
- Campos missing após merge: Não remove colunas, preenche com valores padrão (0 ou NaN conforme necessário)
- Sem Dead Letter Queue (projeto MVP); erros causam falha do ETL

### 3. Cálculo de Scores Compostos
Exemplo `Apetite_Investidor`:
```python
apetite = (0.4 * Score_Renda + 0.3 * Score_Mobilidade - 0.3 * Saturacao_Comercial).clip(0, 1)
```
- Scores individuais normalizados [0,1] com min-max
- Combinação linear com pesos fixos
- Clipping final para garantir [0,1]
- Classificação por limiares (ex: Apetite > 0.75 = "OURO")

### 4. Output Geojson + Parquet
- GeoJSON preserva geometria, permite visualização em MapBox/PyDeck
- Parquet: mesmos dados sem geometria, formato eficiente para análise futura
- Ambos salvos em `data/` após ETL

## Evolução Futura (Referência)

Ver [ARQUITETURA_PROPOSTA.md](ARQUITETURA_PROPOSTA.md) para visão de evolução para Data Lakehouse na AWS (S3, Glue, Athena, Airflow). **Não implemente agora** — este documento descreve a arquitetura MVP vigente.

## Guia para IA: Como Ser Produtivo

1. **Para novos transformadores**: Adicione funções em `etl_engine.py` seguindo assinatura `(pd.Series/pd.DataFrame) → pd.Series/pd.DataFrame` com type hints
2. **Para novos testes**: Use `pytest` + `tmp_path` fixture (ver `test_integration_etl.py`)
3. **Para novo dashboard**: Integre GeoDataFrames + Plotly/PyDeck em `app.py` (ver abas existentes)
4. **Para tratamento de erros**: Use `logging` em vez de `print()` e não silencie exceções
5. **Para novos dados de entrada**: Garanta normalização de nomes antes de joins; documente schema esperado

## Recursos Chave

- [README.md](README.md): Instruções de setup e uso
- [ARQUITETURA_PROPOSTA.md](ARQUITETURA_PROPOSTA.md): Visão de futuro (Data Lakehouse)
- `data/`: Fixtures para teste/desenvolvimento
- `tests/`: Padrão de testes integração; execute antes de commitar


## Diretrizes de Operação do Ambiente (EC2 t2.micro) ⚙️

**Perfil do Ambiente**
- Instância: AWS EC2 **t2.micro** (Free Tier) — **1 vCPU, 1 GB RAM física**.
- Swap: arquivo de swap de **2 GB** configurado para evitar travamentos.
- Acesso: **VS Code Tunnels (vscode.dev)**. Sistema: **Ubuntu 22.04 LTS**.

**Diretrizes obrigatórias (seguir rigorosamente)**
- **Instalação de pacotes**: **NUNCA** sugerir `pip install -r requirements.txt` de uma vez. Recomendar instalação em **blocos**: pacotes leves primeiro (ex.: `polars`, `pytest`, `faker`, `black`, `flake8`), depois pacotes médios (ex.: `numpy`, `pandas`, `scikit-learn`) e por último pacotes pesados/geo/UI (ex.: `geopandas`, `fiona`, `rtree`, `shapely`, `streamlit`, `pydeck`). Esclarecer sempre o risco de pico de memória e orientar o uso de swap e limites de threads.
- **Otimização de código**: Priorizar **Polars** (lazy frames/streaming/chunks) sobre **Pandas** para reduzir uso de RAM; quando ler arquivos usar streaming/chunks e evitar carregar datasets inteiros na memória.
- **Prevenção de OOM**: Ao rodar modelos de ML ou visualizações (Streamlit, PyDeck), **alertar** sobre consumo de RAM e sugerir reduzir o tamanho do dataset, limitar threads (`export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1`), e usar amostragem/execução em lote.
- **Gerenciamento de swap e memória**: Monitorar swap (`free -h`, `vmstat`, `top`/`htop`) e, se houver swapping excessivo, sugerir `gc.collect()` e limpezas de cache ou reduzir o trabalho. Avisar o usuário imediatamente se o processo começar a usar swap intensamente.
- **Exposição de portas**: Lembrar de configurar o **Security Group** da EC2 para portas específicas (ex.: **8501** para Streamlit) antes de iniciar servidores web.
- **Objetivo operacional**: Priorizar **estabilidade e baixo consumo de memória** acima de velocidade.

**Exemplo de instalação segura (em blocos)**
- Bloco leve: `pip install polars pytest faker black flake8`
- Bloco médio: `pip install numpy pandas scikit-learn`
- Bloco pesado/geo (usar swap ativo): `pip install geopandas fiona rtree shapely`
- UI (instalar somente quando necessário): `pip install streamlit plotly pydeck`

**Integração ao fluxo de trabalho do Copilot**
- Sempre aplicar estas diretrizes ao propor comandos, exemplos de execução, mudanças no projeto ou instruções ao usuário.
- Antes de executar tarefas pesadas, **confirmar** com o usuário que está ciente do consumo de memória e que o ambiente está pronto (swap ativo, Security Group configurado, limite de threads definido).

---

**Nota**: Este documento reflete a arquitetura *atual e produção-ready*. Use como fonte de verdade para padrões, não como aspiração futura.
