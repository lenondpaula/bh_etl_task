# BH Strategic Navigator - MVP

Visualização 3D interativa de oportunidades de investimento por bairro em Belo Horizonte.

## 🎯 Sobre o Projeto

**BH Strategic Navigator** é uma ferramenta de análise geoespacial que identifica oportunidades de investimento na região Centro-Sul de Belo Horizonte através de:

- **Pipeline ETL** robusta (Polars → Pandas → GeoPandas)
- **Visualização 3D interativa** com PyDeck
- **Classificação inteligente** de bairros por potencial de investimento
- **Dashboard Streamlit** com controles de filtro em tempo real

---

## 📊 Visualização & Cores

### Escala de Cores por Classificação

| Classificação | Cor | RGB | Significado |
|---|---|---|---|
| **OURO** 🥇 | Ouro | `[255, 215, 0]` | Alta oportunidade (Renda elevada + Mobilidade excelente) |
| **SATURADO** 🔴 | Vermelho/Laranja | `[255, 69, 0]` | Mercado saturado (Muitas empresas, alta concorrência) |
| **PRATA** 🥈 | Azul | `[0, 128, 255]` | Crescimento estável (Bom potencial, mercado aberto) |

### 3D Elevation Scale

- **Escala**: 0-3000 metros (exagero visual para impacto)
- **Fórmula**: `Elevation_3D = Apetite_Investidor × 3000`
- **Baseado em**: Score de mobilidade (40%) + Score de renda (30%) + Inverso de saturação (30%)

---

## 🏘️ Dados & Coverage

### 17 Bairros Centro-Sul Mapeados

**OURO** (4):
- SERRA (6.001 empresas, Apetite 0.881)
- BELVEDERE (160 empresas, Apetite 0.814)
- SÃO LUCAS (912 empresas, Apetite 0.791)
- VILA PARIS (735 empresas, Apetite 0.788)

**SATURADO** (2):
- CENTRO (25.546 empresas)
- SAVASSI (20.622 empresas)

**PRATA** (11):
- SION, ANCHIETA, SANTO ANTÔNIO, CRUZEIRO, SÃO PEDRO, FUNCIONÁRIOS, BOA VIAGEM, LOURDES, CARMO, MANGABEIRAS, SANTA EFIGÊNIA

### Fonte de Dados

- **Geometrias**: Convex hull dos pontos de empresas (cadastro_empresas_centro_sul.csv)
- **Empresas**: 106.166 registros com localização UTM (EPSG:31983 → EPSG:4326)
- **Renda**: Índice IQVU (Índice de Qualidade de Vida Urbana) × 1000
- **Bounding Box Rigoroso**: Lat [-19.965, -19.900] × Lon [-43.980, -43.910]

---

## 🚀 Quick Start

### Setup Ambiente

```bash
# Clone e setup
cd bh_etl_task
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Executar ETL

```bash
python3 rebuild_final_clean.py
```

**Output**:
- `data/bh_final_data.geojson` - Geometrias + atributos
- `data/data_final.parquet` - Dados tabulares

### Iniciar Dashboard

```bash
streamlit run app.py
```

Acesse em `http://localhost:8501`

---

## 📋 Funcionalidades do Dashboard

### Aba 1: Mapa 3D

- **Visualização PyDeck** com torres 3D coloridas por classificação
- **Camera Inicial**: Latitude -19.935, Longitude -43.935, Zoom 13.2, Pitch 55°
- **Interatividade**: Clique nos polígonos para tooltip com:
  - Nome do bairro
  - Renda média (R$)
  - Quantidade de empresas
  - Classificação
  - Score de apetite de investidor

### Aba 2: Análise Scatter

- **Eixo X**: Renda Média
- **Eixo Y**: Quantidade de Empresas
- **Cor**: Classificação (OURO/SATURADO/PRATA)
- **Tamanho**: Score de apetite de investidor

### Sidebar

- **Seleção de Bairros**: Multiselect com todos os 17 bairros (padrão: todos)
- **Métricas Resumidas**:
  - Total de empresas na seleção
  - Bairro com maior renda média
  - Bairro com maior mobilidade

---

## 🏗️ Arquitetura

### ETL Pipeline

```
cadastro_empresas_centro_sul.csv (106K pontos UTM)
                 ↓
    [Filtro Bounding Box - Guilhotina]
    Lat: -19.965 a -19.900
    Lon: -43.980 a -43.910
                 ↓
      [Convex Hull por Bairro]
                 ↓
    [Reprojetar UTM → WGS84]
                 ↓
    [Join com IQVU & Empresas]
                 ↓
    [Calcular Scores & Classificação]
                 ↓
    GeoJSON + Parquet (17 bairros)
```

### Lógica de Classificação

```python
if Qtd_Empresas >= 15000:
    Classificacao = "SATURADO"
elif Apetite_Investidor >= 0.78:
    Classificacao = "OURO"
else:
    Classificacao = "PRATA"
```

---

## 📊 Scores & Normalização

### Score de Apetite de Investidor

```
Apetite = 0.4 × Score_Mobilidade + 0.3 × Score_Renda + 0.3 × (1 - Score_Saturacao)
Range: [0, 1]
```

### Normalização Min-Max

Todos os scores normalizados em [0, 1] usando `sklearn.preprocessing.MinMaxScaler`

---

## 🧪 Testes

```bash
pytest -v
pytest --cov=src
```

**Coverage**: Testes de integração ETL, normalização de texto, cálculo de scores

---

## 🐳 Docker

```bash
docker build -t bh_strategic_navigator:latest .
docker run --rm -p 8501:8501 bh_strategic_navigator:latest
```

---

## 📁 Estrutura de Diretórios

```
.
├── app.py                          # Dashboard Streamlit
├── rebuild_final_clean.py          # Script ETL principal
├── requirements.txt                # Dependências runtime
├── requirements-dev.txt            # Dependências dev
├── data/
│   ├── bairros_limites.csv         # Limites de todos os bairros BH
│   ├── cadastro_empresas_centro_sul.csv  # 106K empresas com coordenadas
│   ├── Score_Renda.csv             # IQVU por bairro
│   ├── bh_final_data.geojson       # Output ETL: geometrias
│   └── data_final.parquet          # Output ETL: tabular
├── tests/
│   ├── test_integration_etl.py
│   ├── test_scoring.py
│   └── test_text_normalization.py
└── README.md                       # Este arquivo
```

---

## 🔄 Workflow de Desenvolvimento

1. **Modificar dados ETL** → editar `rebuild_final_clean.py`
2. **Rodar rebuild** → `python3 rebuild_final_clean.py`
3. **Atualizar UI** → editar `app.py`
4. **Testar** → `pytest -v`
5. **Deploy** → `nohup streamlit run app.py &`

---

## 📚 Referências

- [ARQUITETURA_PROPOSTA.md](ARQUITETURA_PROPOSTA.md) - Visão de evolução para Data Lakehouse AWS
- [copilot-instructions.md](.github/copilot-instructions.md) - Guia para agentes de IA

---

## 📞 Suporte

Para issues e sugestões, abra uma issue ou PR neste repositório.

---

**Última atualização**: Jan 2026 | **Versão**: 1.0.0 MVP
