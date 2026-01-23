# BH Strategic Navigator (bh_etl_task)

Projeto MVP para seleção de locais em Belo Horizonte com pipeline ETL e visualização interativa.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Geração de Dados Sintéticos

```bash
python3 src/generate_mock_data.py
```

Arquivos gerados:
- [data/bairros_data.geojson](data/bairros_data.geojson)
- [data/dados_economicos.csv](data/dados_economicos.csv)

## Executar ETL

```bash
python3 src/etl_engine.py
```

Saídas:
- [data/bh_final_data.geojson](data/bh_final_data.geojson)
- [data/data_final.parquet](data/data_final.parquet)

## Executar Aplicação (Streamlit)

```bash
streamlit run app.py
```

App disponível em http://localhost:8501.

## Testes

```bash
pytest -q
```

## Docker

```bash
docker build -t bh_strategic_navigator:latest .
docker run --rm -p 8501:8501 bh_strategic_navigator:latest
```

## Arquitetura proposta

Veja [ARQUITETURA_PROPOSTA.md](ARQUITETURA_PROPOSTA.md) para a evolução do MVP para um Data Lakehouse na AWS.