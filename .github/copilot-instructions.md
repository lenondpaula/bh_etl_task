# Instruções para Agentes de IA - BH ETL Task

## Visão Geral do Projeto
**bh_etl_task** é um pipeline ETL (Extract, Transform, Load) para processar dados da cidade de Belo Horizonte. O projeto está em fase inicial e será estruturado para extrair, transformar e carregar dados de diversas fontes.

## Arquitetura Esperada (a ser implementada)
- **Camada de Extração**: Conectores para diferentes fontes de dados (APIs, bancos de dados, arquivos)
- **Camada de Transformação**: Lógica de processamento e limpeza de dados
- **Camada de Carregamento**: Destinos de armazenamento (data warehouse, bancos de dados, data lakes)
- **Orquestração**: Agendamento e monitoramento de jobs ETL

## Padrões e Convenções

### Estrutura de Diretórios (quando implementada)
```
├── src/
│   ├── extractors/     # Módulos de extração
│   ├── transformers/   # Módulos de transformação
│   ├── loaders/        # Módulos de carregamento
│   ├── utils/          # Utilitários compartilhados
│   └── config/         # Configurações
├── tests/              # Testes unitários e de integração
├── scripts/            # Scripts de inicialização e utilitários
├── docs/               # Documentação
└── .github/
    └── workflows/      # CI/CD workflows
```

### Convenções de Código
- **Linguagem**: Python (recomendado para pipelines ETL)
- **Style Guide**: PEP 8
- **Type Hints**: Usar type hints em funções públicas
- **Nomes**: 
  - Classes de extractores: `*Extractor` (ex: `APIExtractor`, `DatabaseExtractor`)
  - Classes de transformadores: `*Transformer` (ex: `DataCleaner`, `GeometryTransformer`)
  - Classes de loaders: `*Loader` (ex: `PostgreSQLLoader`, `CSVLoader`)

### Tratamento de Configuração
- Variáveis de ambiente em arquivo `.env` (nunca commitar com dados sensíveis)
- Usar bibliotecas como `python-dotenv` ou `pydantic.settings`
- Exemplo: credenciais de BD, URLs de APIs, caminhos de dados

### Logging e Observabilidade
- Usar `logging` padrão do Python com níveis apropriados
- Registrar início/fim de cada etapa do ETL
- Incluir timestamps e IDs de execução para rastreabilidade

## Fluxos de Trabalho Críticos (a documentar)

### Setup do Desenvolvimento
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Executar Testes (quando implementados)
```bash
pytest tests/ -v
pytest tests/ -v --cov=src  # Com cobertura
```

### Executar Pipeline
```bash
python -m src.main  # Executar job padrão
python -m src.main --config config/job.yaml  # Com configuração customizada
```

## Integração de Dados

### Pontos de Integração Comuns
- **Bancos de Dados**: Suporte a PostgreSQL, MySQL, SQLite
- **APIs**: Tratamento de rate limiting, retry com backoff exponencial
- **Arquivos**: Suporte a CSV, JSON, Parquet, Excel
- **Data Sources Esperadas**: APIs da prefeitura de BH, dados geográficos (GIS), dados econômicos

### Tratamento de Erros e Dados Inválidos
- Implementar Dead Letter Queue para dados rejeitados
- Logs detalhados de erro com contexto completo
- Graceful degradation quando possível
- Notificações em caso de falhas críticas

## Dependências Principais (a instalar)
- **pandas**: Manipulação de dados tabulares
- **sqlalchemy**: Abstração de banco de dados
- **pydantic**: Validação de esquemas
- **requests**: Chamadas HTTP
- **python-dotenv**: Gerenciamento de variáveis de ambiente
- **pytest**: Framework de testes
- **geopandas**: Processamento de dados geoespaciais (se necessário)

## Próximas Prioridades de Implementação
1. Estrutura base de diretórios e configuração
2. Conectores para primeiras fontes de dados (APIs/Bancos)
3. Transformadores básicos para limpeza e normalização
4. Testes unitários e de integração
5. CI/CD pipeline (GitHub Actions)
6. Documentação de dados e schemas

## Recursos para Consulta
- README.md: Descrição geral do projeto
- docs/: Guias de desenvolvimento (quando criado)
- CONTRIBUTING.md: Guia de contribuição (quando criado)

---

**Nota para Agentes de IA**: Este documento descreve a estrutura esperada para este projeto ETL. Como está em fase inicial, use estas convenções como guia ao gerar novo código. Sempre priorize clareza, testabilidade e documentação.
