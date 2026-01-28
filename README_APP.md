# BH Strategic Navigator — App (MVP)

Olá, eu sou o **Lenon Lima de Paula** 👋

Jornalista | Especialista em Ciência de Dados e IA. Este MVP foi criado para fornecer inteligência de mercado para seleção de locais comerciais na região Centro‑Sul de Belo Horizonte, combinando dados econômicos, geoespaciais e visualização 3D.

---

## Visão Técnica

- **Objetivo:** identificar oportunidades de investimento por bairro usando dados de densidade comercial, mobilidade e renda (IQVU).
- **Abordagem:** normalização de nomes, junções espaciais (pontos → bairros), preenchimento de dados por IQVU (Score_Renda.csv) com mapeamentos manuais e heurísticos, e processamento vetorizado para escalabilidade.
- **Visualização:** mapa 3D (PyDeck) com extrusão pela densidade de empresas e cor por `Apetite_Investidor` (score combinado de mobilidade, renda e saturação).

---

## Como usar

1. Use o painel lateral para filtrar bairros do Centro‑Sul e ver indicadores em tempo real.
2. No mapa 3D, altura representa a densidade comercial (extrusão); cor representa o Apetite do Investidor — girar com botão direito do mouse.
3. Em "Cluster Analysis" explore agrupamentos por perfil socioeconômico.

---

## Arquitetura & Dados

- Fonte de empresas: `data/cadastro_empresas_centro_sul.csv` (pontos georreferenciados).
- Limites de bairros: `data/bairros_limites.csv` (polígonos em EPSG:31983).
- IQVU (renda): `data/Score_Renda.csv` (utilizado como fallback e imputado por bairro).
- Saída: `data/bh_final_data.parquet` (GeoDataFrame reduzido ao Centro‑Sul para este MVP).

---

## Contato

- Lenon Lima de Paula — *Project Lead*
- (incluir e‑mail/links conforme desejado)

---

> Nota: Este README é exibido no app via o painel lateral. Para alterações de mapeamento manual (e.g., Savassi / Lourdes), edite `src/etl_engine.py` e reexecute o ETL.