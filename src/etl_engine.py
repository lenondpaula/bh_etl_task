def run_etl(csv_path: str = "data/dados_economicos.csv", geojson_path: str = "data/bairros_data.geojson", out_path: str = "data/bh_final_data.geojson") -> None:
    """Executa o pipeline ETL: leitura, normalização, merge, scoring e saída.

    Args:
        csv_path: Caminho para o CSV de dados econômicos.
        geojson_path: Caminho para o GeoJSON de bairros (geometria).
        out_path: Caminho de saída para o GeoJSON final com atributos calculados.

    Returns:
        None: Salva GeoJSON e Parquet em disco na pasta `data/`.
    """
    log = logging.getLogger("etl_engine")
    log.info("Lendo CSV com Polars: %s", csv_path)
    pl_df = pl.read_csv(csv_path)

    log.info("Lendo GeoJSON com GeoPandas: %s", geojson_path)
    gdf = gpd.read_file(geojson_path)

    # Normalizar nomes em GeoDataFrame
    log.info("Normalizando nomes em GeoDataFrame")
    gdf["Nome_Bairro_NORM"] = gdf["Nome_Bairro"].apply(normalize_text)
    # Remover colunas duplicadas antes do merge para evitar sufixos (_x/_y)
    dup_cols = ["Renda_Media", "Qtd_Empresas", "Qtd_Pontos_Onibus"]
    gdf = gdf.drop(columns=[c for c in dup_cols if c in gdf.columns])

    # Converter Polars -> Pandas e normalizar nomes
    log.info("Convertendo Polars -> Pandas e normalizando nomes")
    df_econ = pl_df.to_pandas()
    df_econ["Nome_Bairro_NORM"] = df_econ["Nome_Bairro"].apply(normalize_text)

    # Selecionar colunas de interesse no CSV antes do merge
    cols = [c for c in ["id_bairro", "Nome_Bairro_NORM", "Renda_Media", "Qtd_Empresas", "Qtd_Pontos_Onibus"] if c in df_econ.columns]
    df_econ_sel = df_econ[cols]

    log.info("Realizando merge entre geometria e dados econômicos")
    merged = gdf.merge(df_econ_sel, on="Nome_Bairro_NORM", how="left")

    # Garantir tipos numéricos
    for col in ["Renda_Media", "Qtd_Empresas", "Qtd_Pontos_Onibus"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
        else:
            merged[col] = np.nan

    # Preencher NaN com 0.0 para evitar buracos no mapa
    merged["Renda_Media"] = merged["Renda_Media"].fillna(0.0)
    merged["Qtd_Empresas"] = merged["Qtd_Empresas"].fillna(0.0)
    merged["Qtd_Pontos_Onibus"] = merged["Qtd_Pontos_Onibus"].fillna(0.0)

    # Engenharia de atributos: normalizações
    log.info("Calculando scores normalizados")
    merged["Score_Mobilidade"] = minmax_scale_series(merged["Qtd_Pontos_Onibus"]).fillna(0.0)
    merged["Score_Renda"] = minmax_scale_series(merged["Renda_Media"]).fillna(0.0)
    merged["Saturacao_Comercial"] = minmax_scale_series(merged["Qtd_Empresas"]).fillna(0.0)

    # Apetite_Investidor
    merged["Apetite_Investidor"] = (merged["Score_Renda"] * 0.4) + (merged["Score_Mobilidade"] * 0.3) - (merged["Saturacao_Comercial"] * 0.3)
    merged["Apetite_Investidor"] = merged["Apetite_Investidor"].clip(lower=0)

    # Classificação básica para consistência
    merged["classificacao"] = "REGULAR"

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Salvando resultado final em %s", out_path)
    merged.to_file(out_path, driver="GeoJSON")

    # Salvar também versão parquet (sem geometria) para integração/consumo tabular
    try:
        parquet_path = Path(out_path).with_name("data_final.parquet")
        cols_out = [
            "id_bairro",
            "Nome_Bairro",
            "Renda_Media",
            "Qtd_Empresas",
            "Qtd_Pontos_Onibus",
            "Apetite_Investidor",
            "classificacao",
        ]
        df_out = merged[[c for c in cols_out if c in merged.columns]].copy()
        pl_df = pl.DataFrame(df_out)
        pl_df.write_parquet(str(parquet_path))
        log.info("Parquet salvo em %s", parquet_path)
    except Exception as e:
        log.warning("Falha ao salvar parquet: %s", e)

    log.info("ETL concluído. Registros processados: %d", len(merged))
    # liberar memória de objetos grandes
    del merged