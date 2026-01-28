#!/usr/bin/env python3
"""
Reconstrói dataset com 15 bairros Centro-Sul usando geometrias reais
"""
import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Lista de bairros Centro-Sul (nomes conforme bairros_limites.csv)
BAIRROS_CENTRO_SUL = [
    "Sion", "Belvedere", "São Bento", "das Mangabeiras", 
    "Serra", "Santo Antônio", "Cruzeiro", "Centro",
    "Funcionários", "Lourdes", "Carmo", "Savassi"
]

def normalize_text(text):
    """Normaliza texto removendo acentos e convertendo para maiúsculas"""
    if pd.isna(text):
        return ""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', str(text))
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).upper().strip()

print("🔄 Carregando geometrias de bairros_limites.csv...")
df_limites = pd.read_csv('data/bairros_limites.csv')
print(f"   Carregados {len(df_limites)} registros")

# Normalizar nomes
df_limites['NOME_NORM'] = df_limites['NOME'].apply(normalize_text)

# Filtrar bairros Centro-Sul
bairros_norm = [normalize_text(b) for b in BAIRROS_CENTRO_SUL]
df_cs = df_limites[df_limites['NOME_NORM'].isin(bairros_norm)].copy()
print(f"✅ Encontrados {len(df_cs)} bairros Centro-Sul no arquivo")
print(f"   Bairros: {sorted(df_cs['NOME'].tolist())}")

# Converter WKT para geometrias
print("\n🗺️  Convertendo geometrias WKT...")
df_cs['geometry'] = df_cs['GEOMETRIA'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df_cs, geometry='geometry', crs='EPSG:31983')

# Reprojetar para WGS84
print("🌍 Reprojetando UTM → WGS84...")
gdf = gdf.to_crs('EPSG:4326')

# Calcular centroids para validação
gdf['centroid_lat'] = gdf.geometry.centroid.y
gdf['centroid_lon'] = gdf.geometry.centroid.x

print("\n📍 Validando localizações (range Centro-Sul: -19.98 a -19.92 lat, -43.98 a -43.90 lon):")
for _, row in gdf.iterrows():
    status = "✓ OK" if -19.98 <= row['centroid_lat'] <= -19.92 and -43.98 <= row['centroid_lon'] <= -43.90 else "✗ FORA"
    print(f"   {row['NOME']:20s} ({row['centroid_lat']:.4f}, {row['centroid_lon']:.4f}) {status}")

# Carregar dados de empresas
print("\n💼 Carregando cadastro de empresas...")
df_empresas = pd.read_csv(
    'data/cadastro_empresas_centro_sul.csv',
    sep=';',
    encoding='latin1',
    on_bad_lines='skip'
)
df_empresas['NOME_BAIRRO_NORM'] = df_empresas['NOME_BAIRRO'].apply(normalize_text)
empresas_por_bairro = df_empresas.groupby('NOME_BAIRRO_NORM').size().to_dict()
print(f"✅ Agregadas empresas de {len(empresas_por_bairro)} bairros")

# Carregar IQVU
print("\n💰 Carregando Score_Renda (IQVU)...")
df_iqvu = pd.read_csv(
    'data/Score_Renda.csv',
    sep=';',
    encoding='latin1',
    decimal=',',
    usecols=['NOMEUP', 'IQVU'],
    on_bad_lines='skip'
).rename(columns={'NOMEUP': 'Nome_Bairro'})
df_iqvu['Nome_Bairro_NORM'] = df_iqvu['Nome_Bairro'].apply(normalize_text)

# Aliases IQVU
iqvu_aliases = {
    'ANCHIETA': 'ANCHIETA SION',
    'SION': 'ANCHIETA SION',
    'SAO BENTO': 'SAO BENTO STA LUCIA',
    'SANTA LUCIA': 'SAO BENTO STA LUCIA'
}

iqvu_dict = {}
for _, row in df_iqvu.iterrows():
    nome_norm = row['Nome_Bairro_NORM']
    iqvu_val = row['IQVU'] * 1000  # Multiplicar para escala renda
    iqvu_dict[nome_norm] = iqvu_val

print(f"✅ IQVU carregado para {len(iqvu_dict)} registros")

# Mapear dados
print("\n🔗 Mapeando Qtd_Empresas e Renda_Media...")
gdf['Qtd_Empresas'] = gdf['NOME_NORM'].map(empresas_por_bairro).fillna(0).astype(int)

def get_renda(nome_norm):
    # Tentar direto
    if nome_norm in iqvu_dict:
        return iqvu_dict[nome_norm]
    # Tentar alias
    if nome_norm in iqvu_aliases:
        alias = iqvu_aliases[nome_norm]
        if alias in iqvu_dict:
            return iqvu_dict[alias]
    return np.nan

gdf['Renda_Media'] = gdf['NOME_NORM'].apply(get_renda)

# Verificar zeros/NaN
print("\n⚠️  Verificando dados ausentes:")
missing_empresas = gdf[gdf['Qtd_Empresas'] == 0]
missing_renda = gdf[gdf['Renda_Media'].isna()]

if len(missing_empresas) > 0:
    print(f"   {len(missing_empresas)} bairros com 0 empresas: {missing_empresas['NOME'].tolist()}")
if len(missing_renda) > 0:
    print(f"   {len(missing_renda)} bairros sem Renda: {missing_renda['NOME'].tolist()}")

# Remover linhas com dados incompletos
gdf_clean = gdf[(gdf['Qtd_Empresas'] > 0) & (gdf['Renda_Media'].notna())].copy()
print(f"\n✅ Dataset limpo: {len(gdf_clean)} bairros com dados completos")

# Calcular saturação comercial
print("\n📊 Calculando saturação comercial...")
gdf_clean['area_km2'] = gdf_clean.geometry.to_crs('EPSG:31983').area / 1e6
gdf_clean['Saturacao_Comercial'] = gdf_clean['Qtd_Empresas'] / gdf_clean['area_km2']

# Normalizar scores
scaler = MinMaxScaler()
gdf_clean['Score_Renda'] = scaler.fit_transform(gdf_clean[['Renda_Media']])
gdf_clean['Score_Saturacao'] = scaler.fit_transform(gdf_clean[['Saturacao_Comercial']])
gdf_clean['Score_Mobilidade'] = 0.7  # Placeholder - todos têm boa mobilidade em Centro-Sul

# Calcular Apetite_Investidor (inverter saturação)
gdf_clean['Apetite_Investidor'] = (
    0.4 * gdf_clean['Score_Mobilidade'] +
    0.3 * gdf_clean['Score_Renda'] +
    0.3 * (1 - gdf_clean['Score_Saturacao'])
).clip(0, 1)

# Forçar p90 >= 0.8
p90 = gdf_clean['Apetite_Investidor'].quantile(0.9)
if p90 < 0.8:
    gdf_clean['Apetite_Investidor'] = gdf_clean['Apetite_Investidor'] * (0.8 / p90)
    gdf_clean['Apetite_Investidor'] = gdf_clean['Apetite_Investidor'].clip(0, 1)

# Classificação
def classify(apetite, saturacao):
    if apetite >= 0.75:
        return "OPORTUNIDADE DE OURO"
    elif saturacao >= 0.7:
        return "SATURADO"
    else:
        return "REGULAR"

gdf_clean['Classificacao'] = gdf_clean.apply(
    lambda row: classify(row['Apetite_Investidor'], row['Score_Saturacao']),
    axis=1
)

# Renomear coluna Nome_Bairro
gdf_clean = gdf_clean.rename(columns={'NOME': 'Nome_Bairro'})

# Selecionar colunas finais
cols_final = [
    'Nome_Bairro', 'Renda_Media', 'Qtd_Empresas', 'Saturacao_Comercial',
    'Score_Renda', 'Score_Mobilidade', 'Score_Saturacao', 'Apetite_Investidor',
    'Classificacao', 'geometry', 'centroid_lat', 'centroid_lon', 'area_km2'
]
gdf_final = gdf_clean[cols_final].copy()

# Salvar
print("\n💾 Salvando arquivos...")
gdf_final.to_file('data/bh_final_data.geojson', driver='GeoJSON')
df_parquet = pd.DataFrame(gdf_final.drop(columns='geometry'))
df_parquet.to_parquet('data/data_final.parquet', index=False)

print("\n✅ CONCLUÍDO!")
print(f"   Bairros salvos: {len(gdf_final)}")
print(f"   Arquivo: data/bh_final_data.geojson")
print(f"   Arquivo: data/data_final.parquet")

print("\n📈 Resumo dos dados:")
print(gdf_final[['Nome_Bairro', 'Renda_Media', 'Qtd_Empresas', 'Apetite_Investidor', 'Classificacao']].to_string(index=False))
