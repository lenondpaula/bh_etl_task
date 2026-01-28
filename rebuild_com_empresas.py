#!/usr/bin/env python3
"""
Reconstrói dataset criando geometrias a partir de pontos de empresas
"""
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, MultiPoint
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 15 bairros prioritários
BAIRROS_ALVO = [
    "ANCHIETA", "SAVASSI", "SION", "SANTO ANTONIO", "LOURDES",
    "FUNCIONARIOS", "SERRA", "BELVEDERE", "SANTO AGOSTINHO", "BARRO PRETO",
    "MANGABEIRAS", "CENTRO", "CRUZEIRO", "SAO BENTO", "CARMO"
]

def normalize_text(text):
    if pd.isna(text):
        return ""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', str(text))
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).upper().strip()

print("💼 Carregando cadastro de empresas...")
df_empresas = pd.read_csv(
    'data/cadastro_empresas_centro_sul.csv',
    sep=';',
    encoding='latin1',
    on_bad_lines='skip'
)
df_empresas['NOME_BAIRRO_NORM'] = df_empresas['NOME_BAIRRO'].apply(normalize_text)

# Filtrar bairros-alvo
df_empresas = df_empresas[df_empresas['NOME_BAIRRO_NORM'].isin(BAIRROS_ALVO)].copy()
print(f"✅ Carregadas {len(df_empresas)} empresas nos bairros-alvo")

# Converter GEOMETRIA WKT para Shapely
print("\n🗺️  Convertendo pontos WKT para geometrias...")
df_empresas['point'] = df_empresas['GEOMETRIA'].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)
df_empresas = df_empresas[df_empresas['point'].notna()].copy()

# Criar GeoDataFrame em UTM
gdf_empresas = gpd.GeoDataFrame(df_empresas, geometry='point', crs='EPSG:31983')
print(f"✅ {len(gdf_empresas)} pontos válidos")

# Criar polígonos por bairro (convex hull)
print("\n📐 Criando polígonos por bairro (convex hull)...")
bairros_data = []

for bairro_norm in BAIRROS_ALVO:
    subset = gdf_empresas[gdf_empresas['NOME_BAIRRO_NORM'] == bairro_norm]
    
    if len(subset) < 3:
        print(f"   ⚠️  {bairro_norm}: apenas {len(subset)} pontos - pulado")
        continue
    
    # Convex hull
    multi_point = MultiPoint(subset.geometry.tolist())
    polygon = multi_point.convex_hull
    
    bairros_data.append({
        'Nome_Bairro_NORM': bairro_norm,
        'Nome_Bairro': subset['NOME_BAIRRO'].iloc[0],
        'Qtd_Empresas': len(subset),
        'geometry': polygon
    })
    print(f"   ✅ {bairro_norm}: {len(subset)} empresas → polígono")

gdf = gpd.GeoDataFrame(bairros_data, crs='EPSG:31983')
print(f"\n✅ {len(gdf)} bairros com geometrias criadas")

# Reprojetar para WGS84
print("🌍 Reprojetando UTM → WGS84...")
gdf = gdf.to_crs('EPSG:4326')

# Calcular centroids
gdf['centroid_lat'] = gdf.geometry.centroid.y
gdf['centroid_lon'] = gdf.geometry.centroid.x

print("\n📍 Validando localizações (range Centro-Sul: -19.98 a -19.92 lat, -43.98 a -43.90 lon):")
for _, row in gdf.iterrows():
    status = "✓ OK" if -19.98 <= row['centroid_lat'] <= -19.92 and -43.98 <= row['centroid_lon'] <= -43.90 else "✗ FORA"
    print(f"   {row['Nome_Bairro_NORM']:20s} ({row['centroid_lat']:.4f}, {row['centroid_lon']:.4f}) {status}")

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
    'SANTA LUCIA': 'SAO BENTO STA LUCIA',
    'MANGABEIRAS': 'SERRA DAS MANGABEIRAS',
    'SERRA': 'SERRA DAS MANGABEIRAS'
}

iqvu_dict = {}
for _, row in df_iqvu.iterrows():
    nome_norm = row['Nome_Bairro_NORM']
    try:
        iqvu_val = float(row['IQVU']) * 1000
        iqvu_dict[nome_norm] = iqvu_val
    except:
        pass

print(f"✅ IQVU carregado para {len(iqvu_dict)} registros")

# Mapear Renda_Media
def get_renda(nome_norm):
    if nome_norm in iqvu_dict:
        return iqvu_dict[nome_norm]
    if nome_norm in iqvu_aliases:
        alias = iqvu_aliases[nome_norm]
        if alias in iqvu_dict:
            return iqvu_dict[alias]
    return np.nan

gdf['Renda_Media'] = gdf['Nome_Bairro_NORM'].apply(get_renda)

# Verificar dados ausentes
print("\n⚠️  Verificando dados ausentes:")
missing_renda = gdf[gdf['Renda_Media'].isna()]
if len(missing_renda) > 0:
    print(f"   {len(missing_renda)} bairros sem Renda: {missing_renda['Nome_Bairro_NORM'].tolist()}")
    print("   Preenchendo com mediana...")
    mediana = gdf['Renda_Media'].median()
    gdf['Renda_Media'] = gdf['Renda_Media'].fillna(mediana)

# Calcular saturação
print("\n📊 Calculando saturação comercial...")
gdf['area_km2'] = gdf.geometry.to_crs('EPSG:31983').area / 1e6
gdf['Saturacao_Comercial'] = gdf['Qtd_Empresas'] / gdf['area_km2']

# Normalizar scores
scaler = MinMaxScaler()
gdf['Score_Renda'] = scaler.fit_transform(gdf[['Renda_Media']])
gdf['Score_Saturacao'] = scaler.fit_transform(gdf[['Saturacao_Comercial']])
gdf['Score_Mobilidade'] = 0.7

# Calcular Apetite
gdf['Apetite_Investidor'] = (
    0.4 * gdf['Score_Mobilidade'] +
    0.3 * gdf['Score_Renda'] +
    0.3 * (1 - gdf['Score_Saturacao'])
).clip(0, 1)

# Forçar p90 >= 0.8
p90 = gdf['Apetite_Investidor'].quantile(0.9)
if p90 < 0.8:
    gdf['Apetite_Investidor'] = gdf['Apetite_Investidor'] * (0.8 / p90)
    gdf['Apetite_Investidor'] = gdf['Apetite_Investidor'].clip(0, 1)

# Classificação
def classify(apetite, saturacao):
    if apetite >= 0.75:
        return "OPORTUNIDADE DE OURO"
    elif saturacao >= 0.7:
        return "SATURADO"
    else:
        return "REGULAR"

gdf['Classificacao'] = gdf.apply(
    lambda row: classify(row['Apetite_Investidor'], row['Score_Saturacao']),
    axis=1
)

# Selecionar colunas finais
cols_final = [
    'Nome_Bairro', 'Renda_Media', 'Qtd_Empresas', 'Saturacao_Comercial',
    'Score_Renda', 'Score_Mobilidade', 'Score_Saturacao', 'Apetite_Investidor',
    'Classificacao', 'geometry', 'centroid_lat', 'centroid_lon', 'area_km2'
]
gdf_final = gdf[cols_final].copy()

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
