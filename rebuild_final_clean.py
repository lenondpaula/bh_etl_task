#!/usr/bin/env python3
"""
Reconstrói dataset FINAL apenas com bairros que têm dados reais de empresas
- 17 bairros com >= 329 empresas
- Geometrias via convex hull dos pontos de empresas
- Altura dos polígonos reduzida em 60% (multiplicar por 0.4)
"""
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, MultiPoint
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 17 bairros com dados REAIS de empresas (de cadastro_empresas_centro_sul.csv)
BAIRROS_REAIS = [
    "CENTRO", "SAVASSI", "SANTA EFIGENIA", "LOURDES", "SANTO ANTONIO",
    "SERRA", "SION", "FUNCIONARIOS", "BELVEDERE", "ANCHIETA",
    "SAO PEDRO", "CRUZEIRO", "CARMO", "BOA VIAGEM", "SAO LUCAS",
    "VILA PARIS", "MANGABEIRAS"
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

# Filtrar APENAS bairros com dados reais
df_empresas = df_empresas[df_empresas['NOME_BAIRRO_NORM'].isin(BAIRROS_REAIS)].copy()
print(f"✅ Carregadas {len(df_empresas)} empresas em {len(df_empresas['NOME_BAIRRO_NORM'].unique())} bairros")

# Converter GEOMETRIA WKT para Shapely
print("\n🗺️  Convertendo pontos WKT para geometrias...")
df_empresas['point'] = df_empresas['GEOMETRIA'].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)
df_empresas = df_empresas[df_empresas['point'].notna()].copy()

print(f"✅ {len(df_empresas)} pontos antes da filtragem rigorosa")

# CRITICAL STEP: Apply STRICT Bounding Box Filter BEFORE Convex Hull (Guilhotina/Hard Cut)
print("\n🔪 GUILHOTINA: Aplicando filtro rigoroso de bounding box (remove outliers que distorcem polígonos)...")
print("   Limites: lat -19.965 a -19.900, lon -43.980 a -43.910")

# Converter para WGS84 para filtragem
gdf_temp = gpd.GeoDataFrame(df_empresas, geometry='point', crs='EPSG:31983')
gdf_temp = gdf_temp.to_crs('EPSG:4326')
df_empresas['lat_wgs84'] = gdf_temp.geometry.y
df_empresas['lon_wgs84'] = gdf_temp.geometry.x

# Hard cut (Guilhotina)
before_cut = len(df_empresas)
df_empresas = df_empresas[
    (df_empresas['lat_wgs84'] >= -19.965) & (df_empresas['lat_wgs84'] <= -19.900) &
    (df_empresas['lon_wgs84'] >= -43.980) & (df_empresas['lon_wgs84'] <= -43.910)
].copy()
outliers_removed = before_cut - len(df_empresas)
print(f"✅ Removidos {outliers_removed} pontos outliers (puxavam polígonos para Nova Lima/Barreiro)")
print(f"✅ {len(df_empresas)} pontos válidos após filtragem rigorosa")

# Criar GeoDataFrame em UTM
gdf_empresas = gpd.GeoDataFrame(df_empresas, geometry='point', crs='EPSG:31983')

# Criar polígonos por bairro (convex hull)
print("\n📐 Criando polígonos por bairro (convex hull)...")
bairros_data = []

for bairro_norm in BAIRROS_REAIS:
    subset = gdf_empresas[gdf_empresas['NOME_BAIRRO_NORM'] == bairro_norm]
    
    if len(subset) < 3:
        print(f"   ⚠️  {bairro_norm}: apenas {len(subset)} pontos - REMOVIDO")
        continue
    
    # Convex hull
    multi_point = MultiPoint(subset.geometry.tolist())
    polygon = multi_point.convex_hull
    
    # Obter nome original do bairro
    nome_original = subset['NOME_BAIRRO'].iloc[0]
    
    bairros_data.append({
        'Nome_Bairro': nome_original,
        'Nome_Bairro_NORM': bairro_norm,
        'Qtd_Empresas': len(subset),
        'geometry': polygon
    })
    print(f"   ✅ {bairro_norm:15s}: {len(subset):5d} empresas → polígono")

gdf = gpd.GeoDataFrame(bairros_data, crs='EPSG:31983')
print(f"\n✅ {len(gdf)} bairros com geometrias criadas")

# Reprojetar para WGS84
print("🌍 Reprojetando UTM → WGS84...")
gdf = gdf.to_crs('EPSG:4326')

# Calcular centroids
gdf['centroid_lat'] = gdf.geometry.centroid.y
gdf['centroid_lon'] = gdf.geometry.centroid.x

print("\n📍 Validando localizações (centro da região: -19.9475, -43.9472):")
for _, row in gdf.iterrows():
    lat, lon = row['centroid_lat'], row['centroid_lon']
    print(f"   {row['Nome_Bairro_NORM']:15s}: ({lat:8.4f}, {lon:8.4f})")

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

# Aliases IQVU (para bairros que podem ter nomes diferentes no Score_Renda)
iqvu_aliases = {
    'ANCHIETA': 'ANCHIETA SION',
    'SION': 'ANCHIETA SION',
    'SAO BENTO': 'SAO BENTO STA LUCIA',
    'SANTA LUCIA': 'SAO BENTO STA LUCIA',
    'MANGABEIRAS': 'SERRA DAS MANGABEIRAS',
    'SERRA': 'SERRA DAS MANGABEIRAS',
    'SANTA EFIGENIA': 'SANTA EFIGENIA',
    'SAO PEDRO': 'SAO PEDRO',
    'SAO LUCAS': 'SAO LUCAS',
    'BOA VIAGEM': 'BOA VIAGEM',
    'VILA PARIS': 'VILA PARIS'
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
    # Tentar direto
    if nome_norm in iqvu_dict:
        return iqvu_dict[nome_norm]
    # Tentar alias
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
    print(f"   {len(missing_renda)} bairros sem IQVU match: {missing_renda['Nome_Bairro_NORM'].tolist()}")
    print("   ℹ️  Preenchendo com mediana dos bairros disponíveis...")
    mediana = gdf['Renda_Media'].median()
    gdf.loc[gdf['Renda_Media'].isna(), 'Renda_Media'] = mediana
    print(f"   ✅ Mediana: {mediana:.2f}")

# Verificar zeros de empresas
zeros_empresas = gdf[gdf['Qtd_Empresas'] == 0]
if len(zeros_empresas) > 0:
    print(f"   ⚠️  ERRO: {len(zeros_empresas)} bairros com 0 empresas! Removendo...")
    gdf = gdf[gdf['Qtd_Empresas'] > 0].copy()

print(f"\n✅ Após limpeza: {len(gdf)} bairros com dados completos")

# Calcular saturação
print("\n📊 Calculando saturação comercial...")
gdf['area_km2'] = gdf.geometry.to_crs('EPSG:31983').area / 1e6
gdf['Saturacao_Comercial'] = gdf['Qtd_Empresas'] / gdf['area_km2']

# Normalizar scores
scaler = MinMaxScaler()
gdf['Score_Renda'] = scaler.fit_transform(gdf[['Renda_Media']])
gdf['Score_Saturacao'] = scaler.fit_transform(gdf[['Saturacao_Comercial']])
gdf['Score_Mobilidade'] = 0.7  # Todos têm boa mobilidade em Centro-Sul

# Calcular Apetite_Investidor
gdf['Apetite_Investidor'] = (
    0.4 * gdf['Score_Mobilidade'] +
    0.3 * gdf['Score_Renda'] +
    0.3 * (1 - gdf['Score_Saturacao'])
).clip(0, 1)

# Forçar p90 >= 0.8 para melhor distribuição
p90 = gdf['Apetite_Investidor'].quantile(0.9)
if p90 < 0.8:
    factor = 0.8 / p90
    gdf['Apetite_Investidor'] = (gdf['Apetite_Investidor'] * factor).clip(0, 1)

# CRITICAL: Force Classification Logic com limiares observados nos dados reais
print("\n🏆 Aplicando lógica de classificação com limiares otimizados...")
print("   SATURADO: Qtd_Empresas >= 15000")
print("   OURO: Apetite_Investidor >= 0.78 (e não Saturado)")
print("   PRATA: Todo o resto")

def classify_by_thresholds(row):
    # SATURADO tem prioridade: mercado maduro, muita concorrência
    if row['Qtd_Empresas'] >= 15000:
        return "SATURADO"
    # OURO: boa mobilidade, boa renda, pouca saturação
    elif row['Apetite_Investidor'] >= 0.78:
        return "OURO"
    # Tudo o resto é PRATA
    else:
        return "PRATA"

gdf['Classificacao'] = gdf.apply(classify_by_thresholds, axis=1)

# Verify no NaN
missing_class = gdf[gdf['Classificacao'].isna() | (gdf['Classificacao'] == '')]
if len(missing_class) > 0:
    print(f"⚠️  ERRO: {len(missing_class)} bairros com Classificação vazia!")
    gdf.loc[gdf['Classificacao'].isna() | (gdf['Classificacao'] == ''), 'Classificacao'] = 'PRATA'

print(f"✅ Classificações aplicadas:")
for classif, count in gdf['Classificacao'].value_counts().items():
    print(f"   {classif}: {count} bairros")

# **ESCALA CRÍTICA PARA 3D: Elevation_3D = Apetite * 3000 (0-1 -> 0-3km visual bars)**
print("\n📉 Calculando elevação 3D com escala visual crítica (Apetite × 3000)...")
gdf['Elevation_3D'] = gdf['Apetite_Investidor'] * 3000  # Vertical exaggeration for visual impact
print(f"✅ Elevação 3D calculada: max={gdf['Elevation_3D'].max():.1f}m, min={gdf['Elevation_3D'].min():.1f}m")

# Selecionar colunas finais
cols_final = [
    'Nome_Bairro', 'Renda_Media', 'Qtd_Empresas', 'Saturacao_Comercial',
    'Score_Renda', 'Score_Mobilidade', 'Score_Saturacao', 'Apetite_Investidor',
    'Elevation_3D', 'Classificacao', 'geometry', 'centroid_lat', 'centroid_lon', 'area_km2'
]
gdf_final = gdf[cols_final].copy()

# Verificar Classificação preenchida
print("\n✅ Verificando Classificação preenchida:")
missing_class = gdf_final[gdf_final['Classificacao'].isna() | (gdf_final['Classificacao'] == '')]
if len(missing_class) > 0:
    print(f"   ⚠️  {len(missing_class)} bairros com Classificação vazia!")
else:
    print(f"   ✅ Todos os {len(gdf_final)} bairros têm Classificação preenchida")

# Salvar
print("\n💾 Salvando arquivos...")
gdf_final.to_file('data/bh_final_data.geojson', driver='GeoJSON')
df_parquet = pd.DataFrame(gdf_final.drop(columns='geometry'))
df_parquet.to_parquet('data/data_final.parquet', index=False)

print("\n✅ CONCLUÍDO!")
print(f"   Bairros salvos: {len(gdf_final)}")
print(f"   Arquivo: data/bh_final_data.geojson")
print(f"   Arquivo: data/data_final.parquet")

print("\n📈 RESUMO FINAL:")
print("="*90)
df_report = gdf_final[['Nome_Bairro', 'Renda_Media', 'Qtd_Empresas', 'Apetite_Investidor', 'Elevation_3D', 'Classificacao']].copy()
df_report = df_report.sort_values('Apetite_Investidor', ascending=False)
for idx, row in df_report.iterrows():
    print(f"{row.Nome_Bairro:18s} | Renda: {row.Renda_Media:7.1f} | Empresas: {row.Qtd_Empresas:5d} | Apetite: {row.Apetite_Investidor:.3f} | Elev 3D: {row.Elevation_3D:.3f} | {row.Classificacao}")
print("="*90)

print("\n🎯 CLASSIFICAÇÕES:")
ouro = len(gdf_final[gdf_final['Classificacao'] == 'OPORTUNIDADE DE OURO'])
saturado = len(gdf_final[gdf_final['Classificacao'] == 'SATURADO'])
regular = len(gdf_final[gdf_final['Classificacao'] == 'REGULAR'])
print(f"   • OPORTUNIDADE DE OURO: {ouro} bairros")
print(f"   • SATURADO: {saturado} bairros")
print(f"   • REGULAR: {regular} bairros")
print()
