from pathlib import Path
import pandas as pd
import numpy as np
# Local implementations to avoid importing src as module (keeps script self-contained)
import unicodedata

def normalize_text(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s).upper()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = ''.join(ch if (ch.isalnum() or ch.isspace()) else ' ' for ch in s)
    s = ' '.join(s.split())
    return s

import numpy as np

def minmax_scale_series(s: pd.Series) -> pd.Series:
    arr = pd.to_numeric(s, errors='coerce')
    minv = arr.min()
    maxv = arr.max()
    if pd.isna(minv) or pd.isna(maxv) or maxv == minv:
        return pd.Series(0.0, index=s.index)
    return ((arr - minv) / (maxv - minv)).astype(float)

PARQUET = Path("data/bh_final_data.parquet")
if not PARQUET.exists():
    raise SystemExit("Parquet not found: data/bh_final_data.parquet")

df = pd.read_parquet(PARQUET)

# MANUAL FORCED MAPPINGS (business rules)
# Map tokens to target bairros (priority order) regardless of current substring matches
manual_map_priorities = {
    'SAVASSI': ['SION', 'FUNCIONARIOS'],
    'LOURDES': ['SANTO AGOSTINHO', 'BELVEDERE'],
    'CENTRO': ['CENTRO-SUL']
}

# helper: available normalized bairro names in the current dataframe
available_norms = set(df['Nome_Bairro_NORM'].dropna().astype(str).unique())
# map token -> chosen target (original casing preserved when possible)
forced_targets = {}
for token, priority_list in manual_map_priorities.items():
    chosen = None
    for candidate in priority_list:
        c_norm = normalize_text(candidate)
        if c_norm in available_norms:
            # pick the first existing candidate
            chosen = candidate
            break
    if chosen:
        forced_targets[token] = chosen
    else:
        # not found — we'll still record the normalized candidate name for later (no-op if missing)
        forced_targets[token] = priority_list[0]

# Apply forced merges: if any row contains the token in its Nome_Bairro_NORM, merge its numeric stats into the target
numeric_cols = [c for c in ['Qtd_Empresas', 'Qtd_Pontos_Onibus', 'Score_Mobilidade', 'Saturacao_Comercial', 'Renda_Media'] if c in df.columns]
for token, target in forced_targets.items():
    token_norm = normalize_text(token)
    target_norm = normalize_text(target)
    # rows to move
    mask_token = df['Nome_Bairro_NORM'].astype(str).str.contains(token_norm, na=False)
    if not mask_token.any():
        # nothing to move for this token, but ensure target exists (if target not present, do nothing)
        continue
    rows_to_move = df[mask_token].copy()
    # find target row index (exact match by normalized name)
    target_idx = df.index[df['Nome_Bairro_NORM'].astype(str) == target_norm].tolist()
    if target_idx:
        tidx = target_idx[0]
        # aggregate numeric columns
        for col in numeric_cols:
            s_target = pd.to_numeric(df.at[tidx, col], errors='coerce')
            s_move = pd.to_numeric(rows_to_move[col].sum(), errors='coerce')
            df.at[tidx, col] = (0 if pd.isna(s_target) else s_target) + (0 if pd.isna(s_move) else s_move)
        # weighted Renda_Media (if both have Qtd_Empresas)
        if 'Renda_Media' in numeric_cols and 'Qtd_Empresas' in df.columns:
            q_target = float(df.at[tidx, 'Qtd_Empresas'] if not pd.isna(df.at[tidx, 'Qtd_Empresas']) else 0.0)
            q_move = float(rows_to_move['Qtd_Empresas'].sum() if 'Qtd_Empresas' in rows_to_move.columns else 0.0)
            r_target = float(df.at[tidx, 'Renda_Media'] if not pd.isna(df.at[tidx, 'Renda_Media']) else 0.0)
            r_move = float(rows_to_move['Renda_Media'].sum() if 'Renda_Media' in rows_to_move.columns else 0.0)
            if (q_target + q_move) > 0:
                df.at[tidx, 'Renda_Media'] = (r_target * q_target + r_move * q_move) / (q_target + q_move)
    else:
        # target not present — create a new aggregated row from rows_to_move and set its Nome_Bairro to target
        new_row = rows_to_move.iloc[[0]].copy()
        new_row.index = [df.index.max() + 1 if len(df.index) else 0]
        new_row.at[new_row.index[0], 'Nome_Bairro'] = target
        new_row.at[new_row.index[0], 'Nome_Bairro_NORM'] = target_norm
        # aggregate numeric cols across moved rows
        for col in numeric_cols:
            new_row.at[new_row.index[0], col] = rows_to_move[col].sum()
        df = pd.concat([df[~mask_token], new_row], ignore_index=False)
        continue
    # drop moved rows (excluding the target row if it was among them)
    df = df[~(mask_token & (df['Nome_Bairro_NORM'].astype(str) != target_norm))]

# expanded Centro-Sul tokens (include business-requested neighborhoods)
desired_centro_tokens = [
    "FUNCIONARIOS", "SAO PEDRO", "SANTO ANTONIO", "LOURDES", "SAVASSI", "ANCHIETA", "CARMO",
    "CRUZEIRO", "SAO LUCAS", "SERRA", "CENTRO", "BOA VIAGEM", "SANTA EFIGENIA", "SANTO AGOSTINHO", "BELVEDERE",
    "SION", "CENTRO-SUL"
]
desired_norms = [normalize_text(t) for t in desired_centro_tokens]

# start with any existing centro tokens (previous mask) plus desired tokens
centro_tokens = ["SAVASSI", "CENTRO", "LOURDES", "BELVEDERE", "SION", "ANCHIETA", "CRUZEIRO", "SANTO AGOSTINHO", "FUNCIONARIOS", "CENTRO-SUL"]
centro_norms = list(set([normalize_text(t) for t in centro_tokens] + desired_norms))

def is_centro_sul(n):
    if not n:
        return False
    s = str(n)
    return any(ct in s for ct in centro_norms)

mask = df['Nome_Bairro_NORM'].apply(lambda x: is_centro_sul(x))

# In case region column exists with CENTRO
region_col = next((c for c in df.columns if 'REG' in c.upper()), None)
if region_col:
    try:
        mask = mask | df[region_col].astype(str).str.contains('CENTRO', case=False, na=False)
    except Exception:
        pass

# collect candidate Centro-Sul rows
cs_df = df.loc[mask].copy()

# If any desired tokens are missing, try to add them from the bairros limits file (by name or nearest polygon)
# Try CSV first, then GeoJSON fallback
bairros_path_csv = Path('data/bairros_limites.csv')
bairros_path_geo = Path('data/bairros_data.geojson')
bairros_gdf = None
for candidate_path in (bairros_path_csv, bairros_path_geo):
    if candidate_path.exists():
        try:
            import shapely.wkt as wkt
            import geopandas as gpd
            if candidate_path.suffix.lower() == '.csv':
                bdf = pd.read_csv(str(candidate_path), sep=',', encoding='utf8', on_bad_lines='skip')
                geom_col = next((c for c in bdf.columns if 'GEOM' in c.upper() or 'GEOMETRIA' in c.upper() or 'WKT' in c.upper()), None)
                name_col_b = next((c for c in bdf.columns if 'NOME' in c.upper() or 'BAIRRO' in c.upper()), None)
                if geom_col:
                    bdf['geometry'] = bdf[geom_col].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)
                if name_col_b is None:
                    bdf['Nome_Bairro'] = bdf.iloc[:,0].astype(str)
                else:
                    bdf['Nome_Bairro'] = bdf[name_col_b].astype(str)
                bairros_gdf = gpd.GeoDataFrame(bdf, geometry='geometry', crs='EPSG:31983')
            else:
                # geojson
                bdf = gpd.read_file(str(candidate_path))
                name_col_b = next((c for c in bdf.columns if 'NOME' in c.upper() or 'BAIRRO' in c.upper()), None)
                if name_col_b is None:
                    bdf['Nome_Bairro'] = bdf.index.astype(str)
                else:
                    bdf['Nome_Bairro'] = bdf[name_col_b].astype(str)
                bairros_gdf = gpd.GeoDataFrame(bdf, geometry='geometry', crs=bdf.crs if hasattr(bdf, 'crs') else 'EPSG:31983')
            # normalize names and compute centroids
            bairros_gdf['Nome_Bairro_NORM'] = bairros_gdf['Nome_Bairro'].apply(normalize_text)
            try:
                bairros_gdf['centroid'] = bairros_gdf.geometry.centroid
            except Exception:
                bairros_gdf['centroid'] = bairros_gdf.geometry.representative_point()
            break
        except Exception:
            bairros_gdf = None

# Try to compute company counts per bairro (spatial join) if possible to populate Qtd_Empresas for added rows
emp_counts = {}
cad_path = Path('data/cadastro_empresas_centro_sul.csv')
if cad_path.exists() and bairros_gdf is not None:
    try:
        cad = pd.read_csv(str(cad_path), sep=';', encoding='utf8', on_bad_lines='skip', low_memory=True)
        # find column that contains 'POINT'
        pt_col = next((c for c in cad.columns if cad[c].astype(str).str.contains('POINT', na=False).any()), None)
        if pt_col:
            import shapely.wkt as wkt
            cad['geometry'] = cad[pt_col].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)
            cad_gdf = gpd.GeoDataFrame(cad, geometry='geometry', crs='EPSG:31983')
            # ensure same crs
            if cad_gdf.crs != bairros_gdf.crs:
                cad_gdf = cad_gdf.set_crs('EPSG:31983')
            # spatial join (points -> bairros)
            joined = gpd.sjoin(cad_gdf[['geometry']].copy(), bairros_gdf[['Nome_Bairro','Nome_Bairro_NORM','geometry']].copy(), how='left', predicate='within')
            counts = joined.groupby('Nome_Bairro_NORM').size().to_dict()
            emp_counts = {k: int(v) for k,v in counts.items()}
    except Exception:
        emp_counts = {}

# helper: set of found normalized names in cs_df
found_norms = set(cs_df['Nome_Bairro_NORM'].dropna().astype(str).unique())
# for each desired norm, if missing, try to add row from bairros_gdf by name; if name not present, map to nearest polygon
if bairros_gdf is not None:
    # prepare centroids
    try:
        bairros_gdf['centroid'] = bairros_gdf.geometry.centroid
    except Exception:
        bairros_gdf['centroid'] = bairros_gdf.geometry.representative_point()
    # compute list of candidate names to ensure
    for token in desired_centro_tokens:
        tnorm = normalize_text(token)
        if tnorm in found_norms:
            continue
        # try find exact match in bairros_gdf
        match = bairros_gdf[bairros_gdf['Nome_Bairro_NORM'].str.contains(tnorm, na=False)]
        chosen_row = None
        if match.shape[0] >= 1:
            chosen_row = match.iloc[0]
        else:
            from difflib import get_close_matches
            candidates = list(bairros_gdf['Nome_Bairro_NORM'].dropna().astype(str).unique())
            close = get_close_matches(tnorm, candidates, n=1, cutoff=0.45)
            if close:
                chosen_row = bairros_gdf[bairros_gdf['Nome_Bairro_NORM'] == close[0]].iloc[0]
            else:
                # choose reference centroid: median of current cs_df (if present) else city center
                ref = None
                try:
                    if not cs_df.empty and 'geometry_wkt' in cs_df.columns and cs_df['geometry_wkt'].notna().any():
                        import shapely.wkt as wkt
                        centroids = [wkt.loads(g).centroid for g in cs_df['geometry_wkt'].dropna()]
                        from shapely.ops import unary_union
                        ref = unary_union(centroids).centroid
                except Exception:
                    ref = None
                if ref is None:
                    ref = bairros_gdf['centroid'].unary_union.centroid
                bairros_gdf['dist_to_ref'] = bairros_gdf['centroid'].distance(ref)
                chosen_row = bairros_gdf.loc[bairros_gdf['dist_to_ref'].idxmin()]
        # map chosen_row into cs_df
        if chosen_row is not None:
            tnorm_target = chosen_row['Nome_Bairro_NORM']
            # avoid duplicating an already present target
            if tnorm_target in set(cs_df['Nome_Bairro_NORM'].dropna().astype(str).unique()):
                continue
            new = {c: 0 for c in df.columns}
            new['Nome_Bairro'] = chosen_row['Nome_Bairro']
            new['Nome_Bairro_NORM'] = tnorm_target
            if 'geometry_wkt' in df.columns:
                new['geometry_wkt'] = chosen_row.geometry.wkt
            elif 'geometry' in df.columns:
                new['geometry'] = chosen_row.geometry
            # populate Qtd_Empresas if available from spatial counts
            if 'Qtd_Empresas' in df.columns and tnorm_target in emp_counts:
                new['Qtd_Empresas'] = emp_counts.get(tnorm_target, 0)
            # flag mapping origin
            new['Mapped_From'] = token
            cs_df = pd.concat([cs_df, pd.DataFrame([new])], ignore_index=True)
            found_norms.add(tnorm_target)

# after ensuring desired tokens, if still less than 15 neighborhoods, add nearest neighbors from bairros_gdf until count >= 15
if bairros_gdf is not None and len(cs_df) < 15:
    # compute current centroids to find nearby bairros
    try:
        import shapely.wkt as wkt
        cur_centroids = [wkt.loads(g).centroid for g in cs_df['geometry'].dropna().head(100)] if 'geometry' in cs_df.columns else []
    except Exception:
        cur_centroids = []
    if cur_centroids:
        ref = pd.Series(cur_centroids).unary_union.centroid
    else:
        ref = bairros_gdf['centroid'].unary_union.centroid
    bairros_gdf['dist_to_ref'] = bairros_gdf['centroid'].distance(ref)
    # iterate sorted by distance and add until we have 15
    for idx, row in bairros_gdf.sort_values('dist_to_ref').iterrows():
        tnorm = row['Nome_Bairro_NORM']
        if tnorm in set(cs_df['Nome_Bairro_NORM'].dropna().astype(str).unique()):
            continue
        new = {c: 0 for c in df.columns}
        new['Nome_Bairro'] = row['Nome_Bairro']
        new['Nome_Bairro_NORM'] = tnorm
        if 'geometry_wkt' in df.columns:
            new['geometry_wkt'] = row.geometry.wkt
        elif 'geometry' in df.columns:
            new['geometry'] = row.geometry
        if 'Qtd_Empresas' in df.columns and tnorm in emp_counts:
            new['Qtd_Empresas'] = emp_counts.get(tnorm, 0)
        cs_df = pd.concat([cs_df, pd.DataFrame([new])], ignore_index=True)
        if len(cs_df) >= 15:
            break

# final check: remove duplicates by Nome_Bairro_NORM (keep aggregated)
cs_df = cs_df.groupby('Nome_Bairro_NORM', dropna=False).agg(lambda x: x.iloc[0] if x.dtype == object else x.sum()).reset_index()

if cs_df.empty:
    raise SystemExit("No Centro-Sul neighborhoods found in the Parquet after enrichment; aborting.")

# Fill Renda_Media NaNs with median
if 'Renda_Media' in cs_df.columns:
    med = float(cs_df['Renda_Media'].median(skipna=True) if not cs_df['Renda_Media'].dropna().empty else 0.0)
    missing = int(cs_df['Renda_Media'].isna().sum())
    if missing > 0:
        cs_df['Renda_Media'] = cs_df['Renda_Media'].fillna(med)
        print(f'Filled {missing} Centro-Sul Renda_Media with median {med:.4f}')

# Recompute Score_Renda and Apetite and normalize on Centro-Sul subset
cs_df['Score_Renda'] = minmax_scale_series(cs_df['Renda_Media'].fillna(0.0)).fillna(0.0)
cs_df['Inverse_Saturacao'] = (1.0 - cs_df['Saturacao_Comercial']).clip(0,1).fillna(0.0)
cs_df['Apetite_Investidor'] = ((0.4 * cs_df['Score_Mobilidade']) + (0.3 * cs_df['Score_Renda']) + (0.3 * cs_df['Inverse_Saturacao'])).clip(0,1)
# normalize Apetite min-max on Centro-Sul
cs_df['Apetite_Investidor'] = minmax_scale_series(cs_df['Apetite_Investidor']).clip(0,1)
# ensure top 10% at least 0.8
p90 = float(cs_df['Apetite_Investidor'].quantile(0.9))
if p90 > 0 and p90 < 0.8:
    factor = 0.8 / p90
    cs_df['Apetite_Investidor'] = (cs_df['Apetite_Investidor'] * factor).clip(0,1)
    print(f'Scaled Centro-Sul Apetite by factor {factor:.3f} to raise p90 from {p90:.3f}')

# Prepare for saving: ensure consistent dtypes for Parquet (strings for object cols) and geometry as WKT
# convert shapely geometry to geometry_wkt if present
if 'geometry' in cs_df.columns and 'geometry_wkt' not in cs_df.columns:
    try:
        import shapely.wkt as wkt
        cs_df['geometry_wkt'] = cs_df['geometry'].apply(lambda g: g.wkt if g is not None else None)
    except Exception:
        cs_df['geometry_wkt'] = None
# coerce object columns to strings to satisfy pyarrow
for c in cs_df.columns:
    if cs_df[c].dtype == object:
        cs_df[c] = cs_df[c].astype(str).fillna('')

cs_df.to_parquet(PARQUET, index=False)
print(f'Wrote filtered Centro-Sul parquet with {len(cs_df)} rows to {PARQUET}')
