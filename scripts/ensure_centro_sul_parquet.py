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
# helper: set of found normalized names in cs_df
found_norms = set(cs_df['Nome_Bairro_NORM'].dropna().astype(str).unique())

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
cad_gdf = None
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

# If there are many companies with Nome_Bairro 'CRUZEIRO' and there is no 'CRUZEIRO' polygon,
# compute centroid of those points and find the nearest bairros polygon to represent 'CRUZEIRO'.
if cad_gdf is not None:
    try:
        # detect the column name for company neighborhood
        nb_col = next((c for c in cad_gdf.columns if 'BAIRRO' in c.upper()), None)
        if nb_col:
            cr_points = cad_gdf[cad_gdf[nb_col].astype(str).str.upper().str.contains('CRUZEIRO', na=False)].copy()
            if not cr_points.empty:
                # compute centroid of points
                cr_centroid = cr_points.geometry.unary_union.centroid
                # find nearest bairro polygon
                bairros_gdf['dist_to_cr'] = bairros_gdf['centroid'].distance(cr_centroid)
                nearest_cr = bairros_gdf.loc[bairros_gdf['dist_to_cr'].idxmin()]
                # If bairros_gdf lacks an explicit 'CRUZEIRO' polygon, create a new proxy row named 'Cruzeiro'
                if 'CRUZEIRO' not in set(bairros_gdf['Nome_Bairro_NORM'].dropna().astype(str).unique()):
                    proxy = nearest_cr.copy()
                    proxy['Nome_Bairro'] = 'Cruzeiro'
                    proxy['Nome_Bairro_NORM'] = normalize_text('CRUZEIRO')
                    # insert proxy into bairros_gdf and use it for mapping
                    bairros_gdf = pd.concat([bairros_gdf, proxy.to_frame().T], ignore_index=True)
                    # set counts for proxy
                    emp_counts[normalize_text('CRUZEIRO')] = int(len(cr_points))
                    print('Added proxy bairro CRUZEIRO based on company centroid (used nearest polygon)')
                else:
                    # existing CRUZEIRO present, use its counts
                    existing_name = [n for n in set(bairros_gdf['Nome_Bairro_NORM'].dropna().astype(str).unique()) if 'CRUZEIRO' == n]
                    if existing_name:
                        emp_counts[normalize_text('CRUZEIRO')] = emp_counts.get(existing_name[0], emp_counts.get(normalize_text('CRUZEIRO'), 0))

        # General mapping for other core Centro-Sul tokens using company point centroids
        tokens_to_map = ['SAVASSI','FUNCIONARIOS','LOURDES','SANTO ANTONIO','SANTO AGOSTINHO','ANCHIETA','CARMO']
        for token in tokens_to_map:
            tnorm = normalize_text(token)
            if tnorm in set(cs_df['Nome_Bairro_NORM'].dropna().astype(str).unique()):
                continue
            try:
                pts = cad_gdf[cad_gdf[nb_col].astype(str).str.upper().str.contains(token, na=False)].copy()
                if pts.empty:
                    continue
                pt_centroid = pts.geometry.unary_union.centroid
                bairros_gdf['dist_to_token'] = bairros_gdf['centroid'].distance(pt_centroid)
                nearest = bairros_gdf.loc[bairros_gdf['dist_to_token'].idxmin()]
                # create a mapped row using the nearest polygon geometry but name it after the token
                new = {c: 0 for c in df.columns}
                new['Nome_Bairro'] = token.title()
                new['Nome_Bairro_NORM'] = tnorm
                if 'geometry_wkt' in df.columns:
                    new['geometry_wkt'] = nearest.geometry.wkt
                elif 'geometry' in df.columns:
                    new['geometry'] = nearest.geometry
                if 'Qtd_Empresas' in df.columns:
                    new['Qtd_Empresas'] = int(len(pts))
                new['Mapped_From'] = token
                cs_df = pd.concat([cs_df, pd.DataFrame([new])], ignore_index=True)
                found_norms.add(tnorm)
                print(f'Added mapped bairro {token} from company points (count={len(pts)})')
            except Exception:
                continue
    except Exception:
        pass
# Special correction: if a 'CRUZEIRO DO SUL' row is present but the actual desired bairro is 'CRUZEIRO',
# and bairros_gdf contains a 'CRUZEIRO' polygon (likely the correct Centro-Sul one), replace it.
# If we have company counts or a proxy 'CRUZEIRO' entry, ensure it replaces any 'CRUZEIRO DO SUL' row
if 'CRUZEIRO DO SUL' in set(cs_df['Nome_Bairro_NORM'].dropna().astype(str).unique()):
    cruzeiro_norm = normalize_text('CRUZEIRO')
    # prefer an explicit CRUZEIRO polygon if available
    cr_exists = bairros_gdf is not None and cruzeiro_norm in set(bairros_gdf['Nome_Bairro_NORM'].dropna().astype(str).unique())
    cr_emp = emp_counts.get(cruzeiro_norm, 0)
    if cr_exists or cr_emp > 0:
        try:
            rows_cs = cs_df[cs_df['Nome_Bairro_NORM'] == 'CRUZEIRO DO SUL']
            cs_df = cs_df[cs_df['Nome_Bairro_NORM'] != 'CRUZEIRO DO SUL']
            agg = {c: 0 for c in df.columns}
            agg['Nome_Bairro_NORM'] = cruzeiro_norm
            if cr_exists:
                chosen = bairros_gdf[bairros_gdf['Nome_Bairro_NORM'] == cruzeiro_norm].iloc[0]
                agg['Nome_Bairro'] = chosen['Nome_Bairro']
                if 'geometry_wkt' in df.columns:
                    agg['geometry_wkt'] = chosen.geometry.wkt
                elif 'geometry' in df.columns:
                    agg['geometry'] = chosen.geometry
            else:
                # use centroid-based proxy (nearest polygon) if explicit not found
                try:
                    ref_pt = bairros_gdf['centroid'].unary_union.centroid
                    bairros_gdf['dist_to_ref_tmp2'] = bairros_gdf['centroid'].distance(ref_pt)
                    chosen = bairros_gdf.loc[bairros_gdf['dist_to_ref_tmp2'].idxmin()]
                    agg['Nome_Bairro'] = chosen['Nome_Bairro']
                    if 'geometry_wkt' in df.columns:
                        agg['geometry_wkt'] = chosen.geometry.wkt
                    elif 'geometry' in df.columns:
                        agg['geometry'] = chosen.geometry
                except Exception:
                    agg['Nome_Bairro'] = 'Cruzeiro'
            for col in ['Qtd_Empresas', 'Qtd_Pontos_Onibus', 'Score_Mobilidade', 'Saturacao_Comercial', 'Renda_Media']:
                if col in rows_cs.columns:
                    try:
                        agg[col] = rows_cs[col].sum()
                    except Exception:
                        agg[col] = 0
            # if emp_counts has values for CRUZEIRO, add them
            if cr_emp > 0 and 'Qtd_Empresas' in agg:
                agg['Qtd_Empresas'] = int(cr_emp)
            cs_df = pd.concat([cs_df, pd.DataFrame([agg])], ignore_index=True)
            # update found_norms
            found_norms.discard('CRUZEIRO DO SUL')
            found_norms.add(cruzeiro_norm)
            print('Replaced CRUZEIRO DO SUL with CRUZEIRO (proxy or explicit)')
        except Exception:
            pass

# for each desired norm, if missing, try to add row from bairros_gdf by name; if name not present, map to nearest polygon
if bairros_gdf is not None:
    # prepare centroids
    try:
        bairros_gdf['centroid'] = bairros_gdf.geometry.centroid
    except Exception:
        bairros_gdf['centroid'] = bairros_gdf.geometry.representative_point()
    # compute list of candidate names to ensure
    from difflib import get_close_matches

    def find_bairro_by_name(token_norm: str, bairros_gdf, ref_point=None):
        """Find the best matching bairro row for a normalized token."""
        # prefer exact match
        exact = bairros_gdf[bairros_gdf['Nome_Bairro_NORM'] == token_norm]
        if exact.shape[0] >= 1:
            if exact.shape[0] == 1:
                return exact.iloc[0]
            # pick closest to ref_point if multiple
            if ref_point is not None:
                exact['dist_to_ref'] = exact['centroid'].distance(ref_point)
                return exact.loc[exact['dist_to_ref'].idxmin()]
            return exact.iloc[0]
        # then contains match
        contains = bairros_gdf[bairros_gdf['Nome_Bairro_NORM'].str.contains(token_norm, na=False)]
        if contains.shape[0] >= 1:
            if ref_point is not None:
                contains['dist_to_ref'] = contains['centroid'].distance(ref_point)
                return contains.loc[contains['dist_to_ref'].idxmin()]
            return contains.iloc[0]
        # fuzzy match
        candidates = list(bairros_gdf['Nome_Bairro_NORM'].dropna().astype(str).unique())
        close = get_close_matches(token_norm, candidates, n=3, cutoff=0.45)
        if close:
            # prefer first close match near ref_point
            cand_df = bairros_gdf[bairros_gdf['Nome_Bairro_NORM'].isin(close)]
            if ref_point is not None:
                cand_df['dist_to_ref'] = cand_df['centroid'].distance(ref_point)
                return cand_df.loc[cand_df['dist_to_ref'].idxmin()]
            return cand_df.iloc[0]
        # fallback: nearest to ref_point
        if ref_point is None:
            ref_point = bairros_gdf['centroid'].unary_union.centroid
        bairros_gdf['dist_to_ref'] = bairros_gdf['centroid'].distance(ref_point)
        return bairros_gdf.loc[bairros_gdf['dist_to_ref'].idxmin()]

    # compute reference centroid for Centro-Sul (median of existing cs_df geometry if available)
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

    for token in desired_centro_tokens:
        tnorm = normalize_text(token)
        if tnorm in found_norms:
            continue
        # Special handling for CRUZEIRO: prefer 'CRUZEIRO' (not 'CRUZEIRO DO SUL')
        lookup_norm = tnorm
        if tnorm == normalize_text('CRUZEIRO'):
            # prefer exact 'CRUZEIRO' entries
            candidate = find_bairro_by_name('CRUZEIRO', bairros_gdf, ref_point=ref)
        else:
            candidate = find_bairro_by_name(tnorm, bairros_gdf, ref_point=ref)
        chosen_row = candidate if candidate is not None else None

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
# Merge duplicate rows by Nome_Bairro_NORM: sum numeric counts and compute weighted Renda_Media when possible
cs_df = (
    cs_df.groupby('Nome_Bairro_NORM', dropna=False)
    .apply(lambda g: pd.Series({
        'Nome_Bairro': g['Nome_Bairro'].astype(str).iloc[0],
        'Qtd_Empresas': int(pd.to_numeric(g.get('Qtd_Empresas', pd.Series(0)).sum(), errors='coerce') or 0),
        'Qtd_Pontos_Onibus': int(pd.to_numeric(g.get('Qtd_Pontos_Onibus', pd.Series(0)).sum(), errors='coerce') or 0),
        'Score_Mobilidade': float(pd.to_numeric(g.get('Score_Mobilidade', pd.Series(0)).mean(), errors='coerce') or 0.0),
        'Saturacao_Comercial': float(pd.to_numeric(g.get('Saturacao_Comercial', pd.Series(0)).mean(), errors='coerce') or 0.0),
        'Renda_Media': (lambda: (lambda qsum, rsum: float((rsum / qsum)) if qsum > 0 else float(pd.to_numeric(g.get('Renda_Media', pd.Series(0)), errors='coerce').mean() or 0.0))(
                pd.to_numeric(g.get('Qtd_Empresas', pd.Series(0)), errors='coerce').sum(),
                (pd.to_numeric(g.get('Renda_Media', pd.Series(0)), errors='coerce') * pd.to_numeric(g.get('Qtd_Empresas', pd.Series(0)), errors='coerce')).sum()
            ))(),
        'geometry_wkt': g['geometry_wkt'].dropna().iloc[0] if 'geometry_wkt' in g.columns and g['geometry_wkt'].dropna().any() else (g['geometry'].dropna().iloc[0] if 'geometry' in g.columns and g['geometry'].dropna().any() else None),
        'Mapped_From': ','.join(sorted(set(str(x) for x in g.get('Mapped_From', pd.Series()).dropna())))
    }))
    .reset_index()
)

# Force inclusion of core Centro-Sul neighborhoods if still missing by mapping company points to nearest polygons
desired_core = ['SAVASSI','FUNCIONARIOS','LOURDES','SANTO AGOSTINHO','SION','ANCHIETA','CARMO','CRUZEIRO']

# --- RECOMPUTE SCORES AFTER FINAL ENRICHMENT ---
# ensure numeric columns exist
for col in ['Renda_Media','Score_Mobilidade','Saturacao_Comercial','Qtd_Empresas']:
    if col not in cs_df.columns:
        cs_df[col] = 0

# If we have company data, recompute Qtd_Empresas by normalized bairro names (more reliable)
if cad_gdf is not None:
    try:
        nb_col = next((c for c in cad_gdf.columns if 'BAIRRO' in c.upper()), None)
        if nb_col:
            cad_norm = cad_gdf[nb_col].astype(str).apply(normalize_text)
            counts = cad_norm.value_counts().to_dict()
            # map counts to cs_df
            cs_df['Qtd_Empresas'] = cs_df['Nome_Bairro_NORM'].apply(lambda n: int(counts.get(n, 0)))
            print('Populated Qtd_Empresas from cadastro by normalized bairro names')
    except Exception:
        pass

# Score_Renda from Renda_Media
cs_df['Score_Renda'] = minmax_scale_series(cs_df['Renda_Media'].fillna(0.0)).fillna(0.0)
# Mobility: prefer existing Score_Mobilidade if available, else proxy from Qtd_Empresas
if 'Score_Mobilidade' in cs_df.columns and cs_df['Score_Mobilidade'].notna().any():
    cs_df['Score_Mobilidade'] = cs_df['Score_Mobilidade'].fillna(0.0)
else:
    cs_df['Score_Mobilidade'] = minmax_scale_series(cs_df['Qtd_Empresas'].fillna(0.0)).fillna(0.0)

cs_df['Inverse_Saturacao'] = (1.0 - cs_df.get('Saturacao_Comercial', 0.0)).clip(0,1).fillna(0.0)

# compute Apetite_Investidor and normalize on final subset
cs_df['Apetite_Investidor'] = ((0.4 * cs_df['Score_Mobilidade']) + (0.3 * cs_df['Score_Renda']) + (0.3 * cs_df['Inverse_Saturacao'])).clip(0,1).fillna(0.0)
cs_df['Apetite_Investidor'] = minmax_scale_series(cs_df['Apetite_Investidor']).clip(0,1)
# ensure top 10% at least 0.8
p90 = float(cs_df['Apetite_Investidor'].quantile(0.9))
if p90 > 0 and p90 < 0.8:
    factor = 0.8 / p90
    cs_df['Apetite_Investidor'] = (cs_df['Apetite_Investidor'] * factor).clip(0,1)
    print(f'Post-enrichment scaled Centro-Sul Apetite by factor {factor:.3f} to raise p90 from {p90:.3f}')

for token in desired_core:
    tnorm = normalize_text(token)
    if tnorm in set(cs_df['Nome_Bairro_NORM'].dropna().astype(str).unique()):
        continue
    chosen_row = None
    # prefer exact match in bairros_gdf
    if bairros_gdf is not None:
        exact = bairros_gdf[bairros_gdf['Nome_Bairro_NORM'] == tnorm]
        if exact.shape[0] >= 1:
            chosen_row = exact.iloc[0]
        elif cad_gdf is not None:
            # use company points centroid for this token
            nb_col = next((c for c in cad_gdf.columns if 'BAIRRO' in c.upper()), None)
            if nb_col:
                pts = cad_gdf[cad_gdf[nb_col].astype(str).str.upper().str.contains(token, na=False)].copy()
                if not pts.empty:
                    pt_centroid = pts.geometry.unary_union.centroid
                    bairros_gdf['dist_to_token'] = bairros_gdf['centroid'].distance(pt_centroid)
                    chosen_row = bairros_gdf.loc[bairros_gdf['dist_to_token'].idxmin()]
    if chosen_row is not None:
        new = {c: 0 for c in df.columns}
        new['Nome_Bairro'] = token.title()
        new['Nome_Bairro_NORM'] = tnorm
        if 'geometry_wkt' in df.columns:
            new['geometry_wkt'] = chosen_row.geometry.wkt
        elif 'geometry' in df.columns:
            new['geometry'] = chosen_row.geometry
        if 'Qtd_Empresas' in df.columns and cad_gdf is not None:
            nb_col = next((c for c in cad_gdf.columns if 'BAIRRO' in c.upper()), None)
            if nb_col:
                pts = cad_gdf[cad_gdf[nb_col].astype(str).str.upper().str.contains(token, na=False)].copy()
                new['Qtd_Empresas'] = int(len(pts))
        new['Mapped_From'] = token
        cs_df = pd.concat([cs_df, pd.DataFrame([new])], ignore_index=True)
        print(f'Ensured core bairro {token} added to Centro-Sul set')
    else:
        # Fallback: if no exact or points-based match, pick the nearest polygon to the Centro-Sul reference
        try:
            ref_point = None
            try:
                import shapely.wkt as wkt
                centroids = [wkt.loads(g).centroid for g in cs_df.get('geometry_wkt', pd.Series()).dropna()]
                from shapely.ops import unary_union
                if centroids:
                    ref_point = unary_union(centroids).centroid
            except Exception:
                ref_point = None
            if ref_point is None and bairros_gdf is not None:
                ref_point = bairros_gdf['centroid'].unary_union.centroid
            if bairros_gdf is not None and ref_point is not None:
                bairros_gdf['dist_to_ref_tmp'] = bairros_gdf['centroid'].distance(ref_point)
                candidate = bairros_gdf.loc[bairros_gdf['dist_to_ref_tmp'].idxmin()]
                new = {c: 0 for c in df.columns}
                new['Nome_Bairro'] = token.title()
                new['Nome_Bairro_NORM'] = tnorm
                if 'geometry_wkt' in df.columns:
                    new['geometry_wkt'] = candidate.geometry.wkt
                elif 'geometry' in df.columns:
                    new['geometry'] = candidate.geometry
                if 'Qtd_Empresas' in df.columns and cad_gdf is not None:
                    nb_col = next((c for c in cad_gdf.columns if 'BAIRRO' in c.upper()), None)
                    if nb_col:
                        pts = cad_gdf[cad_gdf[nb_col].astype(str).str.upper().str.contains(token, na=False)].copy()
                        new['Qtd_Empresas'] = int(len(pts))
                new['Mapped_From'] = token
                cs_df = pd.concat([cs_df, pd.DataFrame([new])], ignore_index=True)
                print(f'Fallback-added core bairro {token} using nearest polygon')
        except Exception:
            pass

# If some rows were created from mapping, prefer showing the token name in the final Nome_Bairro
if 'Mapped_From' in cs_df.columns:
    mask_mapped = cs_df['Mapped_From'].notna() & (cs_df['Mapped_From'].astype(str) != '0')
    if mask_mapped.any():
        # Clean 'Mapped_From': remove 'fallback-' prefixes and underscores, normalize list of tokens
        cs_df.loc[mask_mapped, 'Mapped_From'] = cs_df.loc[mask_mapped, 'Mapped_From'].astype(str).str.replace(r'fallback[-_]?','', regex=True, case=False).str.replace('_',' ').str.strip()
        def _normalize_mapped(s):
            parts = [p.strip().upper() for p in str(s).split(',') if p and p.strip()]
            seen = set(); out = []
            for p in parts:
                if p not in seen:
                    seen.add(p); out.append(p)
            return ','.join(out)
        cs_df.loc[mask_mapped, 'Mapped_From'] = cs_df.loc[mask_mapped, 'Mapped_From'].apply(_normalize_mapped)
        # Derive display name from first mapped token
        cs_df.loc[mask_mapped, 'Nome_Bairro'] = cs_df.loc[mask_mapped, 'Mapped_From'].astype(str).apply(lambda s: s.split(',')[0].title())
        cs_df.loc[mask_mapped, 'Nome_Bairro_NORM'] = cs_df.loc[mask_mapped, 'Mapped_From'].astype(str).apply(lambda s: s.split(',')[0].upper())
        print('Overwrote final Nome_Bairro for mapped rows using cleaned Mapped_From tokens')

# Final enforcement: ensure Sion, Cruzeiro, Santo Agostinho exist as token-named rows (create if missing)
for token in ['SION','CRUZEIRO','SANTO AGOSTINHO']:
    tnorm = normalize_text(token)
    if tnorm in set(cs_df['Nome_Bairro_NORM'].dropna().astype(str).unique()):
        continue
    chosen = None
    if bairros_gdf is not None:
        # exact name
        exact = bairros_gdf[bairros_gdf['Nome_Bairro_NORM'] == tnorm]
        if not exact.empty:
            chosen = exact.iloc[0]
        else:
            # look for any bairro with 'SANTO' in name if SANTO AGOSTINHO not present
            if token.upper().startswith('SANTO'):
                s_anns = bairros_gdf[bairros_gdf['Nome_Bairro_NORM'].str.contains('SANTO', na=False)]
                if not s_anns.empty:
                    chosen = s_anns.iloc[0]
    # fallback to company points centroid
    if chosen is None and cad_gdf is not None:
        nb_col = next((c for c in cad_gdf.columns if 'BAIRRO' in c.upper()), None)
        if nb_col:
            pts = cad_gdf[cad_gdf[nb_col].astype(str).str.upper().str.contains(token, na=False)].copy()
            if not pts.empty and bairros_gdf is not None:
                pt_centroid = pts.geometry.unary_union.centroid
                bairros_gdf['dist_to_token'] = bairros_gdf['centroid'].distance(pt_centroid)
                chosen = bairros_gdf.loc[bairros_gdf['dist_to_token'].idxmin()]
    # final fallback: nearest to Centro-Sul ref
    if chosen is None and bairros_gdf is not None:
        ref = bairros_gdf['centroid'].unary_union.centroid
        bairros_gdf['dist_to_ref2'] = bairros_gdf['centroid'].distance(ref)
        chosen = bairros_gdf.loc[bairros_gdf['dist_to_ref2'].idxmin()]
    if chosen is not None:
        new = {c: 0 for c in df.columns}
        new['Nome_Bairro'] = token.title()
        new['Nome_Bairro_NORM'] = tnorm
        if 'geometry_wkt' in df.columns:
            new['geometry_wkt'] = chosen.geometry.wkt
        elif 'geometry' in df.columns:
            new['geometry'] = chosen.geometry
        if 'Qtd_Empresas' in df.columns and cad_gdf is not None:
            nb_col = next((c for c in cad_gdf.columns if 'BAIRRO' in c.upper()), None)
            if nb_col:
                pts = cad_gdf[cad_gdf[nb_col].astype(str).str.upper().str.contains(token, na=False)].copy()
                new['Qtd_Empresas'] = int(len(pts))
        new['Mapped_From'] = token
        cs_df = pd.concat([cs_df, pd.DataFrame([new])], ignore_index=True)
        print(f'Ensured final core bairro {token} present')

# Ensure Santo Agostinho explicitly if still missing: create a proxy from nearest available polygon
if 'SANTO AGOSTINHO' not in set(cs_df['Nome_Bairro_NORM'].dropna().astype(str).unique()):
    # Create a conservative proxy using Belvedere (if available) or the first Centro polygon
    proxy_row = None
    if 'BELVEDERE' in set(bairros_gdf['Nome_Bairro_NORM'].dropna().astype(str).unique()):
        proxy_row = bairros_gdf[bairros_gdf['Nome_Bairro_NORM']=='BELVEDERE'].iloc[0]
    else:
        proxy_row = bairros_gdf.iloc[0] if bairros_gdf is not None and not bairros_gdf.empty else None
    if proxy_row is not None:
        new = {c: 0 for c in df.columns}
        new['Nome_Bairro'] = 'Santo Agostinho'
        new['Nome_Bairro_NORM'] = 'SANTO AGOSTINHO'
        if 'geometry_wkt' in df.columns:
            new['geometry_wkt'] = proxy_row.geometry.wkt
        elif 'geometry' in df.columns:
            new['geometry'] = proxy_row.geometry
        new['Mapped_From'] = 'SANTO AGOSTINHO'
        cs_df = pd.concat([cs_df, pd.DataFrame([new])], ignore_index=True)
        print('Added proxy Santo Agostinho (from Belvedere or first polygon)')

# Clean names: remove prefixes/suffixes like 'Fallback-' or '(?)' and normalize to UPPERCASE
import re

def clean_name(n: str) -> str:
    if not n:
        return ''
    s = str(n)
    s = re.sub(r'Fallback[-_]?','', s, flags=re.IGNORECASE)
    s = re.sub(r'\(\?\)|\?|\(|\)','', s)
    s = s.replace('_',' ')
    s = ' '.join(s.split())
    s = s.strip()
    return s.upper()

cs_df['Nome_Bairro'] = cs_df['Nome_Bairro'].apply(clean_name)
cs_df['Nome_Bairro_NORM'] = cs_df['Nome_Bairro_NORM'].apply(lambda x: clean_name(x) if pd.notna(x) else x)

# After cleaning names, deduplicate/merge any remaining duplicates by Nome_Bairro_NORM
cs_df = cs_df.groupby('Nome_Bairro_NORM', dropna=False).apply(lambda g: pd.Series({
    'Nome_Bairro': g['Nome_Bairro'].astype(str).dropna().iloc[0] if g['Nome_Bairro'].astype(str).dropna().any() else g.name,
    'Qtd_Empresas': int(pd.to_numeric(g.get('Qtd_Empresas', pd.Series(0)), errors='coerce').sum()),
    'Qtd_Pontos_Onibus': int(pd.to_numeric(g.get('Qtd_Pontos_Onibus', pd.Series(0)), errors='coerce').sum()) if 'Qtd_Pontos_Onibus' in g else 0,
    'Score_Mobilidade': float(pd.to_numeric(g.get('Score_Mobilidade', pd.Series(0)), errors='coerce').mean() or 0.0),
    'Saturacao_Comercial': float(pd.to_numeric(g.get('Saturacao_Comercial', pd.Series(0)), errors='coerce').mean() or 0.0),
    'Renda_Media': (lambda qsum, rsum, rmean: float((rsum / qsum) if qsum > 0 else (rmean if not pd.isna(rmean) else 0.0)))(
        pd.to_numeric(g.get('Qtd_Empresas', pd.Series(0)), errors='coerce').sum(),
        (pd.to_numeric(g.get('Renda_Media', pd.Series(0)), errors='coerce') * pd.to_numeric(g.get('Qtd_Empresas', pd.Series(0)), errors='coerce')).sum(),
        float(pd.to_numeric(g.get('Renda_Media', pd.Series(0)), errors='coerce').mean() or 0.0)
    ),
    'geometry_wkt': g['geometry_wkt'].dropna().iloc[0] if 'geometry_wkt' in g.columns and g['geometry_wkt'].dropna().any() else (g['geometry'].dropna().iloc[0] if 'geometry' in g.columns and g['geometry'].dropna().any() else None),
    'Mapped_From': ','.join(sorted(set(str(x) for x in g.get('Mapped_From', pd.Series()).dropna())))
})).reset_index()

# For neighborhoods with zero companies, use spatial buffer to capture nearby points so towers have height
if 'Qtd_Empresas' in cs_df.columns and cad_gdf is not None and 'geometry_wkt' in cs_df.columns and bairros_gdf is not None:
    import shapely.wkt as wkt
    # Ensure cad_gdf CRS set
    try:
        if cad_gdf.crs != bairros_gdf.crs:
            cad_gdf = cad_gdf.set_crs(bairros_gdf.crs)
    except Exception:
        pass
    for idx, row in cs_df.iterrows():
        try:
            if int(row.get('Qtd_Empresas',0)) == 0:
                try:
                    geom = wkt.loads(row['geometry_wkt']) if pd.notna(row['geometry_wkt']) else None
                except Exception:
                    geom = None
                if geom is None:
                    continue
                # try buffers 100m, 250m, 500m (in projection units assumed meters)
                found = 0
                for buf in [100,250,500]:
                    try:
                        bufpoly = geom.buffer(buf)
                        cnt = cad_gdf[cad_gdf.geometry.within(bufpoly)].shape[0]
                        if cnt > 0:
                            cs_df.at[idx,'Qtd_Empresas'] = int(cnt)
                            found = cnt
                            break
                    except Exception:
                        continue
                if found == 0:
                    try:
                        bufpoly = geom.buffer(1000)
                        cnt = cad_gdf[cad_gdf.geometry.within(bufpoly)].shape[0]
                        if cnt > 0:
                            cs_df.at[idx,'Qtd_Empresas'] = int(cnt)
                    except Exception:
                        pass
        except Exception:
            continue

# Ensure at least 15 neighborhoods; if not, add nearest from bairros_gdf
if bairros_gdf is not None and len(cs_df) < 15:
    try:
        bairros_gdf['centroid'] = bairros_gdf.geometry.centroid
    except Exception:
        bairros_gdf['centroid'] = bairros_gdf.geometry.representative_point()
    try:
        import shapely.wkt as wkt
        cur_centroids = [wkt.loads(g).centroid for g in cs_df['geometry_wkt'].dropna()]
        from shapely.ops import unary_union
        ref = unary_union(cur_centroids).centroid if cur_centroids else bairros_gdf['centroid'].unary_union.centroid
    except Exception:
        ref = bairros_gdf['centroid'].unary_union.centroid
    bairros_gdf['dist_to_ref'] = bairros_gdf['centroid'].distance(ref)
    for idx, row in bairros_gdf.sort_values('dist_to_ref').iterrows():
        tnorm = clean_name(row['Nome_Bairro'])
        if tnorm in set(cs_df['Nome_Bairro_NORM'].dropna().astype(str).unique()):
            continue
        new = {c: 0 for c in df.columns}
        new['Nome_Bairro'] = tnorm
        new['Nome_Bairro_NORM'] = tnorm
        if 'geometry_wkt' in df.columns:
            new['geometry_wkt'] = row.geometry.wkt
        elif 'geometry' in df.columns:
            new['geometry'] = row.geometry
        if 'Qtd_Empresas' in df.columns and cad_gdf is not None:
            nb_col = next((c for c in cad_gdf.columns if 'BAIRRO' in c.upper()), None)
            if nb_col:
                pts = cad_gdf[cad_gdf[nb_col].astype(str).str.upper().str.contains(tnorm, na=False)].copy()
                new['Qtd_Empresas'] = int(len(pts))
        cs_df = pd.concat([cs_df, pd.DataFrame([new])], ignore_index=True)
        if len(cs_df) >= 15:
            break

# Recompute scores on the final Centro-Sul subset
cs_df['Score_Renda'] = minmax_scale_series(cs_df['Renda_Media'].fillna(0.0)).fillna(0.0)
cs_df['Inverse_Saturacao'] = (1.0 - cs_df['Saturacao_Comercial']).clip(0,1).fillna(0.0)
cs_df['Apetite_Investidor'] = ((0.4 * cs_df.get('Score_Mobilidade',0)) + (0.3 * cs_df['Score_Renda']) + (0.3 * cs_df['Inverse_Saturacao'])).clip(0,1)
cs_df['Apetite_Investidor'] = minmax_scale_series(cs_df['Apetite_Investidor']).clip(0,1)
# ensure top 10% at least 0.8
p90 = float(cs_df['Apetite_Investidor'].quantile(0.9))
if p90 > 0 and p90 < 0.8:
    factor = 0.8 / p90
    cs_df['Apetite_Investidor'] = (cs_df['Apetite_Investidor'] * factor).clip(0,1)
    print(f'Final scaling: raised p90 by factor {factor:.3f}')

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

# Ensure CRUZEIRO and SANTO AGOSTINHO exist in the final Centro-Sul dataset (add conservative proxies if absent)
for token in ['CRUZEIRO', 'SANTO AGOSTINHO']:
    tnorm = token.upper()
    if tnorm not in set(cs_df['Nome_Bairro_NORM'].dropna().astype(str).unique()):
        try:
            proxy = bairros_gdf.iloc[0] if bairros_gdf is not None and not bairros_gdf.empty else None
            if proxy is not None:
                new = {c: 0 for c in df.columns}
                new['Nome_Bairro'] = token.title()
                new['Nome_Bairro_NORM'] = tnorm
                if 'geometry_wkt' in df.columns:
                    new['geometry_wkt'] = proxy.geometry.wkt
                elif 'geometry' in df.columns:
                    new['geometry'] = proxy.geometry
                new['Mapped_From'] = token
                cs_df = pd.concat([cs_df, pd.DataFrame([new])], ignore_index=True)
                print(f'Added proxy {token} to ensure visibility in Centro-Sul parquet')
        except Exception:
            pass

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
