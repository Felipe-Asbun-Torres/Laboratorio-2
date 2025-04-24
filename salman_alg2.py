import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
import time

# Cargar el archivo .txt
df = pd.read_csv(r'C:/Users/PC RST/Downloads/CP_fases.txt', sep=r'\s+', engine='python')

# Asegurar que el índice del bloque está bien nombrado
df = df.rename(columns={"0": "block_id"})
df.set_index("block_id", inplace=True)

bancos = df['z'].unique()
bancos.sort()

# Parámetros del algoritmo
features = ['au', 'cpy', 'cueq', 'cus', 'cut', 'density']
Cmax = 10
MERGE_THRESHOLD_1 = 0.7
MERGE_THRESHOLD_2 = 0.5
MAX_CLUSTER_SIZE = 40
MIN_CLUSTER_SIZE = 10
r = 0.4 # Penalización por tipo de roca
MAX_ITERATIONS = 1 # Iteraciones para formar los clusters espaciales 

# Resultado final
df["spatial_cluster"] = -1
global_cluster_id = 0

start_time = time.time()

for z in bancos:
    banco_df = df[df['z'] == z].copy()
    # Asumimos dirección de minado de oeste a este
    banco_df = banco_df.sort_values(by="x")

    if banco_df.shape[0] < 5:
        continue  # saltar bancos muy pequeños

    # ===== Paso 1: K-means sólo con bloques minerales =====
    banco_mineral = banco_df[banco_df["material"] != 8].copy()
    if banco_mineral.shape[0] < 2:
        continue

    scores = {}
    for k in range(2, min(Cmax, len(banco_mineral))):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(banco_mineral[features])
        score = silhouette_score(banco_mineral[features], kmeans.labels_)
        scores[k] = score

    best_k = max(scores, key=scores.get)
    k_final = best_k + 1

    kmeans = KMeans(n_clusters=best_k, random_state=42).fit(banco_mineral[features])
    banco_mineral["cluster"] = kmeans.labels_
    banco_df.loc[banco_mineral.index, "cluster"] = banco_mineral["cluster"]
    banco_df.loc[banco_df["material"] == 8, "cluster"] = k_final - 1

    # ===== Paso 2: Similaridad espacial en el banco =====
    best_DH, best_RU = 0, 0
    best_clusters = None
    best_block_cluster = None

    for iteration in range(MAX_ITERATIONS):        
        max_d = 0.0
        def get_neighbors(i, banco):
            xi, yi = banco.loc[i, ['x', 'y']]
            neighbors = banco.index[banco.apply(
                lambda row: 0 < euclidean((xi, yi), (row['x'], row['y'])) <= 10, axis=1)]
            return list(neighbors)
        
        centroids = kmeans.cluster_centers_

        # Matriz de atributos del banco
        X = banco_mineral[features].values

        # Distancia a cada centroide
        distances = cdist(X, centroids)  # shape: (n_blocks, k_clusters)

        # Convertir a similaridad inversa (más cercano = más parecido)
        epsilon = 1e-6
        similarities = 1 / (distances + epsilon)

        # Normalizar para que sumen 1 → esto será nuestro vector de pertenencia
        membership_vectors = similarities / similarities.sum(axis=1, keepdims=True)

        # Guardar en DataFrame
        membership_df = pd.DataFrame(membership_vectors, index=banco_mineral.index)
        membership_df.columns = [f"m_k{k}" for k in range(best_k)]

        # Nuevas claves
        membership_cols = [f"m_k{k}" for k in range(best_k)]

        # Elimina columnas de pertenencia si ya existen antes del join
        for col in membership_cols:
            if col in banco_df.columns:
                banco_df.drop(columns=col, inplace=True)

        banco_df = banco_df.join(membership_df, how="left")

        similar = {}
        
        for i in banco_df.index:
            if banco_df.loc[i, 'material'] == 8:  # estéril
                continue
            neighbors = get_neighbors(i, banco_df)
            for u in neighbors:
                if banco_df.loc[u, 'material'] == 8:
                    continue
                vec_i = banco_df.loc[i, membership_cols].values
                vec_u = banco_df.loc[u, membership_cols].values
                d_iu = np.linalg.norm(vec_i - vec_u)

                if banco_df.loc[i, 'tipomineral'] != banco_df.loc[u, 'tipomineral']:
                    d_iu /= r  # aumenta la "distancia" si son de distinta roca

                similar[(i, u)] = d_iu
                max_d = max(max_d, d_iu)

        # Convertir a similitud
        S = {k: (max_d - v) / max_d for k, v in similar.items()}

        # ===== Paso 3: Clustering espacial por similitud =====
        clusters = {}
        block_cluster = {}
        visited = set()

        for block in banco_df.index:
            if block in visited:
                continue
            current_cluster = [block]
            visited.add(block)
            neighbors = get_neighbors(block, banco_df)

            for n in neighbors:
                if len(current_cluster) >= MAX_CLUSTER_SIZE:
                    break
                if (block, n) in S and S[(block, n)] > MERGE_THRESHOLD_1 and n not in visited:
                    current_cluster.append(n)
                    visited.add(n)

            for b in current_cluster:
                block_cluster[b] = global_cluster_id
            clusters[global_cluster_id] = current_cluster
            global_cluster_id += 1

        # ===== Paso 4: Asignar bloques sin clúster =====
        for block in banco_df.index:
            if block not in block_cluster:
                neighbors = get_neighbors(block, banco_df)
                for n in neighbors:
                    if (block, n) in S and S[(block, n)] > MERGE_THRESHOLD_2 and n in block_cluster:
                        cid = block_cluster[n]
                        clusters[cid].append(block)
                        block_cluster[block] = cid
                        break

        # ===== Paso 5: Fusión de clústeres pequeños =====
        for cid, members in list(clusters.items()):
            if len(members) < MIN_CLUSTER_SIZE:
                cluster_grades = banco_df.loc[members, 'cueq'].mean()
                neighbors = [get_neighbors(b, banco_df) for b in members]
                neighbors = set([n for sublist in neighbors for n in sublist if n in block_cluster])
                neighbors = list(neighbors - set(members))

                best_match = None
                min_gap = float('inf')
                for n in neighbors:
                    other_cid = block_cluster[n]
                    other_grade = banco_df.loc[clusters[other_cid], 'cueq'].mean()
                    gap = abs(cluster_grades - other_grade)
                    if gap < min_gap:
                        min_gap = gap
                        best_match = other_cid

                if best_match is not None:
                    clusters[best_match].extend(members)
                    for b in members:
                        block_cluster[b] = best_match
                    del clusters[cid]

        # ===== Paso 6: Refinamiento de forma =====
        for block in banco_df.index:
            if block not in block_cluster:
                continue

            cluster_id = block_cluster[block]
            neighbors = get_neighbors(block, banco_df)
            same_cluster_neighbors = [n for n in neighbors if block_cluster.get(n) == cluster_id]

            # Si es un bloque esquina (solo un vecino del mismo clúster)
            if len(same_cluster_neighbors) == 1:
                # Buscar clústeres vecinos
                neighbor_clusters = {}
                for n in neighbors:
                    cid = block_cluster.get(n)
                    if cid is not None and cid != cluster_id:
                        neighbor_clusters[cid] = neighbor_clusters.get(cid, 0) + 1

                # Encontrar clúster vecino con más vecinos en común
                if neighbor_clusters:
                    best_cid = max(neighbor_clusters, key=neighbor_clusters.get)
                    if neighbor_clusters[best_cid] >= 2:  # al menos 2 vecinos en ese clúster
                        # Reasignar el bloque
                        block_cluster[block] = best_cid
                        clusters[cluster_id].remove(block)
                        clusters[best_cid].append(block)
            
        # === Evaluar calidad de clústeres: DH y RU ===
        cluster_ids = set(block_cluster.values())
        total_blocks = len(banco_df[banco_df['material'] != 8])

        dh_list, ru_list = [], []
        for cid in cluster_ids:
            members = [b for b, c in block_cluster.items() if c == cid]
            destinos = banco_df.loc[members, "material"].value_counts()
            tipos = banco_df.loc[members, "tipomineral"].value_counts()
            if destinos.sum() > 0:
                dh_list.append(destinos.max() / destinos.sum())
            if tipos.sum() > 0:
                ru_list.append(tipos.max() / tipos.sum())

        avg_DH = np.mean(dh_list) if dh_list else 0
        avg_RU = np.mean(ru_list) if ru_list else 0

        if avg_DH + avg_RU > best_DH + best_RU:
            best_DH, best_RU = avg_DH, avg_RU
            best_clusters = clusters.copy()
            best_block_cluster = block_cluster.copy()


    # ===== Guardar resultados en el DataFrame global =====
    for b in banco_df.index:
        df.at[b, 'spatial_cluster'] = best_block_cluster.get(b, -1)

# ==============================
# DEFINICIÓN DE PARÁMETROS
# ==============================
R = 0.85  # recuperación
cm = 2  # costo minado (USD/T)
cp = 10  # costo procesamiento (USD/T)
cr = 0.25  # costo de refinación (USD/lb)
P = 3.8  # precio metal (USD/lb)
F = 2204.62  # lb/T
r = 0.1  # tasa de descuento
Mtmin = 3e7  # toneladas mínimas a extraer por año
Mtpro = 6e6  # capacidad de procesamiento (mineral) por año
T = list(range(1, 21))  # períodos 1 a 20

# ==============================
# AGRUPAR Y CONSTRUIR PARÁMETROS
# ==============================
df['cluster_id'] = df['spatial_cluster']
df = df[df['spatial_cluster'] != -1]

cluster_groups = df.groupby(['fase', 'z', 'cluster_id'])

params = {}  # {(f, b, k): {parametros}}

for (f, b, k), group in cluster_groups:
    tonelaje_total = (group['density'] * 1).sum()
    if tonelaje_total == 0:
        continue

    is_mineral = group['material'] != 8
    tonelaje_mineral = (group[is_mineral]['density'] * 1).sum()
    mu = tonelaje_mineral / tonelaje_total
    eps = 1 - mu
    ley_media = np.average(group['cueq'], weights=group['density'])

    params[(f, b, k)] = {
        'mu': mu,
        'eps': eps,
        'M': tonelaje_total,
        'h': ley_media
    }

# ==============================
# MODELO GUROBI
# ==============================
model = Model("PlanificacionMina")

# VARIABLES DE DECISIÓN
x = model.addVars(params.keys(), T, vtype=GRB.CONTINUOUS, name="x")
y = model.addVars(params.keys(), T, vtype=GRB.BINARY, name="y")
z = model.addVars({(f, b, t) for (f, b, k) in params for t in T}, vtype=GRB.BINARY, name="z")

# FUNCIÓN OBJETIVO
model.setObjective(
    quicksum(
        (1 / (1 + r)**t) * (
            ((P - cr) * R * F * params[f, b, k]['h'] * params[f, b, k]['mu'] * x[f, b, k, t])
            - ((cm + cp) * params[f, b, k]['mu'] * x[f, b, k, t])
            - (cm * params[f, b, k]['eps'] * x[f, b, k, t])
        )
        for (f, b, k) in params for t in T
    ), GRB.MAXIMIZE
)

# ==============================
# RESTRICCIONES
# ==============================

# Restricción: solo una ley k puede aplicarse por periodo
for (f, b) in {(f, b) for (f, b, k) in params}:
    for t in T:
        model.addConstr(quicksum(y[f, b, k, t] for k in [kk for (ff, bb, kk) in params if ff == f and bb == b]) <= 1)

# Extracción solo si se aplicó una ley
for (f, b, k), data in params.items():
    for t in T:
        model.addConstr(x[f, b, k, t] <= y[f, b, k, t] * data['M'])

# Tonelaje acumulado no puede exceder disponibilidad
for (f, b, k), data in params.items():
    for t in T:
        model.addConstr(quicksum(x[f, b, k, tt] for tt in range(1, t+1)) <= data['M'])

# Restricción de producción anual mínima
for t in T:
    model.addConstr(quicksum(x[f, b, k, t] for (f, b, k) in params) <= Mtmin)

# Restricción de capacidad de procesamiento anual
for t in T:
    model.addConstr(quicksum(params[f, b, k]['mu'] * x[f, b, k, t] for (f, b, k) in params) <= Mtpro)

# Relación entre z y x (finalización del banco-fase)
for (f, b, k), data in params.items():
    for t in T:
        model.addConstr(z[f, b, t] * data['M'] >= quicksum(x[f, b, k, tt] for tt in range(1, t+1)))

# Secuencia temporal de z variables
for (f, b, t) in z.keys():
    if t > 1:
        model.addConstr(z[f, b, t-1] <= z[f, b, t])

# ==============================
# RESOLVER Y MOSTRAR RESULTADOS
# ==============================
model.optimize()

end_time = time.time()

if model.status == GRB.OPTIMAL:
    print(f"\nVPN óptimo: {model.objVal:.2f} USD")
else:
    print("No se encontró una solución óptima.")

elapsed = end_time - start_time
print(f"\nTiempo de ejecución: {elapsed:.2f} segundos")