# Librerias
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors 
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors
import numpy.matlib as matlib
import scipy as sp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
import mine_tools as mt

def plot_cluster_precedence(
    df1,
    df2,
    block_width=10,
    block_height=10,
    title='Bloques coloreados por clúster con contorno de df2',
    show_grid=True,
    figsize=(12, 10),
    dpi=100,
    outline_color='red'
):
    # Crear mapa de colores por clúster
    unique_clusters = sorted(df1['cluster'].unique())
    cmap = plt.cm.get_cmap('tab20', len(unique_clusters))
    cluster_to_color = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Dibujar bloques de df1 (rellenados por clúster)
    for _, row in df1.iterrows():
        x, y = row['x'], row['y']
        cluster = row['cluster']
        color = cluster_to_color[cluster]
        rect = patches.Rectangle((x, y), block_width, block_height,
                                 linewidth=0.5, edgecolor='black', facecolor=color)
        ax.add_patch(rect)

        # Número de clúster al centro
        ax.text(x + block_width / 2, y + block_height / 2,
                str(cluster), ha='center', va='center', fontsize=6, color='black')

    # Dibujar contorno de bloques en df2
    for _, row in df2.iterrows():
        x, y = row['x'], row['y']
        rect = patches.Rectangle((x, y), block_width, block_height,
                                 linewidth=1.5, edgecolor=outline_color, facecolor='none')
        ax.add_patch(rect)

    # Opciones de presentación
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Límites automáticos
    x_all = pd.concat([df1['x'], df2['x']])
    y_all = pd.concat([df1['y'], df2['y']])
    ax.set_xlim(x_all.min() - block_width, x_all.max() + block_width)
    ax.set_ylim(y_all.min() - block_height, y_all.max() + block_height)

    if show_grid:
        for xg in sorted(df1['x'].unique()):
            ax.axvline(xg, color='gray', linewidth=0.2, linestyle='--')
        for yg in sorted(df1['y'].unique()):
            ax.axhline(yg, color='gray', linewidth=0.2, linestyle='--')

    plt.tight_layout()
    plt.show()



def plot_mine_blocks_adv(df,
                         color_by_col,
                         x_col='x',
                         y_col='y',
                         block_width=10,
                         block_height=10,
                         categorical_threshold=25, # Max unique values to treat as categorical
                         cmap='viridis',           # Colormap for continuous data (or discrete if > threshold)
                         nan_color='lightgrey',    # Color for NaN values
                         vmin=None, vmax=None,     # Manual limits for color scale
                         colorbar_label=None,
                         legend_title=None,        # Explicit title for discrete legend
                         title="Vista de Planta Modelo de Bloques",
                         figsize=(12, 9),
                         dpi=100,
                         xlim=None, ylim=None,
                         show_grid=True,
                         show_block_labels=False,
                         fontsize_block_label=6,
                         save=False,
                         save_name='output.png'):
    """
    Genera un gráfico de vista de planta de bloques, adaptándose a datos
    categóricos (con leyenda) o continuos (con barra de color/heatmap).

    Args:
        df (pd.DataFrame): DataFrame con datos de bloques.
        color_by_col (str): Columna para colorear los bloques.
        x_col (str): Columna de coordenadas X (centro). Defaults to 'x'.
        y_col (str): Columna de coordenadas Y (centro). Defaults to 'y'.
        block_width (float): Ancho de bloques. Defaults to 50.
        block_height (float): Alto de bloques. Defaults to 50.
        categorical_threshold (int): Número máximo de valores únicos para tratar
                                     una columna numérica como categórica. Defaults to 25.
        cmap (str): Nombre del colormap de Matplotlib a usar para datos continuos
                    o para datos categóricos con muchos valores. Defaults to 'viridis'.
        nan_color (str): Color para bloques con valor NaN en 'color_by_col'. Defaults to 'lightgrey'.
        vmin (float, optional): Valor mínimo para la escala de color continua. Defaults to None (min de los datos).
        vmax (float, optional): Valor máximo para la escala de color continua. Defaults to None (max de los datos).
        colorbar_label (str, optional): Etiqueta para la barra de color. Defaults to None (usa 'color_by_col').
        legend_title (str, optional): Título para la leyenda discreta. Defaults to None (usa 'color_by_col').
        title (str): Título del gráfico. Defaults to "Vista de Planta Modelo de Bloques".
        figsize (tuple): Tamaño de la figura. Defaults to (12, 9).
        xlim (tuple, optional): Límites eje X (min, max). Defaults to None (automático).
        ylim (tuple, optional): Límites eje Y (min, max). Defaults to None (automático).
        show_grid (bool): Mostrar cuadrícula. Defaults to True.
        show_block_labels (bool): Mostrar valor dentro del bloque. Defaults to False.
        fontsize_block_label (int): Tamaño fuente etiqueta bloque. Defaults to 6.

    Returns:
        None: Muestra el gráfico.
    """
    if df.empty:
        print("El DataFrame está vacío. No se puede generar el gráfico.")
        return

    required_cols = [x_col, y_col, color_by_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Faltan columnas requeridas. Se necesitan: {required_cols}")
        return

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # --- Determinar si es Categórico o Continuo ---
    col_data = df[color_by_col]
    unique_values = col_data.dropna().unique()
    n_unique = len(unique_values)
    is_numeric = pd.api.types.is_numeric_dtype(col_data)

    # Considerar categórico si no es numérico o si es numérico pero con pocos valores únicos
    is_categorical = not is_numeric or (is_numeric and n_unique < categorical_threshold)

    norm = None
    colormap = None
    color_map_discrete = {} # Para modo categórico

    if is_categorical:
        print(f"Tratando '{color_by_col}' como Categórica ({n_unique} valores únicos).")
        # Generar mapa de colores discreto
        # Usar cmap si hay demasiados colores, o tab10/tab20 si son pocos
        if n_unique <= 10:
            colors = plt.cm.get_cmap('tab10', n_unique)
        elif n_unique <= 20:
             colors = plt.cm.get_cmap('tab20', n_unique)
        else:
             colors = plt.cm.get_cmap(cmap, n_unique) # Usa el cmap especificado si hay muchos
        color_map_discrete = {value: colors(i) for i, value in enumerate(unique_values)}

    else: # Tratar como Continuo
        print(f"Tratando '{color_by_col}' como Continua.")
        # Calcular límites si no se proporcionan (ignorando NaN)
        if vmin is None:
            vmin = np.nanmin(col_data)
        if vmax is None:
            vmax = np.nanmax(col_data)
        print(f"  Escala de color: min={vmin:.2f}, max={vmax:.2f}")

        # Crear normalizador y obtener colormap
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        try:
            colormap = plt.cm.get_cmap(cmap)
        except ValueError:
            print(f"Advertencia: Colormap '{cmap}' no encontrado. Usando 'viridis'.")
            cmap = 'viridis'
            colormap = plt.cm.get_cmap(cmap)


    # --- Dibujar Bloques ---
    for index, row in df.iterrows():
        x_center = row[x_col]
        y_center = row[y_col]
        block_value = row[color_by_col]

        x_corner = x_center - block_width / 2
        y_corner = y_center - block_height / 2

        color = nan_color # Color por defecto si es NaN o no mapeado

        if pd.isna(block_value):
            color = nan_color
        elif is_categorical:
            color = color_map_discrete.get(block_value, nan_color) # Usa el color discreto mapeado
        elif norm is not None and colormap is not None: # Continuo
             try:
                 # Aplicar norma y colormap
                 color = colormap(norm(block_value))
             except Exception as e:
                 print(f"Error al obtener color para valor {block_value}: {e}")
                 color = nan_color # Fallback color

        rect = patches.Rectangle(
            (x_corner, y_corner),
            block_width,
            block_height,
            linewidth=0.5,
            edgecolor='grey', # Mantener borde gris claro
            facecolor=color
        )
        ax.add_patch(rect)

        # Añadir etiqueta de texto si se solicita
        if show_block_labels and not pd.isna(block_value):
            # Si es numérico y entero, mostrar sin decimales
            if isinstance(block_value, (int, float)) and block_value == int(block_value):
                label_text = f"{int(block_value)}"
            elif isinstance(block_value, (int, float)):
                label_text = f"{block_value:.1f}"
            else:
                label_text = str(block_value)

            ax.text(x_center, y_center, label_text,
                    ha='center', va='center',
                    fontsize=fontsize_block_label,
                    color='black' if np.mean(mcolors.to_rgb(color)) > 0.5 else 'white')
    # --- Configuración de Ejes ---
    if xlim is None:
        x_min = df[x_col].min() - block_width
        x_max = df[x_col].max() + block_width
        ax.set_xlim(x_min, x_max)
    else:
        ax.set_xlim(xlim)

    if ylim is None:
        y_min = df[y_col].min() - block_height
        y_max = df[y_col].max() + block_height
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(ylim)

    ax.set_xlabel(f"Coordenada {x_col}")
    ax.set_ylabel(f"Coordenada {y_col}")
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)

    if show_grid:
        ax.grid(True, which='major', linestyle='-', color='darkgrey', linewidth=0.7)

    # --- Añadir Leyenda o Barra de Color ---
    if is_categorical:
        # Crear leyenda discreta
        legend_patches = [patches.Patch(color=color, label=str(value))
                          for value, color in color_map_discrete.items()]
        # Añadir patch para NaN si existen y se usó nan_color
        if df[color_by_col].isna().any():
             legend_patches.append(patches.Patch(color=nan_color, label='NaN/Sin Valor'))

        if legend_patches:
            leg_title = legend_title if legend_title else color_by_col
            ax.legend(handles=legend_patches, title=leg_title, bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar para leyenda
        else:
            plt.tight_layout()
    else: # Continuo: Añadir barra de color
        if colormap and norm:
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([]) # Necesario aunque no se use el array directamente aquí
            cbar = fig.colorbar(sm, ax=ax, shrink=0.75, aspect=20) # Ajusta shrink y aspect
            cbar_lab = colorbar_label if colorbar_label else color_by_col
            cbar.set_label(cbar_lab)
            plt.tight_layout() # Ajustar layout general
        else:
             plt.tight_layout()


    if save:
        plt.savefig(save_name, dpi=300)
    else:
        plt.show()


def Calculate_Adjency_Matrix(FaseBanco, BlockWidth, BlockHeight):
    '''
    Crea la matriz de adyacencia de los bloques de la fase-banco respecto a sus coordenadas x e y.
    Devuelve una matriz sparse (CSR).
    '''
    x = FaseBanco['x'].values
    y = FaseBanco['y'].values

    X1 = matlib.repmat(x.reshape(len(x),1), 1, len(x))
    X2 = X1.T
    Y1 = matlib.repmat(y.reshape(len(y),1), 1, len(y))
    Y2 = Y1.T

    D = np.sqrt((1/BlockWidth**2)*(X1 - X2)**2 + (1/BlockHeight**2)*(Y1 - Y2)**2)

    adjency_matrix = (D <= 1) & (D > 0)
    return adjency_matrix.astype(int)


# --- Función de Adyacencia (sin cambios) ---
def construir_matriz_adyacencia(df_banco, distancia=10.0):
    df_banco_reset = df_banco.reset_index(drop=True)
    coords = df_banco_reset[['x', 'y']].values
    if len(coords) < 2: # KDTree necesita al menos 2 puntos
         print("  Advertencia: Menos de 2 bloques en el banco, no se puede calcular adyacencia.")
         N = len(coords)
         return np.zeros((N, N), dtype=bool), df_banco_reset
    try:
        tree = KDTree(coords)
        vecinos = tree.query_pairs(r=distancia)
    except ValueError as e:
        print(f"  Error al construir KDTree o buscar vecinos: {e}")
        return None, df_banco # Devolver None si falla

    N = len(df_banco_reset)
    A = np.zeros((N, N), dtype=bool)

    for i, j in vecinos:
        if 0 <= i < N and 0 <= j < N:
             A[i, j] = A[j, i] = True
        else:
             print(f"Advertencia: Índice inválido {i} o {j} de query_pairs.")

    # print(f"  Matriz de adyacencia construida para banco. {len(vecinos)} pares adyacentes.")
    return A

# --- Nueva Función de Similitud con Penalización Opcional ---
def calcular_matriz_similitud_con_penalizacion(
    df_current_banco,       # DataFrame del banco actual (índice 0..N-1)
    previous_bench_clusters, # Dict: {original_block_id: cluster_label} del banco ANTERIOR
    block_below_map,       # Dict: {current_original_block_id: prev_original_block_id}
    penalty_factor_c=0.001,  # Factor de penalización (e.g., 0.5). 1.0 = sin penalización.
    wd=2, wg=2, r=0.3, epsilon=1e-6
):
    """
    Calcula la matriz de similitud NxN, aplicando una penalización si los
    bloques de abajo están en clusters diferentes del banco anterior.
    """
    N = len(df_current_banco)
    S = np.full((N, N), np.inf, dtype=float)

    # Asegurar que las columnas necesarias existen
    required_cols = ['x', 'y', 'cut', 'tipomineral', 'id']
    if not all(col in df_current_banco.columns for col in required_cols):
        print("Error: Faltan columnas en df_current_banco para calcular similitud.")
        # Devolver matriz vacía o lanzar error? Devolver -inf por ahora.
        return S

    # Extraer columnas relevantes como arrays NumPy para eficiencia
    coords = df_current_banco[['x', 'y']].values
    leys = df_current_banco['cut'].values
    rocks = df_current_banco['tipomineral'].values
    original_ids = df_current_banco['id'].values # Necesitamos el ID original

    # print(f"  Calculando similitud para {N} bloques. Penalización activa: {penalty_factor_c != 1.0}")
    denominador_max = 0
    for i in range(N):
        for j in range(i + 1, N):
            # --- Cálculo de Similitud Base (igual que antes) ---
            coord_i, coord_j = coords[i], coords[j]
            ley_i, ley_j = leys[i], leys[j]
            rock_i, rock_j = rocks[i], rocks[j]
            distancia = np.linalg.norm(coord_i - coord_j)
            dif_ley = max(abs(ley_i - ley_j), 0.01)
            denominador = (np.power(distancia,wd)) * (np.power(dif_ley,wg))
            if denominador > denominador_max: denominador_max = denominador
            s_base = 1.0 / denominador if denominador != 0 else 0
            similitud = s_base
            if rock_i != rock_j:
                similitud *= r
            # --- Aplicar Penalización Vertical ---
            apply_penalty = False
            # Solo aplicar si hay info del banco anterior y el factor no es 1
            if previous_bench_clusters and penalty_factor_c != 1.0:
                original_id_i = original_ids[i]
                original_id_j = original_ids[j]

                # Buscar IDs de los bloques directamente abajo
                id_below_i = block_below_map.get(original_id_i)
                id_below_j = block_below_map.get(original_id_j)

                # Si ambos bloques de abajo existen...
                if id_below_i is not None and id_below_j is not None:
                    cluster_below_i = previous_bench_clusters.get(id_below_i)
                    cluster_below_j = previous_bench_clusters.get(id_below_j)

                    # ... y ambos fueron asignados a un cluster en el banco anterior...
                    if cluster_below_i is not None and cluster_below_j is not None:
                        # ... y pertenecen a clusters DIFERENTES:
                        if cluster_below_i != cluster_below_j:
                            apply_penalty = True
                            # print(f"    Penalizando par ({i},{j}) [IDs {original_id_i},{original_id_j}]. Abajo: Cls {cluster_below_i} != {cluster_below_j}") # Debug

            if apply_penalty:
                similitud *= penalty_factor_c # Aplicar el factor c

            # Asignar a la matriz
            S[i, j] = S[j, i] = similitud
    for i in range(N):
        for j in range(i + 1, N):
            S[i, j] = S[j, i] = S[i,j]*denominador_max
    # print(f"  Matriz de similitud calculada para banco actual.")
    return S

# --- Función de Clustering Jerárquico (Modificada para aceptar S y A precalculadas) ---
def hierarchical_mine_clustering_adaptado(
    df_processed,            # DataFrame ya filtrado y con índice 0..N-1
    initial_adjacency_matrix, # Matriz A precalculada
    similarity_matrix,      # Matriz S precalculada (puede ser penalizada)
    max_cluster_size,
    target_num_clusters
    # Ya no necesita wd, wg, r, distancia_adyacencia aquí si S y A vienen de fuera
):
    """
    Versión del clustering que opera sobre matrices S y A precalculadas.
    """
    n_blocks = len(df_processed)
    if n_blocks == 0: return [] # No hay nada que clusterizar

    # Usar el ID original si existe, si no, el índice procesado 0..N-1
    original_block_ids = df_processed['id'].values if 'id' in df_processed.columns else df_processed.index.values

    clusters = [{'id': i, # ID interno del cluster (0 a N-1)
                 'blocks': {original_block_ids[i]}, # Conjunto de IDs originales
                 'size': 1,
                 'active': True}
                for i in range(n_blocks)]
    num_active_clusters = n_blocks

    # Copiar matrices para no modificar las originales externamente si es necesario
    S = similarity_matrix.copy()
    A = initial_adjacency_matrix.copy().astype(int) # Asegurar que sea numérica para max()

    if target_num_clusters >= n_blocks:
        return [c for c in clusters] # Devolver clusters individuales
    if target_num_clusters < 1: return []


    # --- Bucle de Clustering (Lógica principal igual que antes) ---
    iteration = 0
    while num_active_clusters > target_num_clusters:
        iteration += 1
        # Buscar mejor par válido (i, j) con i < j
        max_similarity = -np.inf
        best_i, best_j = -1, -1
        active_indices = [idx for idx, c in enumerate(clusters) if c['active']]
        if len(active_indices) <= target_num_clusters: # Condición de parada extra
             break

        candidate_pairs = []
        for i in active_indices:
             for j in active_indices:
                 if i > j: continue
                 if not A[i, j]: continue # Adyacencia
                 combined_size = clusters[i]['size'] + clusters[j]['size']
                 if combined_size > max_cluster_size: continue # Tamaño
                 current_similarity = S[i, j]
                 candidate_pairs.append((current_similarity, j, i)) # (sim, j, i) where j > i

        if not candidate_pairs:
             # print("    No más pares válidos encontrados en esta iteración.")
             break

        candidate_pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        max_similarity, best_j, best_i = candidate_pairs[0] # j > i

        merged_cluster_index = best_i
        removed_cluster_index = best_j

        # Fusionar
        clusters[merged_cluster_index]['blocks'].update(clusters[removed_cluster_index]['blocks'])
        clusters[merged_cluster_index]['size'] = len(clusters[merged_cluster_index]['blocks'])
        clusters[removed_cluster_index]['active'] = False
        num_active_clusters -= 1

        # Actualizar S y A (Complete Link para S, Max para A)
        active_indices_now = [idx for idx, c in enumerate(clusters) if c['active']]
        for k in active_indices_now:
            if k == merged_cluster_index: continue
            sim_ik = S[merged_cluster_index, k]
            sim_jk = S[removed_cluster_index, k]
            new_similarity = min(sim_ik, sim_jk) if sim_ik > -np.inf and sim_jk > -np.inf else -np.inf
            S[merged_cluster_index, k] = S[k, merged_cluster_index] = new_similarity

            adj_ik = A[merged_cluster_index, k]
            adj_jk = A[removed_cluster_index, k]
            new_adjacency = max(adj_ik, adj_jk)
            A[merged_cluster_index, k] = A[k, merged_cluster_index] = new_adjacency

        S[removed_cluster_index, :] = S[:, removed_cluster_index] = -np.inf
        A[removed_cluster_index, :] = A[:, removed_cluster_index] = 0

    # --- Preparar Resultado Final ---
    final_clusters_output = []
    for cluster_data in clusters:
        if cluster_data['active']:
            final_clusters_output.append({
                'cluster_internal_id': cluster_data['id'],
                'blocks': cluster_data['blocks'],
                'size': cluster_data['size']
            })
    return final_clusters_output

# --- Función de Clustering Jerárquico (Modificada para aceptar S y A precalculadas) ---
def hierarchical_mine_clustering_adaptado2(
    df_processed,            # DataFrame ya filtrado y con índice 0..N-1
    initial_adjacency_matrix, # Matriz A precalculada
    similarity_matrix,      # Matriz S precalculada (puede ser penalizada)
    max_cluster_size,
    target_num_clusters
    # Ya no necesita wd, wg, r, distancia_adyacencia aquí si S y A vienen de fuera
):
    """
    Versión del clustering que opera sobre matrices S y A precalculadas.
    """
    n_blocks = len(df_processed)
    if n_blocks == 0: return [] # No hay nada que clusterizar

    # Usar el ID original si existe, si no, el índice procesado 0..N-1
    original_block_ids = df_processed['id'].values if 'id' in df_processed.columns else df_processed.index.values

    clusters = [{'id': i, # ID interno del cluster (0 a N-1)
                 'blocks': {original_block_ids[i]}, # Conjunto de IDs originales
                 'size': 1,
                 'active': True}
                for i in range(n_blocks)]
    num_active_clusters = n_blocks

    # Copiar matrices para no modificar las originales externamente si es necesario
    S = similarity_matrix.copy()
    A = initial_adjacency_matrix.copy().astype(int) # Asegurar que sea numérica para max()

    if target_num_clusters >= n_blocks:
        return [c for c in clusters] # Devolver clusters individuales
    if target_num_clusters < 1: return []


    # --- Bucle de Clustering (Lógica principal igual que antes) ---
    iteration = 0
    while num_active_clusters > target_num_clusters:
        iteration += 1
        # Buscar mejor par válido (i, j) con i < j
        max_similarity = -np.inf
        best_i, best_j = -1, -1
        active_indices = [idx for idx, c in enumerate(clusters) if c['active']]
        if len(active_indices) <= target_num_clusters: # Condición de parada extra
             break

        candidate_pairs = []
        for i in active_indices:
             for j in active_indices:
                 if i > j: continue
                 if not A[i, j]: continue # Adyacencia
                 
                 current_similarity = S[i, j]
                 candidate_pairs.append((current_similarity, j, i)) # (sim, j, i) where j > i

        if not candidate_pairs:
             # print("    No más pares válidos encontrados en esta iteración.")
             break

        candidate_pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        max_similarity, best_j, best_i = candidate_pairs[0] # j > i
        if not (clusters[best_i]['size'] + clusters[best_j]['size'] > max_cluster_size): 
            merged_cluster_index = best_i
            removed_cluster_index = best_j

            # Fusionar
            clusters[merged_cluster_index]['blocks'].update(clusters[removed_cluster_index]['blocks'])
            clusters[merged_cluster_index]['size'] = len(clusters[merged_cluster_index]['blocks'])
            clusters[removed_cluster_index]['active'] = False
            num_active_clusters -= 1

            # Actualizar S y A (Complete Link para S, Max para A)
            S[i,:] = np.minimum(S[i,:], S[j,:])
            S[j,:] = 0
            A[i,:] = np.maximum(A[i,:],A[j,:])
            A[j,:] = 0
        else: A[best_i,best_j] = 0 

    # --- Preparar Resultado Final ---
    final_clusters_output = []
    for cluster_data in clusters:
        if cluster_data['active']:
            final_clusters_output.append({
                'cluster_internal_id': cluster_data['id'],
                'blocks': cluster_data['blocks'],
                'size': cluster_data['size']
            })
    return final_clusters_output


# --- Función para encontrar el bloque de abajo (NECESITA IMPLEMENTACIÓN REAL) ---
def find_block_below_mapping(df_curr, df_prev, bench_col='banco'):
    """
    Placeholder MUY BÁSICO para encontrar el bloque debajo.
    ASUME que los bloques están perfectamente alineados en X, Y
    y que los DataFrames tienen las columnas 'x', 'y', 'ID'.
    NECESITA una implementación robusta para tu caso real.
    """
    if df_prev is None or df_prev.empty:
        return {} # No hay banco previo con qué comparar

    print(f"    Generando mapeo de bloques inferiores (Actual: {df_curr[bench_col].iloc[0]} vs Previo: {df_prev[bench_col].iloc[0]})...")
    block_below_map = {}
    # Crear un índice espacial o un lookup eficiente para el banco previo
    # Usar merge es una opción, aunque puede ser lento para DFs grandes
    try:
        # Asegurar que las columnas de merge existen
        if not all(c in df_curr.columns for c in ['x', 'y', 'id']): return {}
        if not all(c in df_prev.columns for c in ['x', 'y', 'id']): return {}

        # Renombrar ID del previo para evitar colisión
        df_prev_lookup = df_prev[['x', 'y', 'id']].rename(columns={'id': 'id_prev'})

        # Hacer merge basado en coordenadas x, y
        merged_df = pd.merge(
            df_curr[['id', 'x', 'y']],
            df_prev_lookup,
            on=['x', 'y'],
            how='left' # Mantener todos los bloques del banco actual
        )

        # Crear el diccionario desde el merge exitoso
        # Iterar sobre filas donde ID_prev no es NaN
        for _, row in merged_df.dropna(subset=['id_prev']).iterrows():
             block_below_map[int(row['id'])] = int(row['id_prev'])

    except Exception as e:
         print(f"    Error durante el merge para encontrar bloques inferiores: {e}")
         # Podrías intentar un método basado en KDTree si el merge falla o es inapropiado

    print(f"    Mapeo de bloques inferiores creado. {len(block_below_map)} bloques mapeados.")
    return block_below_map


# --- NUEVA FUNCIÓN ORQUESTADORA ---
def cluster_mina_por_fase_banco(
    df_mina,                  # DataFrame completo
    fases_a_procesar,         # Lista de fases a procesar
    max_cluster_size,
    target_num_clusters_per_bench, # Asume target fijo por banco
    distancia_adyacencia,
    wd, wg, r,                # Parámetros de similitud base
    penalty_factor_c,         # Penalización vertical (0 < c < 1)
    banco_col='banco'         # Nombre de la columna que identifica el banco/nivel Z

):
    """
    Orquesta el proceso de clustering iterando por fase y luego por banco (desc).
    Aplica penalización vertical en bancos subsiguientes dentro de una fase.
    """
    # Añadir columna para almacenar el ID del cluster final asignado a cada bloque
    df_mina['cluster'] = -1 # -1 indica no asignado inicialmente
    df_mina['final_cluster_label'] = "None" # Etiqueta más descriptiva

    # Diccionario para guardar los resultados detallados por si se necesitan
    all_clusters_info = {}

    # Asegurarse que la columna de banco existe
    if banco_col not in df_mina.columns:
        print(f"Error: La columna de banco '{banco_col}' no existe en el DataFrame.")
        return df_mina, all_clusters_info

    for phase in fases_a_procesar:
        print(f"\n--- Procesando Fase {phase} ---")
        # Filtrar por fase y asegurarse que la columna ID existe
        if 'id' not in df_mina.columns:
             print("Error: Se requiere una columna 'id' única para cada bloque.")
             continue # Saltar esta fase
        df_phase = df_mina[df_mina['fase'] == phase].copy()
        if df_phase.empty:
            print(f"  Fase {phase}: No se encontraron bloques.")
            continue

        # Obtener y ordenar bancos (niveles Z) de mayor a menor
        benches = sorted(df_phase[banco_col].unique(), reverse=True)
        print(f"  Fase {phase}: Bancos a procesar (descendente): {benches}")

        previous_bench_clusters = None # Reiniciar para la nueva fase {original_id: cluster_label}
        df_previous_banco = None       # DataFrame del banco anterior procesado

        for i, current_banco_level in enumerate(benches):
            is_first_bench = (i == 0)
            print(f"\n  -- Procesando Banco {current_banco_level} (Fase {phase}) {'[PRIMERO]' if is_first_bench else ''} --")

            # Filtrar DataFrame para el banco actual
            df_current_banco = df_phase[df_phase[banco_col] == current_banco_level].copy()
            if df_current_banco.empty:
                print(f"   Banco {current_banco_level}: No se encontraron bloques.")
                # ¿Qué hacer con previous_bench_clusters? Mantener el anterior? Resetear?
                # Resetear es más seguro si hay bancos vacíos intermedios.
                previous_bench_clusters = None
                df_previous_banco = None
                continue # Saltar al siguiente banco

            # 1. Calcular Adyacencia (siempre necesaria)
            A_current, df_current_processed = construir_matriz_adyacencia(df_current_banco, distancia_adyacencia)
            if A_current is None:
                print(f"   Banco {current_banco_level}: Error calculando adyacencia. Saltando banco.")
                previous_bench_clusters = None # Resetear por seguridad
                df_previous_banco = None
                continue

            n_current = len(df_current_processed)
            if n_current <= target_num_clusters_per_bench:
                 print(f"   Banco {current_banco_level}: Número de bloques ({n_current}) <= target ({target_num_clusters_per_bench}). Asignando clusters individuales.")
                 # Asignar cada bloque a su propio cluster
                 current_bench_cluster_map = {row['id']: idx+1 for idx, row in df_current_processed.iterrows()}
                 # Simular salida de clustering para consistencia
                 current_final_clusters = [{'cluster_internal_id': idx, 'blocks': {row['id']}, 'size': 1}
                                          for idx, row in df_current_processed.iterrows()]
            else:
                # 2. Calcular Similitud (con o sin penalización)
                S_current = None
                block_below_map = {} # Vacío si es el primer banco

                if is_first_bench:
                    print(f"   Banco {current_banco_level}: Calculando similitud estándar.")
                    S_current = Calculate_Similarity_Matrix(df_current_processed, wd=wd, wg=wg, r=r) # Usa la versión original sin penalización
                else:
                    print(f"   Banco {current_banco_level}: Calculando similitud con penalización c={penalty_factor_c}.")
                    # Encontrar mapeo de bloques inferiores (¡IMPLEMENTACIÓN CRÍTICA!)
                    block_below_map = find_block_below_mapping(df_current_processed, df_previous_banco, bench_col=banco_col)
                    S_current = calcular_matriz_similitud_con_penalizacion(
                        df_current_processed,
                        previous_bench_clusters,
                        block_below_map,
                        penalty_factor_c=penalty_factor_c,
                        wd=wd, wg=wg, r=r
                    )

                # 3. Ejecutar Clustering
                print(f"   Banco {current_banco_level}: Ejecutando clustering para {n_current} bloques...")
                current_final_clusters = hierarchical_mine_clustering_adaptado(
                    df_processed=df_current_processed, # DF con índice 0..N-1
                    initial_adjacency_matrix=A_current,
                    similarity_matrix=S_current,
                    max_cluster_size=max_cluster_size,
                    target_num_clusters=target_num_clusters_per_bench
                )

                # 4. Procesar y almacenar resultados
                if current_final_clusters:
                     print(f"   Banco {current_banco_level}: Clustering completado. {len(current_final_clusters)} clusters encontrados.")
                     current_bench_cluster_map = {}
                     for cluster_idx, cluster_data in enumerate(current_final_clusters):
                         cluster_label = cluster_idx + 1 # Etiqueta 1..K para este banco
                         # Crear una etiqueta global única (opcional pero recomendado)
                         global_cluster_label = f"F{phase}_B{current_banco_level}_C{cluster_label}"
                         for block_id in cluster_data['blocks']:
                             current_bench_cluster_map[block_id] = cluster_label # Mapeo para el siguiente banco
                             # Asignar al DataFrame principal usando la etiqueta global
                             df_mina.loc[df_mina['id'] == block_id, 'cluster'] = cluster_label # Guarda el ID local del banco
                             df_mina.loc[df_mina['id'] == block_id, 'final_cluster_label'] = global_cluster_label # Guarda etiqueta global


                     all_clusters_info[(phase, current_banco_level)] = current_final_clusters # Guardar detalles
                     # Actualizar para la próxima iteración
                     previous_bench_clusters = current_bench_cluster_map
                     df_previous_banco = df_current_processed.copy() # Guardar DF procesado para el mapeo inferior
                else:
                     print(f"   Banco {current_banco_level}: Clustering falló o no produjo clusters.")
                     previous_bench_clusters = None # Resetear si falla
                     df_previous_banco = None

    print("\n--- Proceso de Clustering Secuencial por Fase y Banco Completado ---")
    return df_mina, all_clusters_info


# --- Función de Similitud Original (sin penalización) ---
# (Necesaria para el primer banco de cada fase)
def Calculate_Similarity_Matrix(df_banco, wd=2, wg=2, r=0.2):
    N = len(df_banco)
    S = np.full((N, N), np.inf, dtype=float)
    if not all(col in df_banco.columns for col in ['x', 'y', 'cut', 'tipomineral']): return S
    coords = df_banco[['x', 'y']].values
    leys = df_banco['cut'].values
    rocks = df_banco['tipomineral'].values
    denominador_max = 0
    for i in range(N):
        for j in range(i + 1, N):
            coord_i, coord_j = coords[i], coords[j]
            ley_i, ley_j = leys[i], leys[j]
            rock_i, rock_j = rocks[i], rocks[j]
            distancia = np.linalg.norm(coord_i - coord_j)
            dif_ley = max(abs(ley_i - ley_j), 0.01)
            denominador = ( np.power(distancia,wd)) * ( np.power(dif_ley,wg))
            if denominador > denominador_max: denominador_max = denominador
            similitud = 1.0 / denominador
            if rock_i != rock_j: similitud *= r
            S[i, j] = S[j, i] = similitud
    for i in range(N):
        for j in range(i + 1, N):
            S[i, j] = S[j, i] = S[i,j]*denominador_max

    return S


def plot_phase_clusters_3d_interactive_bancos(df_clustered, phase_to_plot,
                                            cluster_col='final_cluster_label',
                                            x_col='x', y_col='y', z_col='banco',
                                            id_col='id',
                                            marker_size=3):
    """
    Genera un gráfico 3D interactivo de los bloques para una fase dada,
    coloreados por cluster, con botones para mostrar/ocultar bancos,
    manteniendo los límites de los ejes fijos.

    Args:
        # ... (argumentos iguales) ...
    """
    print(f"\nGenerando gráfico 3D interactivo con selección de banco y ejes fijos para Fase {phase_to_plot}...")

    # 1. Filtrar por la fase deseada y limpiar datos
    df_plot = df_clustered[df_clustered['fase'] == phase_to_plot].copy()
    df_plot[cluster_col] = df_plot[cluster_col].astype(str)
    df_plot = df_plot[~df_plot[cluster_col].isin(["None", "-1", "nan"])]

    if df_plot.empty:
        print(f"No se encontraron bloques clusterizados válidos para la Fase {phase_to_plot}.")
        return

    # --- NUEVO: Calcular rangos completos para ejes ---
    # Añadir un pequeño padding para que los puntos no queden justo en el borde
    x_min, x_max = df_plot[x_col].min(), df_plot[x_col].max()
    y_min, y_max = df_plot[y_col].min(), df_plot[y_col].max()
    z_min, z_max = df_plot[z_col].min(), df_plot[z_col].max()

    padding_factor = 0.05 # 5% de padding
    x_pad = (x_max - x_min) * padding_factor if (x_max > x_min) else 1 # Evitar padding 0 si solo hay un valor
    y_pad = (y_max - y_min) * padding_factor if (y_max > y_min) else 1
    z_pad = (z_max - z_min) * padding_factor if (z_max > z_min) else 1

    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]
    z_range = [z_min - z_pad, z_max + z_pad]
    # ---------------------------------------------

    # 2. Obtener listas únicas de clusters y bancos (ordenados)
    unique_clusters = sorted(df_plot[cluster_col].unique())
    unique_benches = sorted(df_plot[z_col].unique(), reverse=True)
    num_clusters = len(unique_clusters)

    # 3. Crear mapa de colores (igual que antes)
    color_sequence = px.colors.qualitative.Alphabet
    if num_clusters > len(color_sequence):
        color_sequence = (color_sequence * (num_clusters // len(color_sequence) + 1))[:num_clusters]
    cluster_color_map = {cluster: color_sequence[i] for i, cluster in enumerate(unique_clusters)}

    # 4. Crear la figura y añadir trazas (igual que antes)
    fig = go.Figure()
    traces_meta = []
    for i, cluster_label in enumerate(unique_clusters):
        # ... (resto del bucle para añadir trazas igual que antes) ...
        df_cluster = df_plot[df_plot[cluster_col] == cluster_label]
        cluster_color = cluster_color_map[cluster_label]
        show_legend_for_cluster = True
        for bench_level in unique_benches:
            df_bench_cluster = df_cluster[df_cluster[z_col] == bench_level]
            if not df_bench_cluster.empty:
                fig.add_trace(go.Scatter3d(
                    x=df_bench_cluster[x_col], y=df_bench_cluster[y_col], z=df_bench_cluster[z_col],
                    mode='markers', marker=dict(size=marker_size, color=cluster_color, opacity=0.8),
                    name=cluster_label, legendgroup=cluster_label,
                    customdata=df_bench_cluster[[id_col, z_col, cluster_col]].values,
                    hovertemplate=(f"<b>ID:</b> %{{customdata[0]}}<br><b>{z_col.capitalize()}:</b> %{{customdata[1]}}<br><b>{cluster_col}:</b> %{{customdata[2]}}<br>X: %{{x:.1f}}<br>Y: %{{y:.1f}}<br>Z: %{{z:.1f}}<extra></extra>"),
                    showlegend=show_legend_for_cluster
                ))
                traces_meta.append({'trace_index': len(fig.data)-1, 'banco': bench_level})
                show_legend_for_cluster = False


    # 5. Crear botones (igual que antes)
    buttons = []
    buttons.append(dict(label="Mostrar Todos", method="update", args=[{"visible": [True] * len(fig.data)}, {"title": f"Clusters 3D - Fase {phase_to_plot} (Todos los Bancos)"}]))
    for bench_level in unique_benches:
        visibility_list = [meta['banco'] == bench_level for meta in traces_meta]
        buttons.append(dict(label=f"Banco {bench_level}", method="update", args=[{"visible": visibility_list}, {"title": f"Clusters 3D - Fase {phase_to_plot} (Banco {bench_level})"}]))

    # 6. Actualizar layout: AÑADIR RANGOS DE EJES FIJOS
    fig.update_layout(
        uirevision='keep_cam_view', # Mantener vista de cámara
        updatemenus=[dict(
            active=0, buttons=buttons, direction="down",
            pad={"r": 10, "t": 10}, showactive=True,
            x=0.1, xanchor="left", y=1.1, yanchor="top"
        )],
        title=f"Clusters 3D - Fase {phase_to_plot} (Todos los Bancos)",
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title=x_col.upper(),
            yaxis_title=y_col.upper(),
            zaxis_title=z_col.upper(),
            aspectmode='data',
            # --- FIJAR RANGOS DE EJES ---
            xaxis=dict(range=x_range, autorange=False), # Añadir autorange=False explícitamente
            yaxis=dict(range=y_range, autorange=False),
            zaxis=dict(range=z_range, autorange=False)
            # ---------------------------
        ),
        legend_title_text=f"{cluster_col} (Click para mostrar/ocultar)"
    )

    # 7. Mostrar gráfico
    fig.show()

def Calculate_Arcs(df_sup, df_inf, BlockWidth=10, BlockHeight=10, arcs=defaultdict(list)):
    '''
    Asume que el df_inf corresponde a la fase banco inferior a df_sup, calcula los arcos de precedencia verticales
    '''
    A_clusters = Calculate_Vertical_Adyacency_Matrix(df_sup, df_inf, BlockWidth=BlockWidth, BlockHeight=BlockHeight)
    f_inf = df_inf['fase'][0]
    b_inf = df_inf['banco'][0]
    f_sup = df_sup['fase'][0]
    b_sup = df_sup['banco'][0]
    for i in range(A_clusters.shape[1]):  # i: índice de clusters inferiores
        for j in range(A_clusters.shape[0]):  # j: índice de clusters superiores
            if A_clusters.T[i][j] == 1:
                arcs[(int(f_inf), int(b_inf), i + 1)].append((int(f_sup), int(b_sup), j + 1))

    return arcs

def Precedencia_Fase_Banco(df):
    df = df.copy()
    # Obtener los valores únicos ordenados de 'z'
    sorted_unique_z = np.sort(df['z'].unique())

    # Crear una nueva columna 'nivel' basada en la posición de 'z' en el array ordenado
    df['nivel'] = df['z'].apply(lambda x: np.where(sorted_unique_z == x)[0][0])
    info_fb = {}
    fases_a_procesar = np.sort(df['fase'].unique())
    for f in fases_a_procesar:
        benches = np.sort(df[df['fase']==f]['banco'].unique())
        for b in benches:
            df_inf = df[(df['fase']==f)&(df['banco']==b)]
            df_inf.reset_index(drop=True, inplace=True)
            z_level = df_inf['nivel'][0]
            min_x = df_inf['x'].min()
            max_x = df_inf['x'].max()
            min_y = df_inf['y'].min()
            max_y = df_inf['y'].max()
            info_fb[(int(f),int(b))] = (z_level, min_x, max_x, min_y,max_y)

    # Lista para guardar los pares (f_i, b_i) y (f_j, b_j) que cumplen la condición
    resultados = []

    # Iterar sobre todos los pares (f1, b1) y (f2, b2)
    for (f1, b1), (z1, min_x1, max_x1, min_y1, max_y1) in info_fb.items():
        for (f2, b2), (z2, min_x2, max_x2, min_y2, max_y2) in info_fb.items():
            # Condición 1: z1 == z2+1 (es necesario que )
            if z1 == z2+1:
                # Condición 2: Los cuadros delimitados por (x, y) se intersectan
                if (min_x1 <= max_x2 and max_x1 >= min_x2) and (min_y1 <= max_y2 and max_y1 >= min_y2):
                    resultados.append(((f1, b1), (f2, b2)))
    return resultados
def Global_Vertical_Arc_Calculation(df):
    resultados = Precedencia_Fase_Banco(df)
    arcs = defaultdict(list)
    # Iterar
    for par in resultados:
        ((f_sup,b_sup),(f_inf,b_inf)) = par
        df_inf = df[(df['fase']==f_inf)&(df['banco']==b_inf)].copy()
        df_inf.reset_index(drop=True, inplace=True)
        df_sup = df[(df['fase']==f_sup)&(df['banco']==b_sup)].copy()
        df_sup.reset_index(drop=True, inplace=True)
        arcs = Calculate_Arcs(df_sup,df_inf,arcs=arcs)
    return arcs



def Coefficient_Variation(FaseBanco, col='cut'):
    ID_Clusters = np.sort(FaseBanco['cluster'].dropna().unique())
    num_clusters = len(ID_Clusters)
    if num_clusters == 0:
        return np.nan, []

    sum_cv = 0
    CV_distribution = []
    for id in ID_Clusters:
        Cluster = FaseBanco.loc[FaseBanco['cluster'] == id]
        if len(Cluster) == 1:
            cv = 0  # Definimos CV=0 si solo hay un bloque
        else:
            std = Cluster[col].std()
            mean = Cluster[col].mean()
            if pd.isna(std) or pd.isna(mean) or mean == 0:
                cv = 0
            else:
                cv = std / mean
        
        sum_cv += cv
        CV_distribution.append(cv)
        
    CV = sum_cv / num_clusters
    return CV, CV_distribution



def Calculate_Vertical_Adyacency_Matrix(df_sup, df_inf, BlockWidth=10, BlockHeight=10, arcs=defaultdict(list), cluster_col='cluster'):
    '''
    Asume que el df_inf corresponde a la fase banco inferior a df_sup, calcula los arcos de precedencia verticales
    '''
    x1 = df_sup['x'].values
    y1 = df_sup['y'].values
    x2 = df_inf['x'].values
    y2 = df_inf['y'].values

    X1 = matlib.repmat(x1.reshape(len(x1), 1), 1, len(x2))  # (n_sup, n_inf)
    X2 = matlib.repmat(x2.reshape(1, len(x2)), len(x1), 1)  # (n_sup, n_inf)

    Y1 = matlib.repmat(y1.reshape(len(y1), 1), 1, len(y2))  # (n_sup, n_inf)
    Y2 = matlib.repmat(y2.reshape(1, len(y2)), len(y1), 1)  # (n_sup, n_inf)


    Dx = np.abs((1/BlockWidth)*(X1 - X2))
    Dy = np.abs((1/BlockHeight)*(Y1 - Y2))
    adjency_matrix = (np.sqrt(Dx**2 + Dy**2)<=1) # Distancia euclidiana normalizada
    # adjency_matrix = (Dx <= 1) & (Dy <= 1) # Mide la adyacencia por direcciones X e Y, usando 1, dos bloques son adyacentes si están a una distancia de 1 bloque o menos

    # adjency_matrix = (D <= BlockWidth) & (D > 0)
    adjency_matrix = sp.sparse.csr_matrix(adjency_matrix).astype(int)

    # Obtener clusters únicos y sus índices
    clusters_sup = df_sup[cluster_col].unique()
    clusters_inf = df_inf[cluster_col].unique()

    cluster_sup_to_idx = {c: i for i, c in enumerate(clusters_sup)}
    cluster_inf_to_idx = {c: i for i, c in enumerate(clusters_inf)}

    n_sup = len(clusters_sup)
    n_inf = len(clusters_inf)

    # Inicializar matriz de adyacencia entre clusters
    A_clusters = np.zeros((n_sup, n_inf), dtype=int)

    # Recorrer los pares adyacentes entre bloques
    rows, cols = adjency_matrix.nonzero()  # A_block: sup x inf
    for i_sup, i_inf in zip(rows, cols):
        c_sup = df_sup.iloc[i_sup][cluster_col]
        c_inf = df_inf.iloc[i_inf][cluster_col]

        idx_sup = cluster_sup_to_idx[c_sup]
        idx_inf = cluster_inf_to_idx[c_inf]

        A_clusters[idx_sup, idx_inf] = 1


    return A_clusters

'''
Metodo de clustering de toda una mina, con opciones para armar la matriz de similaridad, usar shape refinement, etc.
'''
def mine_clustering(df, BlockWidth = 10, BlockHeight = 10, params=dict()):
    params.setdefault('penalizacion_c', 0.5)
    c_coef = params['penalizacion_c']
    df['cluster'] = -1
    sorted_unique_z = np.sort(df['z'].unique()) #Orden ascendente de z
    
    res = Precedencia_Fase_Banco(df)
    precedence_tree = defaultdict(list)
    for node, predecessor in res:
        precedence_tree[node].append(predecessor)

    for z in sorted_unique_z:
        print(f"---Procesando Z={z}")
        sliced_df = df[df['z']==z]
        lista_fases_bancos = list((sliced_df[['fase', 'banco']].drop_duplicates()).itertuples(index=False, name=None)) # Lista de tuplas únicas de (fase, banco) en z
        for (f,b) in lista_fases_bancos:
            print(f"========Procesando Fase {f} - Banco {b}========")
            slice_fase_banco = sliced_df[(sliced_df['fase']==f) & (sliced_df['banco']==b)]
            A = mt.Calculate_Adjency_Matrix(slice_fase_banco, BlockWidthX=BlockWidth, BlockWidthY=BlockHeight)
            S = mt.Calculate_Similarity_Matrix(slice_fase_banco, params=params)
            if (f,b) not in precedence_tree: # Si no hay precedencias verticales
                print(f"---No hay precedencias verticales")
                fase_banco_clusterizado = mt.Clustering_Tabesh(slice_fase_banco, A, S, params=params)[0]
                df.loc[fase_banco_clusterizado.index, 'cluster'] = fase_banco_clusterizado['cluster'] # Asignar clusters al df original

            if (f,b) in precedence_tree: # Si hay precedencias verticales
                print(f"---Precedencias Detectadas")
                print(precedence_tree[(f,b)])
                dfs = [] # Lista para almacenar los dataframes precedentes
                for (f_,b_) in precedence_tree[(f,b)]:
                    dfs.append( (df[(df['fase']==f_)&(df['banco']==b_)]).copy() )
                offset=0
                for df_ in dfs:
                    if not df_.empty:
                        # Ajustamos los clusters
                        df_['cluster'] = df_['cluster'] + offset
                        # Actualizamos el offset: suma el máximo cluster de este df
                        offset = df_['cluster'].max()
                
                df_inf = pd.concat(dfs, ignore_index=True) #dataframe inferior a f,b
                df_sup = slice_fase_banco.copy()
                df_sup.reset_index(drop=True, inplace=True)
                df_sup['cluster'] = df_sup['id']
                A_vertical = Calculate_Vertical_Adyacency_Matrix(df_sup,df_inf, BlockWidth=BlockWidth/4, BlockHeight=BlockHeight/4) # Tiene que ser estricto pues es para calcular C
                # Ahora calculamos la matriz C
                C = A_vertical@A_vertical.T
                C = (1-C)*c_coef+C
                #Ahora la similaridad
                S = S*C
                print(f"---Clustering de Fase {f} - Banco {b}")
                fase_banco_clusterizado = mt.Clustering_Tabesh(slice_fase_banco, A, S, params=params)[0]
                df.loc[fase_banco_clusterizado.index, 'cluster'] = fase_banco_clusterizado['cluster'] # Asignar clusters al df original
    return df
# Función auxiliar para empaquetar los argumentos
def clustering_wrapper(args):
    mina_df, BlockWidth, BlockHeight, params = args
    return mine_clustering(mina_df, BlockWidth, BlockHeight, params)

def mine_clustering_parallel(df, BlockWidth = 10, BlockHeight = 10, params=dict()):
    # Separamos por fase
    fases = np.sort(df['fase'].unique())
    lista_minas = [df[df['fase'] == f].copy() for f in fases]

    # Empaquetamos argumentos para cada fase
    args_list = [(mina, BlockWidth, BlockHeight, params) for mina in lista_minas]

    # Procesamos en paralelo
    with Pool(cpu_count()) as pool:
        resultados = pool.map(clustering_wrapper, args_list)

    # Opcional: juntar todo en un solo DataFrame
    resultado_final = pd.concat(resultados, ignore_index=True)
    return resultado_final

