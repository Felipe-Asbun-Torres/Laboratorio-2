import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors 
import time

def plot_fase_banco(FaseBanco, column_hue='cut', cmap='plasma', show_block_label=True, show_grid=True, xsize = 10, ysize = 10):
    fig, ax = plt.subplots(figsize=(xsize, ysize), dpi=100)
    norm = None
    colormap = None
    color_map_discrete = {}
    variables_continuas = FaseBanco.select_dtypes(include='float64').columns.tolist()
    
    fase = FaseBanco['fase'][0]
    z = FaseBanco['z'][0]

    col_data = FaseBanco[column_hue]
    if column_hue in variables_continuas:
        is_continuous = True
    else:
        is_continuous = False
    
    if is_continuous:
        vmin = np.min(col_data)
        vmax = np.max(col_data)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.get_cmap(cmap)
    else:
        if len(col_data.unique())<=20:
            colors = plt.get_cmap('tab20', len(col_data.unique()))
            color_map_discrete = {val: colors(i) for i, val in enumerate(col_data.unique())}
        else:
            colors = plt.get_cmap(cmap, len(col_data.unique()))
            color_map_discrete = {val: colors(i) for i, val in enumerate(col_data.unique())}

    for i, row in FaseBanco.iterrows():
        x_center = row['x']
        y_center = row['y']
        block_value = row[column_hue]
        block_width = 10
        block_height = 10

        x_corner = x_center - block_width / 2
        y_corner = y_center - block_height / 2

        if is_continuous:
            color = colormap(norm(block_value))
        else:
            color = color_map_discrete.get(block_value, 'gray')
        
        rect = patches.Rectangle((x_corner, y_corner), block_width, block_height, linewidth=0.5, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        if show_block_label:
            if is_continuous:
                block_value = np.trunc(block_value*10)/10
            else:
                block_value = int(block_value)
            ax.text(x_center, y_center, str(block_value), ha='center', va='center', fontsize=8, color='black')
    
    if is_continuous:
        x_min = FaseBanco['x'].min() - block_width
        x_max = FaseBanco['x'].max() + block_width
        ax.set_xlim(x_min, x_max)
    else:
        x_min = FaseBanco['x'].min() - block_width
        x_max = FaseBanco['x'].max() + 5*block_width
        ax.set_xlim(x_min, x_max)
    y_min = FaseBanco['y'].min() - block_height
    y_max = FaseBanco['y'].max() + block_height
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Fase {fase} - Banco (Z={z}) - {column_hue}')
    ax.grid(show_grid, color='gray', linestyle='--', linewidth=0.5)

    # ax.plot(fase_banco['x'][0], fase_banco['y'][0], 'ro', markersize=10,)

    if is_continuous:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label(column_hue, rotation=270, labelpad=15)
    else:
        legend_patches = [patches.Patch(color=color, label=str(value)) for value, color in color_map_discrete.items()]
        ax.legend(handles=legend_patches, title=column_hue, loc='upper right', fontsize=8, title_fontsize=10)
    plt.tight_layout()
    plt.show()


def Calculate_Adjency_Matrix(fasebanco):
    '''
    Crea la matriz de adyacencia de los bloques de la fase-banco respecto a sus coordenadas x e y.
    Devuelve una matriz sparse (CSR).
    '''
    x = fasebanco['x'].values
    y = fasebanco['y'].values

    X1 = np.matlib.repmat(x.reshape(len(x),1), 1, len(x))
    X2 = X1.T

    Y1 = np.matlib.repmat(y.reshape(len(y),1), 1, len(y))
    Y2 = Y1.T

    D = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)

    adjency_matrix = (D <= 10) & (D > 0)
    adjency_matrix = sp.sparse.csr_matrix(adjency_matrix).astype(int)
    return adjency_matrix


def Calculate_Similarity_Matrix(
        fasebanco, 
        distancia = True, 
        ley = False, 
        destino = True, 
        directional_mining = False,
        tipo_roca = True,
        peso_distancia = 1, 
        peso_ley = 1, 
        penalizacion_destino = 0.9, 
        penalizacion_roca = 0.9,
        peso_directional_mining=1,
        tol_ley = 0.0001,
        tol_dir = 0.0001,
        P_inicio = (-1,-1),
        P_final = (1,1)
        ):
    '''
    Calcula la similaridad entre los bloques de la fase-banco, de acuerdo a distancia, ley o destino.
    No es recomendable que todos los criterios de similaridad sean True y, a su vez, al menos uno de ellos debe ser True.as_integer_ratio
    Devuelve una matriz densa.
    '''
    if not(distancia or ley or destino or directional_mining):
        raise ValueError("Al menos distancia, ley, o destino deben ser True.")

    n = len(fasebanco)

    similarity_matrix = np.zeros((n, n), dtype=float)
    if distancia:
        x = fasebanco['x'].values
        y = fasebanco['y'].values

        X1 = np.matlib.repmat(x.reshape(len(x),1), 1, len(x))
        X2 = X1.T

        Y1 = np.matlib.repmat(y.reshape(len(y),1), 1, len(y))
        Y2 = Y1.T

        D = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)

        ND = D / np.max(D)
    else:
        ND = 1

    if ley:
        g = fasebanco['cut'].values
        G1 = np.matlib.repmat(g.reshape(len(g),1), 1, len(g))
        G2 = G1.T

        G = np.maximum(np.abs(G1 - G2), tol_ley)

        NG = G / np.max(G)
    else:
        NG = 1

    if destino:
        t = fasebanco['destino'].values
        T1 = np.matlib.repmat(t.reshape(len(t),1), 1, len(t))
        T2 = T1.T
        T = T1 != T2

        T = np.ones((n,n)) - (1-penalizacion_destino)*T*np.ones((n,n))
    else:
        T = 1

    if tipo_roca:
        rock = fasebanco['tipomineral'].values
        R1 = np.matlib.repmat(rock.reshape(len(rock),1), 1, len(rock))
        R2 = R1.T
        R = R1 != R2

        R = np.ones((n,n)) - (1-penalizacion_roca)*R*np.ones((n,n))

    if directional_mining:
        x = fasebanco['x'].values
        y = fasebanco['y'].values
        X1 = np.matlib.repmat(x.reshape(len(x),1), 1, len(x))
        # X2 = X1.T
        Y1 = np.matlib.repmat(y.reshape(len(y),1), 1, len(y))
        # Y2 = Y1.T

        M1 = (X1 - P_inicio[0])**2 + (Y1 - P_inicio[1])**2
        M2 = (X1 - P_final[0])**2 + (Y1 - P_final[1])**2
        Mi = np.multiply( np.sign( M1 - M2 ), np.sqrt( np.abs(M1 - M2 )) )
        DM = np.maximum(np.abs(Mi - Mi.T), tol_dir)
    else:
        DM = 1
    numerator = np.multiply(R,T)
    denominator = np.multiply(ND**peso_distancia, NG**peso_ley)
    denominator = np.multiply(denominator, DM**peso_directional_mining)
    similarity_matrix = np.divide(numerator,  denominator, where=np.ones((n,n)) - np.diag(np.ones(n)) ==1)
    return similarity_matrix

def Hadamard_Product_Sparse(A, B):
    '''
    Calcula el producto de Hadamard (componente a componente) entre una matriz sparse (A) y una densa (B).
    Devuelve una matriz sparse (CSR).
    '''
    rows, cols = A.nonzero()
    data = []
    res_row = []
    res_col = []

    for i in range(len(rows)):
        row = rows[i]
        col = cols[i]
        data.append(A[row, col] * B[row, col])
        res_row.append(row)
        res_col.append(col)
    resultado_sparse = sp.sparse.csr_matrix((data, (res_row, res_col)), shape=A.shape)
    return resultado_sparse

# def Find_Most_Similar_Adjacent_Clusters_Hadamard(AdjencyMatrix, SimilarityMatrix):
#     '''
#     Encuentra los dos clusters adyacentes más similares entre sí, de acuerdo a la matriz de similaridad y la matriz de adyacencia.
#     Devuelve una lista con los índices de los clusters más similares. Si no encuentra pares similares, devuelve None.
#     Depende de la función Hadamard_Product_Sparse.
#     '''
#     Sim_Matrix = Hadamard_Product_Sparse(AdjencyMatrix, SimilarityMatrix)
#     Sim_Matrix.eliminate_zeros()
#     if Sim_Matrix.nnz == 0:
#         return None
#     most_similar_clusters = []
#     max_similarity = -1

#     rows, cols = Sim_Matrix.nonzero()
#     for i in range(len(rows)):
#         row = rows[i]
#         col = cols[i]
#         if Sim_Matrix[row, col] > max_similarity:
#             max_similarity = Sim_Matrix[row, col]
#             most_similar_clusters = [row, col]
#     return most_similar_clusters

def Find_Most_Similar_Adjacent_Clusters_Hadamard(AdjencyMatrix, SimilarityMatrix):
    '''
    Encuentra los dos clusters adyacentes más similares entre sí, de acuerdo a la matriz de similaridad y la matriz de adyacencia.
    Devuelve una lista con los índices de los clusters más similares. Si no encuentra pares similares, devuelve None.
    Depende de la función Hadamard_Product_Sparse.
    '''
    Sim_Matrix = Hadamard_Product_Sparse(AdjencyMatrix, SimilarityMatrix)
    Sim_Matrix.eliminate_zeros()
    if Sim_Matrix.nnz == 0:
        return None
    Sim_Matrix = Sim_Matrix.tocoo()
    index_max_similarity = np.argmax(Sim_Matrix.data)
    row_max = Sim_Matrix.row[index_max_similarity]
    col_max = Sim_Matrix.col[index_max_similarity]
    
    if row_max < col_max:
        i = row_max
        j = col_max
    else:
        i = col_max
        j = row_max
    return [i, j]

def Find_Corner_Blocks(FaseBanco, AdjencyMatrix):
    '''
    Encuentra los bloques esquinas de la fase-banco. Se agregaron un par de criterios adicionales respecto a la definición de Tabesh, para bloques con sólo 1 o 2 vecinos.
    Devuelve un diccionario cuyas llaves son los bloques esquina y cuyos valores son los clusters con los que es vecino.
    '''
    n = len(FaseBanco)
    corner_blocks = {}
    rows, cols = AdjencyMatrix.nonzero()
    for i in range(n):
        Mismo_Cluster = 0
        Distinto_Cluster = 0
        Clusters_Vecinos = []
        i_is_row = np.where(rows == i)[0]
        for j in i_is_row:
            if FaseBanco.iloc[i]['cluster'] == FaseBanco.iloc[cols[j]]['cluster']:
                Mismo_Cluster += 1
            else:
                Distinto_Cluster += 1
                Clusters_Vecinos.append(FaseBanco.iloc[cols[j]]['cluster'].astype(int))

        if Distinto_Cluster == 2 and Mismo_Cluster == 0 and (len(np.unique(Clusters_Vecinos)) == 2): ## 2 vecinos
            corner_blocks.update({i: np.sort(Clusters_Vecinos)})
        if (Mismo_Cluster <= 1 and Distinto_Cluster > 1 and (len(np.unique(Clusters_Vecinos)) < len(Clusters_Vecinos))) or (Mismo_Cluster == 0 and Distinto_Cluster == 1):  ## 1 o 3 vecinos
            corner_blocks.update({i: np.sort(Clusters_Vecinos)})
    return corner_blocks

def Find_Corner_Blocks_Tabesh(FaseBanco, AdjencyMatrix):
    '''
    Encuentra los bloques esquinas de la fase-banco. Utiliza el criterio de Tabesh (2013) para definir los bloques esquinas.
    Devuelve un diccionario cuyas llaves son los bloques esquina y cuyos valores son los clusters con los que es vecino.
    '''
    n = len(FaseBanco)
    corner_blocks = {}
    rows, cols = AdjencyMatrix.nonzero()
    for i in range(n):
        Mismo_Cluster = 0
        Distinto_Cluster = 0
        Clusters_Vecinos = []
        i_is_row = np.where(rows == i)[0]
        for j in i_is_row:
            if FaseBanco.iloc[i]['cluster'] == FaseBanco.iloc[cols[j]]['cluster']:
                Mismo_Cluster += 1
            else:
                Distinto_Cluster += 1
                Clusters_Vecinos.append(FaseBanco.iloc[cols[j]]['cluster'].astype(int))
        
        if (Mismo_Cluster <= 1 and Distinto_Cluster > 1 and (len(np.unique(Clusters_Vecinos)) < len(Clusters_Vecinos))):
            corner_blocks.update({i: np.sort(Clusters_Vecinos)})
    return corner_blocks

def Clustering(
        FaseBanco, 
        Average_Desired_Length_Cluster = 30, 
        Max_Cluster_Length = 35, 
        Min_Cluster_Length = 10, 
        PostProcessing = True, 
        Iterations_PostProcessing = 5, 
        Tabesh = True,
        distancia = True,
        ley = False,
        destino = True,
        directional_mining = False,
        peso_distancia = 2,
        peso_ley = 0.5,
        penalizacion_destino = 0.8,
        peso_directional_mining=1,
        tol_ley = 0.0001,
        tol_dir = 0.0001,
        P_inicio = (-1,-1),
        P_final = (1,1),
        Debug = False
        ):
        
    '''
    Realiza un clustering jerárquico y agregativo de la fase-banco.
    Average_Desired_Length_Cluster, Max_Cluster_Length y Min_Cluster_Length son restricciones blandas del tamaño de los clusters. Es decir, no siempre se van a satisfacer.
    PostProcessing es una heurística para mejorar la forma de los clusters y para eliminar clusters pequeños.
    Depende de las funciones Calculate_Adjency_Matrix, Calculate_Similarity_Matrix, Find_Most_Similar_Adjacent_Clusters_Hadamard y Find_Corner_Blocks.
    Devuelve un DataFrame de la fase-banco con la columna 'cluster' que indica el cluster al que pertenece cada bloque, y las matrices de adyacencia y de similaridad originales.
    '''
    fase_banco = FaseBanco.copy()
    
    
    adj_matrix_sparse = Calculate_Adjency_Matrix(fase_banco)
    adj_matrix_copy = adj_matrix_sparse.copy()
    sim_matrix = Calculate_Similarity_Matrix(fase_banco, distancia=distancia, ley=ley, destino=destino, peso_distancia=peso_distancia, peso_ley=peso_ley, penalizacion_destino=penalizacion_destino, tol_ley=tol_ley, directional_mining=directional_mining, tol_dir=tol_dir, P_inicio=P_inicio, P_final=P_final, peso_directional_mining=peso_directional_mining)
    sim_matrix_copy = sim_matrix.copy()

    print(f'Nonzero entries of Adjency Matrix: {adj_matrix_sparse.nnz}')
    
    N_Clusters = len(fase_banco)
    n = N_Clusters
    Max_N_Clusters = len(fase_banco) // Average_Desired_Length_Cluster

    fase_banco['cluster'] = np.arange(N_Clusters).astype(int)
    Clusters_Eliminados = 0
    t1 = time.time()
    tries = 0
    while N_Clusters > Max_N_Clusters:
        t1_fmsac = time.time()
        C = Find_Most_Similar_Adjacent_Clusters_Hadamard(adj_matrix_sparse, sim_matrix)
        t2_fmsac = time.time()
        tries += 1
        if Debug:
            print(f'Try: {tries}, time: {t2_fmsac-t1_fmsac}')
        if C is None:
            break
        (i,j) = C
        cluster_i = fase_banco[fase_banco['cluster'] == fase_banco.iloc[i]['cluster']]
        cluster_j = fase_banco[fase_banco['cluster'] == fase_banco.iloc[j]['cluster']]

        if len(cluster_i) + len(cluster_j) <= Max_Cluster_Length:
            sim_matrix[i,:] = np.min([sim_matrix[i,:], sim_matrix[j,:]], axis=0)
            sim_matrix[:,i] = np.min([sim_matrix[:,i], sim_matrix[:,j]], axis=0)
            sim_matrix[j,:] = np.zeros(n)
            sim_matrix[:,j] = np.zeros(n)
            
            adj_matrix_sparse = adj_matrix_sparse.tolil()
            adj_matrix_sparse[i,:] = adj_matrix_sparse[i,:].maximum(adj_matrix_sparse[j,:])
            adj_matrix_sparse[:,i] = adj_matrix_sparse[:,i].maximum(adj_matrix_sparse[:,j])

            adj_matrix_sparse[j,:] = np.zeros(n)
            adj_matrix_sparse[:,j] = np.zeros(n)
            # new_row_sparse = sp.sparse.csr_matrix(adj_matrix_sparse[i,:].maximum(adj_matrix_sparse[j,:]), shape=(1,n))
            # before = adj_matrix_sparse[:i]
            # after = adj_matrix_sparse[i+1:]
            # adj_matrix_sparse = sp.sparse.vstack([before, new_row_sparse, after])
            
            # new_col_sparse = sp.sparse.csr_matrix(adj_matrix_sparse[:,i].maximum(adj_matrix_sparse[:,j]), shape=(n,1))
            # before = adj_matrix_sparse[:i]
            # after = adj_matrix_sparse[i+1:]
            # adj_matrix_sparse = sp.sparse.vstack([before, new_row_sparse, after])

            # new_row_sparse = sp.sparse.csr_matrix(np.zeros(n), shape=(1,n))
            # before = adj_matrix_sparse[:j]
            # after = adj_matrix_sparse[j+1:]
            # adj_matrix_sparse = sp.sparse.vstack([before, new_row_sparse, after])
            
            adj_matrix_sparse = adj_matrix_sparse.tocsr()
            adj_matrix_sparse.eliminate_zeros()

            fase_banco.loc[fase_banco['cluster'] == fase_banco.iloc[j]['cluster'], 'cluster'] = fase_banco.iloc[i]['cluster'].astype(int)
            N_Clusters -= 1
            Clusters_Eliminados += 1
        else:
            adj_matrix_sparse[i,j] = 0
            sim_matrix[i,j] = 0
            adj_matrix_sparse.eliminate_zeros()
    t2 = time.time()
    tiempo_agregacion = t2 - t1
    N_Clusters = len(fase_banco['cluster'].unique())
    fase_banco['cluster'] = fase_banco['cluster'].map(lambda x: np.array(range(1,N_Clusters+1))[np.where(fase_banco['cluster'].unique() == x)[0][0]] if x in fase_banco['cluster'].unique() else x)
    print(f"========PreProcessing Results========")
    print(f"Clusters objetivo: {Max_N_Clusters}")
    print(f"Clusters eliminados: {Clusters_Eliminados}")
    print(f"Total de clusters: {N_Clusters}")
    print(f'Tiempo: {tiempo_agregacion}')

    if PostProcessing:
        ID_Small_Clusters = fase_banco['cluster'].value_counts().loc[fase_banco['cluster'].value_counts() < Min_Cluster_Length].index.tolist()
        max_i_cluster = fase_banco['cluster'].max() + 1
        t1 = time.time()        
        for i in ID_Small_Clusters:
            blocks = fase_banco.loc[fase_banco['cluster'] == i, 'cluster'].index
            for j in blocks:
                fase_banco.loc[j, 'cluster'] = max_i_cluster
                max_i_cluster += 1

        for i in range(Iterations_PostProcessing):
            if Tabesh:
                Corner_Blocks = Find_Corner_Blocks_Tabesh(fase_banco, adj_matrix_copy)
            else:
                Corner_Blocks = Find_Corner_Blocks(fase_banco, adj_matrix_copy)
            
            if len(Corner_Blocks) == 0:
                break
            for i in Corner_Blocks.keys():
                if len(Corner_Blocks[i]) == 2:
                    Cluster_to_insert = Corner_Blocks[i][int(np.round(np.random.rand()))]
                elif len(Corner_Blocks[i]) == 1:
                    choose = [Corner_Blocks[i][0], fase_banco.iloc[i]['cluster']]
                    Cluster_to_insert = choose[int(np.round(np.random.rand()))]
                else:
                    Cluster_to_insert = np.unique_counts(Corner_Blocks[i]).values[np.unique_counts(Corner_Blocks[i]).counts.argmax()]
                fase_banco.loc[i, 'cluster'] = Cluster_to_insert
        t2 = time.time()
        tiempo_postprocesado = t2-t1
        N_Clusters = len(fase_banco['cluster'].unique())
        fase_banco['cluster'] = fase_banco['cluster'].map(lambda x: np.array(range(1,N_Clusters+1))[np.where(fase_banco['cluster'].unique() == x)[0][0]] if x in fase_banco['cluster'].unique() else x)
        
        print(f"========PostProcessing Results========")
        print(f"Total de clusters: {N_Clusters}")
        print(f'Tiempo: {tiempo_postprocesado}')

    if PostProcessing:
        execution_time = tiempo_agregacion + tiempo_postprocesado
    else:
        execution_time = tiempo_agregacion
    return [fase_banco, N_Clusters, adj_matrix_copy, sim_matrix_copy, execution_time, adj_matrix_sparse]


def ClusteringSalman(
        FaseBanco, 
        Average_Desired_Length_Cluster = 30, 
        Max_Cluster_Length = 35, 
        Min_Cluster_Length = 10, 
        PostProcessing = True, 
        Iterations_PostProcessing = 5, 
        Tabesh = True,
        distancia = True,
        ley = False,
        destino = True,
        directional_mining = False,
        peso_distancia = 2,
        peso_ley = 0.5,
        penalizacion_destino = 0.8,
        peso_directional_mining=1,
        tol_ley = 0.0001,
        tol_dir = 0.0001,
        P_inicio = (-1,-1),
        P_final = (1,1),
        Debug = False
        ):
        
    '''
    Realiza un clustering jerárquico y agregativo de la fase-banco.
    Average_Desired_Length_Cluster, Max_Cluster_Length y Min_Cluster_Length son restricciones blandas del tamaño de los clusters. Es decir, no siempre se van a satisfacer.
    PostProcessing es una heurística para mejorar la forma de los clusters y para eliminar clusters pequeños.
    Depende de las funciones Calculate_Adjency_Matrix, Calculate_Similarity_Matrix, Find_Most_Similar_Adjacent_Clusters_Hadamard y Find_Corner_Blocks.
    Devuelve un DataFrame de la fase-banco con la columna 'cluster' que indica el cluster al que pertenece cada bloque, y las matrices de adyacencia y de similaridad originales.
    '''
    fase_banco = FaseBanco.copy()
    
    
    adj_matrix_sparse = Calculate_Adjency_Matrix(fase_banco)
    adj_matrix_copy = adj_matrix_sparse.copy()
    sim_matrix = Calculate_Similarity_Matrix(fase_banco, distancia=distancia, ley=ley, destino=destino, peso_distancia=peso_distancia, peso_ley=peso_ley, penalizacion_destino=penalizacion_destino, tol_ley=tol_ley, directional_mining=directional_mining, tol_dir=tol_dir, P_inicio=P_inicio, P_final=P_final, peso_directional_mining=peso_directional_mining)
    sim_matrix_copy = sim_matrix.copy()

    print(f'Nonzero entries of Adjency Matrix: {adj_matrix_sparse.nnz}')
    
    N_Clusters = len(fase_banco)
    n = N_Clusters
    Max_N_Clusters = len(fase_banco) // Average_Desired_Length_Cluster

    fase_banco['cluster'] = np.arange(N_Clusters).astype(int)
    Clusters_Eliminados = 0
    t1 = time.time()
    tries = 0
    while N_Clusters > Max_N_Clusters:
        t1_fmsac = time.time()
        C = Find_Most_Similar_Adjacent_Clusters_Hadamard(adj_matrix_sparse, sim_matrix)
        t2_fmsac = time.time()
        tries += 1
        if Debug:
            print(f'Try: {tries}, time: {t2_fmsac-t1_fmsac}')
        if C is None:
            break
        (i,j) = C
        cluster_i = fase_banco[fase_banco['cluster'] == fase_banco.iloc[i]['cluster']]
        cluster_j = fase_banco[fase_banco['cluster'] == fase_banco.iloc[j]['cluster']]

        if len(cluster_i) + len(cluster_j) <= Max_Cluster_Length:
            sim_matrix[i,:] = np.min([sim_matrix[i,:], sim_matrix[j,:]], axis=0)
            sim_matrix[:,i] = np.min([sim_matrix[:,i], sim_matrix[:,j]], axis=0)
            sim_matrix[j,:] = np.zeros(n)
            sim_matrix[:,j] = np.zeros(n)
            
            adj_matrix_sparse = adj_matrix_sparse.tolil()
            adj_matrix_sparse[i,:] = adj_matrix_sparse[i,:].maximum(adj_matrix_sparse[j,:])
            adj_matrix_sparse[:,i] = adj_matrix_sparse[:,i].maximum(adj_matrix_sparse[:,j])

            adj_matrix_sparse[j,:] = np.zeros(n)
            adj_matrix_sparse[:,j] = np.zeros(n)
            adj_matrix_sparse = adj_matrix_sparse.tocsr()
            adj_matrix_sparse.eliminate_zeros()

            fase_banco.loc[fase_banco['cluster'] == fase_banco.iloc[j]['cluster'], 'cluster'] = fase_banco.iloc[i]['cluster'].astype(int)
            N_Clusters -= 1
            Clusters_Eliminados += 1
        else:
            adj_matrix_sparse[i,j] = 0
            sim_matrix[i,j] = 0
            adj_matrix_sparse.eliminate_zeros()
    t2 = time.time()
    tiempo_agregacion = t2 - t1
    N_Clusters = len(fase_banco['cluster'].unique())
    fase_banco['cluster'] = fase_banco['cluster'].map(lambda x: np.array(range(1,N_Clusters+1))[np.where(fase_banco['cluster'].unique() == x)[0][0]] if x in fase_banco['cluster'].unique() else x)
    print(f"========PreProcessing Results========")
    print(f"Clusters objetivo: {Max_N_Clusters}")
    print(f"Clusters eliminados: {Clusters_Eliminados}")
    print(f"Total de clusters: {N_Clusters}")
    print(f'Tiempo: {tiempo_agregacion}')

    if PostProcessing:
        for i in range(Iterations_PostProcessing):
            if Tabesh:
                Corner_Blocks = Find_Corner_Blocks_Tabesh(fase_banco, adj_matrix_copy)
            else:
                Corner_Blocks = Find_Corner_Blocks(fase_banco, adj_matrix_copy)
            
            if len(Corner_Blocks) == 0:
                break
            for i in Corner_Blocks.keys():
                if len(Corner_Blocks[i]) == 2:
                    Cluster_to_insert = Corner_Blocks[i][int(np.round(np.random.rand()))]
                elif len(Corner_Blocks[i]) == 1:
                    choose = [Corner_Blocks[i][0], fase_banco.iloc[i]['cluster']]
                    Cluster_to_insert = choose[int(np.round(np.random.rand()))]
                else:
                    Cluster_to_insert = np.unique_counts(Corner_Blocks[i]).values[np.unique_counts(Corner_Blocks[i]).counts.argmax()]
                fase_banco.loc[i, 'cluster'] = Cluster_to_insert
        
        ID_Small_Clusters = fase_banco['cluster'].value_counts().loc[fase_banco['cluster'].value_counts() < Min_Cluster_Length].index.tolist()
        t1 = time.time()
        # print(ID_Small_Clusters)
        for id in ID_Small_Clusters:
            sum_density = fase_banco.loc[fase_banco['cluster']==id]['density'].sum()
            ley_media = fase_banco.loc[fase_banco['cluster']==id]['cut']*fase_banco.loc[fase_banco['cluster']==id]['density']
            ley_media = ley_media.sum()/sum_density

            Bloques_Frontera = []
            cluster = fase_banco.loc[fase_banco['cluster']==id]
            id_clusters_vecinos = []
            for bloque in cluster.index:
                vecinos = adj_matrix_copy[bloque,:].toarray()[0].nonzero()[0]
                # print(vecinos)
                for v in vecinos:
                    id_vecino = fase_banco['cluster'][v]
                    if id_vecino != id:
                        Bloques_Frontera.append(bloque)
                        id_clusters_vecinos.append(id_vecino)
            id_clusters_vecinos = np.unique(id_clusters_vecinos)
            # print(id_clusters_vecinos)
            min = np.inf
            id_min = 0
            for p in id_clusters_vecinos:
                sum_density_p = fase_banco.loc[fase_banco['cluster']==p]['density'].sum()
                ley_media_p = fase_banco.loc[fase_banco['cluster']==p]['cut']*fase_banco.loc[fase_banco['cluster']==p]['density']
                ley_media_p = ley_media_p.sum()/sum_density_p

                diff_ley_media = np.abs(ley_media-ley_media_p)
                print(diff_ley_media)
                if diff_ley_media < min:
                    id_min = p
                print(id_min)
            
            fase_banco.loc[fase_banco['cluster'] == id, 'cluster'] = id_min
        
        N_Clusters = len(fase_banco['cluster'].unique())
        fase_banco['cluster'] = fase_banco['cluster'].map(lambda x: np.array(range(1,N_Clusters+1))[np.where(fase_banco['cluster'].unique() == x)[0][0]] if x in fase_banco['cluster'].unique() else x)
        t2 = time.time()
        tiempo_postprocesado = t2-t1
        print(f"========PostProcessing Results========")
        print(f"Total de clusters: {N_Clusters}")
        print(f'Tiempo: {tiempo_postprocesado}')

    if PostProcessing:
        execution_time = tiempo_agregacion + tiempo_postprocesado
    else:
        execution_time = tiempo_agregacion
    return [fase_banco, N_Clusters, adj_matrix_copy, sim_matrix_copy, execution_time, adj_matrix_sparse]