import numpy as np
import numpy.matlib as matlib
import pandas as pd
import scipy as sp

import time
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors 
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go

from pathlib import Path
from copy import deepcopy

from multiprocessing import Pool, cpu_count
from collections import defaultdict



###############################################################
#################### Clustering Functions #####################
###############################################################

def Calculate_Adjency_Matrix(FaseBanco, BlockWidthX=10, BlockWidthY=10, Sectores=[]):
    """
    Construye la matriz de adyacencia entre bloques de una fase-banco en 2D, considerando sus posiciones (x, y)
    y el tamaño físico de cada bloque. Opcionalmente permite restringir la conectividad dentro de sectores definidos.

    Parámetros
    ----------
    FaseBanco : pandas.DataFrame
        DataFrame que contiene los bloques de una fase en un banco. Debe incluir al menos las columnas 'x' y 'y'.
    BlockWidthX : float, opcional
        Ancho del bloque en la dirección X. Por defecto es 10.
    BlockWidthY : float, opcional
        Ancho del bloque en la dirección Y. Por defecto es 10.
    Sectores : list de listas, opcional
        Lista de sectores, donde cada sector se define como una lista de 4 tuplas (P1, P2, P3, P4) que representan
        los vértices de un polígono rectangular (en sentido antihorario). Solo se consideran adyacencias dentro
        del mismo sector.

    Retorna
    -------
    adjency_matrix : scipy.sparse.csr_matrix
        Matriz dispersa en formato CSR que representa las conexiones de adyacencia entre bloques.
        Dos bloques están conectados si comparten borde en X o Y (adyacencia de Von Neumann).

    Notas
    -----
    - Si no se especifican sectores, la matriz de adyacencia se construye globalmente.
    - Si se especifican sectores, los bloques fuera de los sectores quedan conectados solo entre sí.
    - En caso de usar sectores, se requiere la función auxiliar `Hadamard_Product_Sparse`.

    Ejemplo
    -------
    >>> adj = Calculate_Adjency_Matrix(fase_banco, BlockWidthX=10, BlockWidthY=10)
    >>> adj.shape
    (500, 500)  # si hay 500 bloques

    """
    x = FaseBanco['x'].values
    y = FaseBanco['y'].values

    X1 = matlib.repmat(x.reshape(len(x),1), 1, len(x))
    X2 = X1.T
    Y1 = matlib.repmat(y.reshape(len(y),1), 1, len(y))
    Y2 = Y1.T

    D = np.sqrt((1/BlockWidthX**2)*(X1 - X2)**2 + (1/BlockWidthY**2)*(Y1 - Y2)**2)

    adjency_matrix = (D <= 1) & (D > 0)
    adjency_matrix = sp.sparse.csr_matrix(adjency_matrix).astype(int)

    if Sectores:
        fase_banco = FaseBanco.copy()
        fase_banco['sector'] = 0
        sector_counter = 1

        n = len(fase_banco)
        sector_matrix = np.zeros((n,n), dtype=int)
        for sector in Sectores:
            mask = ( ((fase_banco['x'] > sector[0][0])*(fase_banco['y'] > sector[0][1]))*((fase_banco['x'] < sector[1][0])*(fase_banco['y'] > sector[1][1]))*((fase_banco['x'] < sector[2][0])*(fase_banco['y'] < sector[2][1]))*((fase_banco['x'] > sector[3][0])*(fase_banco['y'] < sector[3][1])) ) == True
            idx = fase_banco.loc[mask].index
            fase_banco.loc[idx, 'sector'] = sector_counter
            sector_counter += 1
            for i in idx:
                row = np.zeros(n)
                row[idx] = 1
                sector_matrix[i,:] = row

        idx = fase_banco.loc[fase_banco['sector']==0].index
        for i in idx:
            row = np.zeros(n)
            row[idx] = 1
            sector_matrix[i,:] = row
        
        adjency_matrix = Hadamard_Product_Sparse(adjency_matrix, sector_matrix)
        adjency_matrix.eliminate_zeros()

        # if debug:
        #     return adjency_matrix, sector_matrix, fase_banco
        # else:
        return adjency_matrix

    else:
        return adjency_matrix


def Calculate_Vertical_Adyacency_Matrix(df_sup, df_inf, BlockWidth=10, BlockHeight=10, arcs=defaultdict(list), cluster_col='cluster'):
    """
    Calcula la matriz de adyacencia vertical entre clusters de dos fases consecutivas de la mina.

    Se asume que `df_sup` corresponde a una fase superior (más alta en Z) y `df_inf` a la fase inmediatamente inferior.
    La función identifica bloques adyacentes verticalmente según su distancia euclidiana en el plano (x, y), normalizada
    por el tamaño del bloque, y luego construye una matriz de precedencia entre clusters de ambas fases.

    Parámetros
    ----------
    df_sup : pandas.DataFrame
        DataFrame que representa la fase superior. Debe contener las columnas 'x', 'y' y `cluster_col`.
    df_inf : pandas.DataFrame
        DataFrame que representa la fase inferior. Debe contener las columnas 'x', 'y' y `cluster_col`.
    BlockWidth : float, opcional
        Ancho de los bloques en la dirección X. Usado para normalizar distancias. Por defecto es 10.
    BlockHeight : float, opcional
        Altura (profundidad) de los bloques en la dirección Y. Usado para normalizar distancias. Por defecto es 10.
    arcs : collections.defaultdict(list), opcional
        Parámetro no utilizado directamente en esta función, pero se puede usar como acumulador externo de arcos.
        Por defecto es un defaultdict vacío.
    cluster_col : str, opcional
        Nombre de la columna que identifica el cluster al que pertenece cada bloque. Por defecto es 'cluster'.

    Retorna
    -------
    A_clusters : numpy.ndarray de tipo int
        Matriz binaria de tamaño (n_sup, n_inf), donde `n_sup` es el número de clusters en la fase superior y
        `n_inf` es el número de clusters en la fase inferior. La entrada (i, j) es 1 si hay al menos un bloque del
        cluster `i` en la fase superior que es adyacente a un bloque del cluster `j` en la fase inferior.

    Notas
    -----
    - La adyacencia se determina en base a la distancia euclidiana en el plano (x, y), normalizada por
      el tamaño de los bloques. Se considera adyacente si la distancia ≤ 1.
    - El resultado permite definir relaciones de precedencia entre clusters en una estructura tipo DAG.
    - El DataFrame debe estar limpio y no contener valores faltantes en las columnas relevantes.

    Ejemplo
    -------
    >>> A = Calculate_Vertical_Adyacency_Matrix(df_sup, df_inf, BlockWidth=10, BlockHeight=10)
    >>> A.shape
    (12, 15)  # 12 clusters en fase superior, 15 en fase inferior
    """
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


def Calculate_Similarity_Matrix(FaseBanco, params = dict()):
    """
    Calcula una matriz de similaridad entre bloques de una fase-banco, combinando criterios espaciales,
    geológicos y operacionales.

    La similaridad entre cada par de bloques se calcula en función de:
    - distancia espacial (x, y),
    - diferencia en ley de mineral (columna 'cut'),
    - diferencia en destino ('destino'),
    - diferencia en tipo de roca ('tipomineral'),
    - orientación relativa respecto a una dirección de minería.

    Los criterios se ponderan mediante parámetros configurables. Si un criterio no se desea considerar,
    su peso o penalización puede ser fijado a 0.

    Parámetros
    ----------
    FaseBanco : pandas.DataFrame
        DataFrame que representa los bloques de una fase en un banco.
        Debe incluir al menos las columnas: 'x', 'y', 'cut', 'destino'.
        Opcionalmente: 'tipomineral'.

    params : dict, opcional
        Diccionario con los parámetros de configuración:
        - 'peso_distancia' : float, peso para la distancia euclidiana entre bloques (default: 2)
        - 'peso_ley' : float, peso para la diferencia de ley ('cut') (default: 0)
        - 'tolerancia_ley' : float, valor mínimo admisible para diferencia de ley (default: 0.001)
        - 'peso_direccion_mineria' : float, peso para penalizar bloques con diferente orientación (default: 0.25)
        - 'tolerancia_direccion_mineria' : float, valor mínimo admisible para diferencia direccional (default: 0.001)
        - 'vector_mineria' : list de dos tuplas [(x1,y1), (x2,y2)], representa dirección de avance de la mina
        - 'penalizacion_destino' : float entre 0 y 1. Penaliza si los bloques tienen distinto destino (default: 0.9)
        - 'penalizacion_roca' : float entre 0 y 1. Penaliza si los bloques tienen distinto tipo de roca (default: 0.9)

    Retorna
    -------
    similarity_matrix : numpy.ndarray
        Matriz densa (n x n) donde cada entrada (i, j) representa el grado de similaridad entre los bloques i y j.
        Valores más altos indican mayor similaridad. La diagonal no se considera (por convención, se omite).

    Errores
    -------
    ValueError
        Si el DataFrame `FaseBanco` está vacío.

    Notas
    -----
    - La similaridad es mayor cuando los bloques son cercanos, tienen leyes similares, el mismo destino y tipo de roca,
      y se alinean con la misma dirección de minería.
    - La función es sensible a los pesos: un peso 0 excluye completamente un criterio.
    - Las matrices se normalizan internamente para evitar dominancia de escala.

    Ejemplo
    -------
    >>> sim = Calculate_Similarity_Matrix(fase, {'peso_distancia': 1, 'peso_ley': 1, 'penalizacion_destino': 0.8})
    >>> sim.shape
    (500, 500)
    """
    if FaseBanco.empty:
        raise ValueError('Inserte una FaseBanco no vacía')
    
    params.setdefault('peso_distancia', 2)
    params.setdefault('peso_ley', 0)
    params.setdefault('tolerancia_ley', 0.001)
    params.setdefault('peso_direccion_mineria', 0.25)
    params.setdefault('tolerancia_direccion_mineria', 0.001)
    params.setdefault('vector_mineria', [])
    params.setdefault('penalizacion_destino', 0.9)
    params.setdefault('penalizacion_roca', 0.9)

    peso_distancia = params['peso_distancia']
    peso_ley = params['peso_ley']
    tol_ley = params['tolerancia_ley']
    peso_directional_mining = params['peso_direccion_mineria']
    tol_directional_mining = params['tolerancia_direccion_mineria']
    if (len(params['vector_mineria'])==0) | (peso_directional_mining==0):
        peso_directional_mining = 0
    else:
        P_inicio, P_final = params['vector_mineria']
    penalizacion_destino = params['penalizacion_destino']
    penalizacion_roca = params['penalizacion_roca']

    n = len(FaseBanco)

    similarity_matrix = np.zeros((n, n), dtype=float)

    x = FaseBanco['x'].values
    y = FaseBanco['y'].values

    # Distancia
    if peso_distancia==0:
        D = 1
        ND = 1
    else:
        X1 = matlib.repmat(x.reshape(len(x),1), 1, len(x))
        X2 = X1.T
        Y1 = matlib.repmat(y.reshape(len(y),1), 1, len(y))
        Y2 = Y1.T

        D = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)
        ND = D / np.max(D)

    # Ley
    if peso_ley == 0:
        G = 1
        NG = 1
    else:
        g = FaseBanco['cut'].values
        G1 = matlib.repmat(g.reshape(len(g),1), 1, len(g))
        G2 = G1.T

        G = np.maximum(np.abs(G1 - G2), tol_ley)
        NG = G / np.max(G)

    # Destino
    if penalizacion_destino==0:
        T = 1
    else:
        t = FaseBanco['destino'].values
        T1 = matlib.repmat(t.reshape(len(t),1), 1, len(t))
        T2 = T1.T
        T = T1 != T2

        T = np.ones((n,n)) - (1-penalizacion_destino)*T*np.ones((n,n))

    # Tipo Roca/Material
    if 'tipomineral' in FaseBanco.columns:
        if penalizacion_roca==0:
            R=1
        else:
            rock = FaseBanco['tipomineral'].values
            R1 = matlib.repmat(rock.reshape(len(rock),1), 1, len(rock))
            R2 = R1.T
            R = R1 != R2

            R = np.ones((n,n)) - (1-penalizacion_roca)*R*np.ones((n,n))
    else:
        R = 1

    # Dirección de minería
    if peso_directional_mining==0:
        DM = 1
        NDM = 1
    else:
        M1 = (X1 - P_inicio[0])**2 + (Y1 - P_inicio[1])**2
        M2 = (X1 - P_final[0])**2 + (Y1 - P_final[1])**2
        Mi = np.multiply( np.sign( M1 - M2 ), np.sqrt( np.abs(M1 - M2 )) )
        DM = np.maximum(np.abs(Mi - Mi.T), tol_directional_mining)

    NDM = DM / np.max(DM)

    numerator = np.multiply(R,T)
    denominator = np.multiply(ND**peso_distancia, NG**peso_ley)
    denominator = np.multiply(denominator, NDM**peso_directional_mining)
    similarity_matrix = np.divide(numerator,  denominator, where=np.ones((n,n)) - np.diag(np.ones(n))==1)

    return similarity_matrix


def Find_Most_Similar_Adjacent_Clusters(AdjencyMatrix, SimilarityMatrix):
    """
    Encuentra el par de clusters adyacentes con mayor similaridad, considerando tanto la conectividad
    (adyacencia) como el grado de similitud entre ellos.

    La función utiliza un producto de Hadamard entre la matriz de adyacencia y la matriz de similaridad,
    de modo que solo se consideran pares de clusters que estén conectados directamente. Luego,
    identifica el par con mayor valor de similaridad.

    Parámetros
    ----------
    AdjencyMatrix : scipy.sparse.spmatrix
        Matriz dispersa binaria que representa la adyacencia entre clusters (por ejemplo, de una fase minera).
        Debe estar en un formato soportado por `Hadamard_Product_Sparse`, como CSR o COO.
    
    SimilarityMatrix : numpy.ndarray
        Matriz densa (n x n) que representa el grado de similaridad entre cada par de clusters.

    Retorna
    -------
    pares_similares : list[int, int] o None
        Lista con los índices `[i, j]` del par de clusters más similares entre sí que además sean adyacentes.
        El índice menor siempre aparece primero (es decir, `i < j`). Si no hay ningún par adyacente, retorna `None`.

    Requiere
    --------
    - La función `Hadamard_Product_Sparse` debe estar definida en el entorno.

    Notas
    -----
    - Si hay múltiples pares con la misma similaridad máxima, se devuelve solo el primero encontrado.
    - La función es útil en procesos iterativos de fusión de clusters basados en conectividad y similitud.

    Ejemplo
    -------
    >>> pares = Find_Most_Similar_Adjacent_Clusters(ad_matrix, sim_matrix)
    >>> print(pares)
    [2, 7]  # Los clusters 2 y 7 son adyacentes y los más similares
    """
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


def Clustering_Tabesh(FaseBanco, AdjencyMatrix, SimilarityMatrix, params=dict(), animation_mode=False):
    """
    Ejecuta un algoritmo de clustering jerárquico y agregativo sobre una fase-banco, basado en la metodología de Tabesh.

    A partir de una matriz de adyacencia y una matriz de similaridad entre bloques, el algoritmo fusiona iterativamente
    los clusters más similares y adyacentes, hasta cumplir una restricción de tamaño promedio de cluster.
    Se aplican restricciones adicionales como el tamaño máximo por cluster.

    Parámetros
    ----------
    FaseBanco : pandas.DataFrame
        DataFrame que representa los bloques de una fase en un banco. Debe incluir al menos las columnas:
        'x', 'y', y variables geológicas o de destino, según se hayan usado en la construcción de la matriz de similaridad.

    AdjencyMatrix : scipy.sparse.spmatrix
        Matriz de adyacencia dispersa entre bloques, que define qué bloques se consideran vecinos directos.

    SimilarityMatrix : numpy.ndarray
        Matriz densa que indica el grado de similaridad entre pares de bloques.

    params : dict, opcional
        Parámetros de configuración del clustering. Las claves reconocidas son:
        - 'Average_Length_Cluster' : int. Tamaño promedio objetivo por cluster (default: 30)
        - 'Max_Length_Cluster'     : int. Tamaño máximo permitido por cluster (default: 35)
        - 'Reset_Clusters_Index'   : bool. Si True, los índices de los clusters se reenumeran al final (default: True)
        - 'debug'                  : bool. Muestra información de depuración si es True (default: False)

    animation_mode : bool, opcional
        Si es True, se guarda el historial completo del proceso de fusión de clusters, útil para animaciones o visualización.

    Retorna
    -------
    fase_banco : pandas.DataFrame
        DataFrame original con una nueva columna `'cluster'` que indica el cluster asignado a cada bloque.

    execution_time : float
        Tiempo total de ejecución del proceso (en segundos).

    N_Clusters : int
        Número total de clusters resultantes.

    history : list de DataFrame (solo si `animation_mode=True`)
        Historial del estado del DataFrame `fase_banco` después de cada fusión, para reconstrucción del proceso paso a paso.

    Notas
    -----
    - El algoritmo fusiona pares de clusters más similares y adyacentes hasta alcanzar un número objetivo de clusters,
      determinado por `Average_Length_Cluster`.
    - Se utiliza la función `Find_Most_Similar_Adjacent_Clusters` para seleccionar los candidatos a fusión.
    - La matriz de adyacencia y de similaridad se actualizan dinámicamente después de cada fusión.
    - Si la fusión entre dos clusters supera el tamaño máximo permitido (`Max_Length_Cluster`), se descarta dicha fusión.

    Ejemplo
    -------
    >>> resultado, tiempo, n_clusters = Clustering_Tabesh(fase, adj, sim)
    >>> resultado['cluster'].nunique()
    17
    """
    params.setdefault('Average_Length_Cluster', 30)
    params.setdefault('Max_Length_Cluster', 35)
    params.setdefault('Reset_Clusters_Index', True)
    params.setdefault('debug', False)

    Average_Desired_Length_Cluster = params['Average_Length_Cluster']
    Max_Cluster_Length = params['Max_Length_Cluster']
    Reset_Clusters_Index = params['Reset_Clusters_Index']
    debug = params['debug']

    history = []

    
    fase_banco = FaseBanco.copy()
    adj_matrix_sparse = AdjencyMatrix
    sim_matrix = SimilarityMatrix
    execution_time = 0

    if debug:
        print(f'Nonzero entries of Adjency Matrix: {AdjencyMatrix.nnz}')
        tries = 0

    N_Clusters = len(fase_banco)
    n = N_Clusters
    Max_N_Clusters = N_Clusters // Average_Desired_Length_Cluster

    fase_banco['cluster'] = np.arange(N_Clusters).astype(int)
    Clusters_Eliminados = 0
    t1 = time.time()

    while N_Clusters > Max_N_Clusters:
        t1_fmsac = time.time()
        C = Find_Most_Similar_Adjacent_Clusters(adj_matrix_sparse, sim_matrix)
        t2_fmsac = time.time()
        
        if debug:
            tries += 1
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

            if animation_mode:
                history.append(fase_banco.copy())
        else:
            adj_matrix_sparse[i,j] = 0
            adj_matrix_sparse[j,i] = 0
            sim_matrix[i,j] = 0
            sim_matrix[j,i] = 0
            adj_matrix_sparse.eliminate_zeros()

    t2 = time.time()
    execution_time = t2 - t1
    N_Clusters = len(fase_banco['cluster'].unique())

    if Reset_Clusters_Index:
        fase_banco['cluster'] = fase_banco['cluster'].map(lambda x: np.array(range(1,N_Clusters+1))[np.where(fase_banco['cluster'].unique() == x)[0][0]] if x in fase_banco['cluster'].unique() else x)

    if animation_mode:
        history.append(fase_banco.copy())

    print(f"========PreProcessing Results========")
    print(f"Tamaño Fase-Banco: {n}")
    print(f"Clusters objetivo: {Max_N_Clusters}")
    print(f"Clusters eliminados: {Clusters_Eliminados}")
    print(f"Total de clusters: {N_Clusters}")
    print(f'Tiempo: {execution_time}')

    if animation_mode:
        return fase_banco, execution_time, N_Clusters, history
    return fase_banco, execution_time, N_Clusters


def Clustering_mina(mina, cm=2, cr=0.25, cp=10, P=4, R=0.85, alpha_ley_corte=0,
               options=dict()):
    """
    Realiza un proceso jerárquico de clusterización sobre un bloque de mina dividido por fases y bancos, 
    agrupando los bloques en clusters con criterios geológicos, económicos y geométricos.

    Este método clasifica bloques de una mina en clústeres operacionales, aplicando restricciones de ley de corte, 
    precedencias operativas y formas geométricas. Se realiza por fases y bancos, permitiendo un refinamiento posterior 
    de la forma de los clústeres y el control sobre el tamaño y distribución espacial de los mismos. 
    Se generan métricas geológicas y relaciones de precedencia entre clusters para planificaciones futuras.

    Parameters
    ----------
    mina : pandas.DataFrame
        DataFrame que contiene los bloques mineros, con columnas esperadas como 'x', 'y', 'z', 'fase', 'banco', 
        'cut' (ley), 'id' y otras relacionadas con características físicas y económicas del bloque.

    cm : float, optional
        Costo marginal por tonelada de material extraído (default es 2).

    cr : float, optional
        Costo de refinación por tonelada (default es 0.25).

    cp : float, optional
        Costo de procesamiento por tonelada (default es 10).

    P : float, optional
        Precio por unidad de metal contenido (default es 4).

    R : float, optional
        Recuperación metalúrgica (default es 0.85).

    alpha_ley_corte : float, optional
        Parámetro entre 0 y 1 para interpolar entre ley marginal y ley crítica como ley de corte 
        (default es 0).

    options : dict, optional
        Diccionario con múltiples parámetros opcionales para controlar pesos, tolerancias, 
        refinamiento de forma, rutas de guardado y visualización. Algunos parámetros esperados son:
        - BlockWidthX, BlockWidthY, BlockHeightZ
        - peso_distancia, peso_ley, peso_direccion_mineria
        - penalizacion_destino, penalizacion_roca, penalizacion_c
        - tamaño_maximo_cluster, tamaño_promedio_cluster, tamaño_minimo_cluster
        - Shape_Refinement (None, 'Tabesh', 'Modificado')
        - Iteraciones_Shape_Refinement
        - save, save_images, path_save, path_save_images
        - show_block_label, cmap

    Returns
    -------
    mina_clusterizada : pandas.DataFrame
        DataFrame de bloques con columna adicional 'cluster' indicando el agrupamiento al que pertenece cada bloque.

    metrics_df : pandas.DataFrame
        DataFrame con métricas por fase y banco: 
        RU (Rock Unity), DDF (Destination Dilution Factor), CV (Coefficient of Variation), 
        y sus respectivas distribuciones.

    precedences_df : pandas.DataFrame
        DataFrame con las precedencias entre clústeres dentro de cada fase-banco, útil para ordenamiento y planificación.

    Raises
    ------
    ValueError
        Si el DataFrame `mina` está vacío.

    Notes
    -----
    - El proceso de clusterización se hace banco por banco dentro de cada fase.
    - Se usa un refinamiento de forma opcional para mejorar la geometría de los clusters.
    - Los pesos y penalizaciones permiten balancear la influencia de factores como distancia, ley y dirección de minería.
    - El cálculo de ley de corte depende de parámetros económicos y se puede ajustar con `alpha_ley_corte`.
    - Se asumen funciones auxiliares como `Clustering_Tabesh`, `Shape_Refinement_Tabesh`, `Calculate_Adjency_Matrix`, etc.
    - Si se habilita, se guardan visualizaciones por banco con leyenda y vectores de minería.

    Examples
    --------
    >>> df_result, metrics, precedences = Clustering_mina(mina_df)
    >>> df_result['cluster'].unique()
    array([0, 1, 2, 3, ...])
    """
    if mina.empty:
        raise ValueError('Inserte un dataframe "mina" no vacio.')
    
    FTL = 2204.62

    ley_marginal = cp/((P-cr)*FTL*R)
    ley_critica = (cp+cm)/((P-cr)*FTL*R)

    ley_corte = ((1-alpha_ley_corte)*ley_marginal + alpha_ley_corte*(ley_critica))*100

    if not('destino' in mina.columns):
        mina['destino'] = [1 if mina.iloc[i]['cut']>= ley_corte else 0 for i in range(len(mina))]


    options.setdefault('BlockWidthX', 10)
    options.setdefault('BlockWidthY', 10)
    options.setdefault('BlockHeightZ', 16)
    options.setdefault('area_minima_operativa', np.pi*80*80)


    options.setdefault('peso_distancia', 2)
    options.setdefault('peso_ley', 0)
    options.setdefault('tolerancia_ley', 0.001)
    options.setdefault('peso_direccion_mineria', 0.25)
    options.setdefault('tolerancia_direccion_mineria', 0.001)
    options.setdefault('penalizacion_destino', 0.9)
    options.setdefault('penalizacion_roca', 0.9)
    options.setdefault('penalizacion_c', 0.5)

    options.setdefault('tamaño_maximo_cluster', 0.1)
    options.setdefault('tamaño_promedio_cluster', 0.075)
    options.setdefault('tamaño_minimo_cluster', 0.01)
    options.setdefault('tolerancia_tamaño_maximo_cluster', 10)
    options.setdefault('tolerancia_tamaño_minimo_cluster', 5)

    options.setdefault('Shape_Refinement', 'Tabesh') # None, 'Tabesh' o 'Modificado'
    options.setdefault('Iteraciones_Shape_Refinement', 5)

    options.setdefault('save', True)
    options.setdefault('save_images', True)
    options.setdefault('path_save', 'Clusterizacion/Resultados/')
    options.setdefault('path_save_images', 'Clusterizacion/Imagenes/')
    options.setdefault('path_params', 'Clusterizacion/Params/')
    options.setdefault('show_block_label', True)
    options.setdefault('cmap', 'plasma')

    BlockWidthX = options['BlockWidthX']
    BlockWidthY = options['BlockWidthY']
    BlockHeightZ = options['BlockHeightZ']
    area_minima_operativa = options['area_minima_operativa']


    
    peso_distancia = options['peso_distancia']
    peso_ley = options['peso_ley']
    tol_ley = options['tolerancia_ley']
    peso_directional_mining = options['peso_direccion_mineria']
    tol_directional_mining = options['tolerancia_direccion_mineria']
    penalizacion_destino = options['penalizacion_destino']
    penalizacion_roca = options['penalizacion_roca']
    penalizacion_c = options['penalizacion_c']

    
    if options['tamaño_maximo_cluster'] <= 1:
        Max_Cluster_Length_multiplier = options['tamaño_maximo_cluster']
        max_cluster_mode = True
    else:
        Max_Cluster_Length_st = int(options['tamaño_maximo_cluster'])
        max_cluster_mode = False
    
    if options['tamaño_promedio_cluster'] <= 1:
        Average_Cluster_Length_multiplier = options['tamaño_promedio_cluster']
        average_cluster_mode = True
    else:
        Average_Cluster_Length_st = int(options['tamaño_promedio_cluster'])
        average_cluster_mode = False
    
    if options['tamaño_minimo_cluster'] <= 1:
        Min_Cluster_Length_multiplier = options['tamaño_minimo_cluster']
        min_cluster_mode = True
    else:
        Min_Cluster_Length_st = int(options['tamaño_minimo_cluster'])
        min_cluster_mode = False

    Iterations_PostProcessing = options['Iteraciones_Shape_Refinement']

    save = options['save']
    save_images = options['save_images']
    path_save = options['path_save']
    path_params = options['path_params']
    path_save_images = options['path_save_images']
    show_block_label = options['show_block_label']
    cmap = options['cmap']

    print(f'---------Ley de corte usada: {ley_corte}---------\n')
    fases_mina = sorted(mina['fase'].unique())
    Tamaños_fase_banco = []
    Tiempos_clusterizacion_fase_banco = []

    contador_bancos = 1
    contador_clusters = 0

    Diccionario_Centros = dict()

    for fase in fases_mina:
        mini_mina = mina[mina['fase'] == fase]
        bancos = sorted(mini_mina['banco'].unique())[::-1]
        if penalizacion_c > 0:
            res = Precedencia_Fase_Banco(mini_mina)
            precedence_tree = defaultdict(list)
            for node, predecessor in res:
                precedence_tree[node].append(predecessor)
                
        if peso_directional_mining > 0:
            cono = list( np.load(Path(path_params + f'cono_{fase}.npy')) )

            rampa = list( np.load(Path(path_params + f'puntos_rampa_{fase}.npy')) )

            z_min = Z_min(mini_mina, cono, area_minima_operativa)
            puntos_iniciales = Puntos_Iniciales(mini_mina, rampa, z_min)

        for banco in bancos:
            print(f'\n')
            print(f'Fase-Banco N° {fase}-{banco}')
            fase_banco = mina[(mina['fase'] == fase) & (mina['banco'] == banco)].copy()
            z_fb = fase_banco['z'].values[0]

            x_min = np.min(fase_banco['x'].values)-BlockWidthX
            y_min = np.min(fase_banco['y'].values)-BlockWidthY
            x_max = np.max(fase_banco['x'].values)+BlockWidthX
            y_max = np.max(fase_banco['y'].values)+BlockWidthY

            peso_directional_mining_new = peso_directional_mining
            vector_mineria = []
            if peso_directional_mining>0:
                P1 = ()
                for p in puntos_iniciales:
                    if p[2]==z_fb:
                        P1 = (p[0], p[1])
                        alpha = (p[0] - x_min)/(x_max - x_min)
                        beta = (p[1] - y_min)/(y_max - y_min)
                        P2 = ( alpha*x_min + (1-alpha)*x_max, beta*y_min + (1-beta)*y_max)
                        vector_mineria = (P1, P2)
                        break

                if len(P1)==0:
                    peso_directional_mining_new = 0


            tamaño_fase_banco = len(fase_banco)
            Tamaños_fase_banco.append(tamaño_fase_banco)
            print(f'Tamaño de la fase-banco: {tamaño_fase_banco}')
            

            adjency_matrix = Calculate_Adjency_Matrix(fase_banco, BlockWidthX, BlockWidthY)
            similarity_matrix = Calculate_Similarity_Matrix(
                fase_banco, 
                params={
                    'peso_distancia': peso_distancia,
                    'peso_ley': peso_ley,
                    'tolerancia_ley': tol_ley,
                    'peso_direccion_mineria': peso_directional_mining_new,
                    'tolerancia_direccion_mineria': tol_directional_mining,
                    'vector_mineria': vector_mineria,
                    'penalizacion_destino': penalizacion_destino,
                    'penalizacion_roca': penalizacion_roca
                }
                )
            
            if max_cluster_mode:
                Max_Cluster_Length = int(Max_Cluster_Length_multiplier*tamaño_fase_banco)
            else:
                Max_Cluster_Length = Max_Cluster_Length_st
            if average_cluster_mode:
                Average_Cluster_Length = int(Average_Cluster_Length_multiplier*tamaño_fase_banco)
            else:
                Average_Cluster_Length = Average_Cluster_Length_st
            if min_cluster_mode:
                Min_Cluster_Length = int(Min_Cluster_Length_multiplier*tamaño_fase_banco)
            else:
                Min_Cluster_Length = Min_Cluster_Length_st

            if Max_Cluster_Length==0:
                mult = 2/3
            else:
                mult = Average_Cluster_Length/Max_Cluster_Length


            if Max_Cluster_Length < options['tolerancia_tamaño_maximo_cluster']:
                Max_Cluster_Length = options['tolerancia_tamaño_maximo_cluster']
                Average_Cluster_Length = int(mult*Max_Cluster_Length)
            if Min_Cluster_Length < options['tolerancia_tamaño_minimo_cluster']:
                Min_Cluster_Length = options['tolerancia_tamaño_minimo_cluster']

            print(f'MaxCL: {Max_Cluster_Length}, AveCL: {Average_Cluster_Length}, MinCL: {Min_Cluster_Length}')
            if (penalizacion_c>0):
                if ((fase,banco) in precedence_tree):
                    print(f"---Precedencias Detectadas")
                    print(precedence_tree[(fase,banco)])
                    dfs = [] # Lista para almacenar los dataframes precedentes
                    for (f_,b_) in precedence_tree[(fase,banco)]:
                        dfs.append( (mina_clusterizada[(mina_clusterizada['fase']==f_)&(mina_clusterizada['banco']==b_)]).copy() )
                    offset=0
                    for df_ in dfs:
                        if not df_.empty:
                            # Ajustamos los clusters
                            df_['cluster'] = df_['cluster'] + offset
                            # Actualizamos el offset: suma el máximo cluster de este df
                            offset = df_['cluster'].max()
                    
                    df_inf = pd.concat(dfs, ignore_index=True) #dataframe inferior a f,b
                    df_sup = fase_banco.copy()
                    df_sup.reset_index(drop=True, inplace=True)
                    df_sup['cluster'] = df_sup['id']
                    A_vertical = Calculate_Vertical_Adyacency_Matrix(df_sup,df_inf, BlockWidth=BlockWidthX/4, BlockHeight=BlockWidthY/4) # Tiene que ser estricto pues es para calcular C
                    # Ahora calculamos la matriz C
                    C = A_vertical@A_vertical.T
                    C = (1-C)*penalizacion_c+C
                    #Ahora la similaridad
                    similarity_matrix = similarity_matrix*C



            fase_banco, execution_time_agg, N_Clusters_agg = Clustering_Tabesh(
                fase_banco, adjency_matrix, similarity_matrix,
                params={'Average_Length_Cluster': Average_Cluster_Length,
                        'Max_Length_Cluster': Max_Cluster_Length}
                )
            
            execution_time_ref = 0
            N_Clusters_ref = N_Clusters_agg
            if options['Shape_Refinement']=='Tabesh':
                fase_banco, execution_time_ref, N_Clusters_ref = Shape_Refinement_Tabesh(
                    fase_banco,
                    adjency_matrix,
                    Min_Cluster_Length=Min_Cluster_Length,
                    Iterations_PostProcessing=Iterations_PostProcessing
                )
            elif options['Shape_Refinement']=='Modificado':
                fase_banco, execution_time_ref, N_Clusters_ref = Shape_Refinement_Mod(
                    fase_banco,
                    adjency_matrix,
                    Min_Cluster_Length=Min_Cluster_Length,
                    Iterations_PostProcessing=Iterations_PostProcessing
                )
            
            Clusters_Ordenados, fase_banco, Centers, execution_time_prec = Precedencias_Clusters_Agend(fase_banco, vector_mineria, BlockWidthX, BlockWidthY, Distance_Option=True)

            Diccionario_Centros[(fase, banco)] = Centers

            ru, ru_distribution = Rock_Unity(fase_banco)
            ddf, ddf_distribution = Destination_Dilution_Factor(fase_banco)
            cv, cv_distribution = Coefficient_Variation(fase_banco)

            if contador_bancos == 1:
                mina_clusterizada = fase_banco.copy()
                metrics_df = pd.DataFrame(
                    {'fase': fase, 
                    'banco': banco, 
                    'RU': ru, 
                    'DDF': ddf, 
                    'CV': cv, 
                    'RU_dist': [ru_distribution],
                    'DDF_dist': [ddf_distribution],
                    'CV_dist': [cv_distribution]}
                    )
                precedences_df = pd.DataFrame({'fase': fase, 'banco': banco, 'cluster': [int(k) for k in Clusters_Ordenados.keys()], 'precedences': [[int(x) for x in v] for v in Clusters_Ordenados.values()]})
            else:
                mina_clusterizada = pd.concat([mina_clusterizada, fase_banco], ignore_index=True)
                metrics_df = pd.concat(
                    [metrics_df, 
                    pd.DataFrame(
                        {'fase': fase, 
                        'banco': banco, 
                        'RU': ru, 
                        'DDF': ddf, 
                        'CV': cv, 
                        'RU_dist': [ru_distribution],
                        'DDF_dist': [ddf_distribution],
                        'CV_dist': [cv_distribution]}
                    )], 
                    ignore_index=True)

                precedences_df = pd.concat(
                    [precedences_df, 
                    pd.DataFrame(
                        {'fase': fase, 
                        'banco': banco, 
                        'cluster': [int(k) for k in Clusters_Ordenados.keys()], 
                        'precedences': [[int(x) for x in v] for v in Clusters_Ordenados.values()]}
                        )], 
                    ignore_index=True)

            contador_bancos += 1
            contador_clusters += N_Clusters_ref
            tiempo_ejecucion = execution_time_agg + execution_time_ref + execution_time_prec
            Tiempos_clusterizacion_fase_banco.append(tiempo_ejecucion)

            if save_images:
                path_image = path_save_images
                

                plot_fase_banco(fase_banco, column_hue='cluster', params={
                    'BlockWidthX': BlockWidthX,
                    'BlockWidthY': BlockWidthY,
                    'dpi': 150,
                    'flechas': [vector_mineria] if len(vector_mineria)>0 else [],
                    'precedencias': Clusters_Ordenados,
                    'centros': Centers,
                    'guardar_imagen': True,
                    'path': path_image,
                    'show_block_label': show_block_label,
                    'cmap': cmap
                })
            
    print(f'Clusters creados: {contador_clusters}')
    print(f'Tiempo total de ejecucion: {sum(Tiempos_clusterizacion_fase_banco)}')


    return mina_clusterizada, metrics_df, precedences_df


def clustering_wrapper(args):
    """
    Desempaqueta los argumentos necesarios y ejecuta ``Clustering_mina`` sobre
    un subconjunto (fase) del DataFrame de bloques mineros.

    Parameters
    ----------
    args : tuple
        Tupla con ocho elementos en el siguiente orden::

            (minimina, cm, cr, cp, P, R, alpha_ley_corte, options)

        donde

        * minimina : pandas.DataFrame  
          Sub‑DataFrame que contiene todos los bloques de una fase específica.
        * cm : float  
          Costo marginal por tonelada.
        * cr : float  
          Costo de refinación por tonelada.
        * cp : float  
          Costo de procesamiento por tonelada.
        * P : float  
          Precio del metal por unidad de contenido.
        * R : float  
          Recuperación metalúrgica.
        * alpha_ley_corte : float  
          Factor ∈ [0, 1] que ajusta la interpolación entre ley marginal
          y ley crítica al calcular la ley de corte.
        * options : dict  
          Diccionario de configuración que se pasa sin cambios
          a ``Clustering_mina``.

    Returns
    -------
    tuple
        ``(mina_clusterizada, metrics_df, precedences_df)`` tal como lo
        devuelve ``Clustering_mina`` para la fase procesada.

    Notes
    -----
    Esta función está pensada para usarse con ``multiprocessing.Pool`` dentro
    de :pyfunc:`Clustering_parallel`, por lo que debe ser *pickle‑safe*.
    No realiza validaciones de los tipos de cada elemento; asume que
    ``Clustering_parallel`` preparó la tupla correctamente.
    """
    minimina, cm, cr, cp, P, R, alpha_ley_corte, options = args
    return Clustering_mina(minimina, cm=cm, cr=cr, cp=cp, P=P, R=R, alpha_ley_corte=alpha_ley_corte, options=options)


def Clustering_parallel(mina, workers=1, cm=2, cr=0.25, cp=10, P=4, R=0.85, alpha_ley_corte=0,
               options=dict()):
    """
    Ejecuta la clusterización de bloques mineros por fase en paralelo, 
    utilizando múltiples procesos para acelerar el agrupamiento por clústeres.

    Esta función divide el DataFrame de entrada por fases y aplica `Clustering_mina` a cada una en paralelo. 
    Devuelve la mina clusterizada completa y guarda automáticamente los resultados si se especifica.

    Parameters
    ----------
    mina : pandas.DataFrame
        DataFrame con los bloques de la mina. Debe contener, al menos, las columnas 'fase' y 'banco'.

    workers : int, optional
        Número de procesos paralelos a utilizar. Si es -1, se usan todos los núcleos disponibles. Default es 1.

    cm : float, optional
        Costo marginal por tonelada (default es 2).

    cr : float, optional
        Costo de refinación por tonelada (default es 0.25).

    cp : float, optional
        Costo de procesamiento por tonelada (default es 10).

    P : float, optional
        Precio del metal (default es 4).

    R : float, optional
        Recuperación metalúrgica (default es 0.85).

    alpha_ley_corte : float, optional
        Parámetro entre 0 y 1 que interpola entre ley marginal y crítica para calcular la ley de corte (default es 0).

    options : dict, optional
        Diccionario de configuración con los siguientes campos relevantes:
        - 'save' : bool, si se guardan los resultados (default: True).
        - 'path_save' : str, carpeta donde se guardan los archivos CSV y el tiempo de ejecución.

    Returns
    -------
    mina_final : pandas.DataFrame
        DataFrame resultante con todos los bloques clusterizados por fase y banco. Incluye columna 'cluster'.

    Side Effects
    ------------
    - Guarda los resultados en CSV si `options['save']` es True:
        - `mina_clusterizada.csv`
        - `metricas.csv`
        - `precedencias.csv`
    - Guarda el tiempo total de ejecución en `tiempo.npy`.

    Notes
    -----
    - La función asume que existe una función auxiliar llamada `clustering_wrapper`, que aplica `Clustering_mina` a un subconjunto de datos.
    - Se requiere que el DataFrame `mina` tenga al menos la columna 'fase'.
    - El agrupamiento se hace de forma independiente por fase, lo que permite paralelizar sin dependencia de datos entre fases.

    Examples
    --------
    >>> mina_clusterizada = Clustering_parallel(mina_df, workers=4)
    >>> mina_clusterizada['cluster'].unique()
    array([0, 1, 2, ...])
    """
    
    options.setdefault('save', True)
    options.setdefault('path_save', 'Clusterizacion/Resultados/')
    save = options['save']
    path_save = options['path_save']

    t1 = time.time()
    fases = np.sort(mina['fase'].unique())
    lista_minas = [mina[mina['fase'] == f].copy() for f in fases]

    args_list = [(minimina, cm, cr, cp, P, R, alpha_ley_corte, options) for minimina in lista_minas]

    if workers == -1:
        workers = cpu_count()

    with Pool(processes=workers) as pool:
        resultados = pool.map(clustering_wrapper, args_list)

    minas, metricas, precedencias = map(list, zip(*resultados))
    mina_final = pd.concat(minas, ignore_index=True)
    metricas_final = pd.concat(metricas, ignore_index=True)
    precedencias_final = pd.concat(precedencias, ignore_index=True)
    # resultado_final = pd.concat(resultados, ignore_index=True)

    t2 = time.time()
    Tiempo_Clusterizacion = t2 - t1

    print(f'\nTiempo total de Clusterizacion: {Tiempo_Clusterizacion}')
    num_clusters = 0
    for f in fases:
        bancos = mina_final[mina_final['fase']==f]['banco'].unique()
        for b in bancos:
            num_clusters += len( mina_final[(mina_final['fase']==f) & (mina_final['banco']==b)]['cluster'].unique() )
    print(f'\nNumero total de Clusters creados: {num_clusters}')

    if save:
        path_arch = Path(path_save + 'mina_clusterizada.csv')
        path_arch.parent.mkdir(parents=True, exist_ok=True)
        mina_final.to_csv(path_arch, index=False)

        metricas_final.to_csv(Path(path_save + 'metricas.csv'), index=False)
        precedencias_final.to_csv(Path(path_save + 'precedencias.csv'), index=False)

        np.save(path_save + 'tiempo.npy', Tiempo_Clusterizacion)
        np.save(path_save + 'tiempo.npy', num_clusters)

    return mina_final



###############################################################
##################### Plotting Functions ######################
###############################################################

def plot_fase_banco(FaseBanco, column_hue='cut', text_hue=None, params=dict(), ax=None):
    """
    Dibuja en 2‑D los bloques de una *fase‑banco* minera y los colorea según
    una variable continua.

    La función valida la presencia de las columnas necesarias, admite múltiples
    elementos de anotación (flechas, sectores, elipses, precedencias) y puede
    guardar la figura en disco.

    Parameters
    ----------
    FaseBanco : pandas.DataFrame
        DataFrame con los bloques de **una sola** fase y banco.  
        Se requieren como mínimo las columnas:
        ``'x', 'y', 'z', 'fase', 'banco', column_hue``.  
        Si se destacan bloques, también se exige ``'id'``.

    column_hue : str, default 'cut'
        Columna numérica continua usada para el mapa de colores.

    text_hue : str or None, optional
        Columna cuyos valores se mostrarán dentro de cada bloque.
        Si es ``None`` se utiliza `column_hue`.

    params : dict, optional
        Opciones de dibujo. Claves más relevantes::
            BlockWidthX, BlockWidthY  : tamaño del bloque (default 10).
            xsize, ysize, dpi         : tamaño y resolución de la figura.
            cmap                      : colormap (default 'plasma').
            destacar_bloques          : set de IDs a resaltar.
            puntos, flechas, sectores, elipse
            precedencias, centros     : datos para flechas de precedencia.
            guardar_imagen (bool)     : si se guarda la figura.
            path                      : carpeta de guardado.
            show_block_label, show_grid, show_legend
            cluster_lw, arrow_lw, arrow_color,
            precedence_lw, precedence_cmap.

    ax : matplotlib.axes.Axes or None, optional
        Ejes sobre los que se dibuja. Si es ``None`` se crea uno nuevo.

    Returns
    -------
    matplotlib.figure.Figure
        Figura generada (devuelta también cuando se guarda en disco).

    Raises
    ------
    ValueError
        Si `FaseBanco` está vacío.

    KeyError
        Si falta alguna de las columnas obligatorias.

    Notes
    -----
    * Actualmente solo admite variables **continuas** para `column_hue`.
    * El archivo de salida se nombra como  
      ``fase_{fase}_banco_{banco}_hue_{column_hue}.png`` y se guarda en
      ``params['path']`` si ``guardar_imagen`` es True.
    * Devuelve la figura para permitir manipulaciones posteriores.

    Examples
    --------
    >>> fig = plot_fase_banco(df_fb, column_hue='cut',
    ...                       params={'destacar_bloques': {1, 42},
    ...                               'guardar_imagen': True,
    ...                               'path': 'salidas/'})
    """
    if FaseBanco.empty:
        raise ValueError("El DataFrame 'FaseBanco' está vacío. No se puede graficar.")

    params.setdefault('BlockWidthX', 10)
    params.setdefault('BlockWidthY', 10)
    params.setdefault('xsize', 10)
    params.setdefault('ysize', 10)
    params.setdefault('xlim', ())
    params.setdefault('ylim', ())
    params.setdefault('cmap', 'plasma')
    params.setdefault('show_block_label', True)
    params.setdefault('show_grid', True)
    params.setdefault('show_legend', True)
    params.setdefault('dpi', 100)
    params.setdefault('destacar_bloques', set())
    params.setdefault('puntos', [])
    params.setdefault('flechas', [])
    params.setdefault('sectores', [])
    params.setdefault('precedencias', dict())
    params.setdefault('centros', dict())
    params.setdefault('elipse', [])
    params.setdefault('guardar_imagen', False)
    params.setdefault('path', '')
    params.setdefault('cluster_lw', 2.5)
    params.setdefault('arrow_lw', 6)
    params.setdefault('arrow_color', 'blueviolet')
    params.setdefault('precedence_cmap', 'binary')
    params.setdefault('precedence_lw', 2.5)
    
    BlockWidthX = params['BlockWidthX']
    BlockWidthY = params['BlockWidthY']
    xsize = params['xsize']
    ysize = params['ysize']
    cmap = params['cmap']
    show_block_label = params['show_block_label']
    show_grid = params['show_grid']
    show_legend = params['show_legend']
    dpi = params['dpi']
    highlight_blocks = params['destacar_bloques']
    points = params['puntos']
    arrows = params['flechas']
    sectors = params['sectores']
    precedences = params['precedencias']
    centers = params['centros']
    elipse = params['elipse']
    save_as_image = params['guardar_imagen']
    path_to_save = params['path']
    cluster_lw = params['cluster_lw']
    xlim = params['xlim']
    ylim = params['ylim']
    arrow_lw = params['arrow_lw']
    arrow_color = params['arrow_color']

    required_cols = {'x', 'y', 'fase', 'z', 'banco', column_hue}

    if text_hue:
        required_cols.add(text_hue)

    if params.get('destacar_bloques'):
        required_cols.add('id')

    missing = required_cols - set(FaseBanco.columns)
    if missing:
        raise KeyError(
            f"Las siguientes columnas son obligatorias y no están en el DataFrame: {sorted(missing)}"
        )

    if not text_hue:
        text_hue = column_hue
    
    xsize=int(xsize)
    ysize=int(ysize)
    dpi=float(dpi)

    if ax is None:
        fig, ax = plt.subplots(figsize=(xsize, ysize), dpi=dpi)
    else:
        fig = ax.figure
        ax.clear()
    norm = None
    colormap = None
    variables_continuas = FaseBanco.select_dtypes(include=['float64', 'float32']).columns.tolist()
    
    fase = FaseBanco['fase'].values[0]
    z = FaseBanco['z'].values[0]
    banco = FaseBanco['banco'].values[0]

    col_data = FaseBanco[column_hue]

    vmin = np.min(col_data)
    vmax = np.max(col_data)

    if vmin == vmax: 
        vmax = vmin + 1e-6

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    for i, row in FaseBanco.iterrows():
        x_center = row['x']
        y_center = row['y']
        block_value = row[column_hue]

        x_corner = x_center - BlockWidthX / 2
        y_corner = y_center - BlockWidthY / 2

        color = colormap(norm(block_value))

        rect = patches.Rectangle((x_corner, y_corner), BlockWidthX, BlockWidthY, linewidth=0.5, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        if show_block_label:
            block_text = row[text_hue]
            if text_hue in variables_continuas:
                block_text = np.trunc(block_text*10)/10
            else:
                block_text = int(block_text)
            ax.text(x_center, y_center, str(block_text), ha='center', va='center', fontsize=8, color='black')
    

    x_vals = [FaseBanco['x'].min(), FaseBanco['x'].max()]
    y_vals = [FaseBanco['y'].min(), FaseBanco['y'].max()]

    for arrow in arrows:
        P1, P2 = arrow
        x_vals.extend([P1[0], P2[0]])
        y_vals.extend([P1[1], P2[1]])

    for p in points:
        x_vals.append(p[0])
        y_vals.append(p[1])

    margin_x = 5 * BlockWidthX
    margin_y = 5 * BlockWidthY

    
    x_min = min(x_vals) - margin_x
    x_max = max(x_vals) + margin_x
    y_min = min(y_vals) - margin_y
    y_max = max(y_vals) + margin_y

    if len(xlim)==0:
        ax.set_xlim(x_min, x_max)
    else:
        ax.set_xlim(xlim[0], xlim[1])
    
    if len(ylim)==0:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(ylim[0], ylim[1])

    if column_hue=='cluster':
        num_clusters = len(FaseBanco['cluster'].unique())
        ax.set_title(f'Fase {fase} - Banco (Z={z}) - Hue {column_hue} - N° Clusters {num_clusters}')
    else:
        ax.set_title(f'Fase {fase} - Banco (Z={z}) - Hue {column_hue}')
        
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    ax.grid(show_grid, color='gray', linestyle='--', linewidth=0.5)

    if len(highlight_blocks)>0:
        blocks = FaseBanco[FaseBanco['id'].isin(highlight_blocks)]
        x = blocks['x']
        y = blocks['y']       
        ax.plot(x, y, 'ro', markersize=10)

    for p in points:
        ax.plot(p[0], p[1], 'bo', markersize=5)

    for a in arrows:
        P1, P2 = a
        ax.annotate('', xy=P2, xytext=P1, arrowprops=dict(arrowstyle='-|>', color=arrow_color, lw=arrow_lw, mutation_scale=15))
    
    arrow_info = []
    for cluster_dst, lista_pre in precedences.items():
        for cluster_src in lista_pre:
            arrow_info.append((centers[cluster_src],
                            centers[cluster_dst],
                            cluster_src))

    cmap = plt.get_cmap(params['precedence_cmap'])
    clusters_src = sorted({info[2] for info in arrow_info}) 
    n_src = len(clusters_src)
    color_by_src = {c: cmap(i/(n_src-1 if n_src>1 else 1))
                    for i, c in enumerate(clusters_src)}

    for P_src, P_dst, cluster_src in arrow_info:
        ax.annotate(
            '', xy=P_dst, xytext=P_src,
            arrowprops=dict(
                arrowstyle='-|>',
                lw=params['precedence_lw'],
                color=color_by_src[cluster_src],
                mutation_scale=15
            )
        )

    sector_count = 1
    for s in sectors:
        P1, P2, P3, P4 = s
        center = ( (P1[0]+P3[0])/2, (P1[1]+P3[1])/2 )
        ax.annotate('', xy=P2, xytext=P1, arrowprops=dict(arrowstyle='-', color='black', lw=2, mutation_scale=5))
        ax.annotate('', xy=P3, xytext=P2, arrowprops=dict(arrowstyle='-', color='black', lw=2, mutation_scale=5))
        ax.annotate('', xy=P4, xytext=P3, arrowprops=dict(arrowstyle='-', color='black', lw=2, mutation_scale=5))
        ax.annotate('', xy=P1, xytext=P4, arrowprops=dict(arrowstyle='-', color='black', lw=2, mutation_scale=5))
        ax.annotate(str(sector_count), xy=center, color='black', fontsize=32)
        sector_count+=1
    
    for el in elipse:
        a, b, alpha, x_centro, y_centro = el
        Theta = np.linspace(0, 2*np.pi, 100)
        X_elipse = a*np.cos(Theta)*np.cos(alpha) - b*np.sin(Theta)*np.sin(alpha) + x_centro
        Y_elipse = a*np.cos(Theta)*np.sin(alpha) + b*np.sin(Theta)*np.cos(alpha) + y_centro

        ax.plot(X_elipse, Y_elipse, color='black')


    if column_hue == 'cluster':
        vecinos = {
            (-BlockWidthX, 0): lambda xc,yc: ([xc-BlockWidthX/2]*2,
                                            [yc-BlockWidthY/2, yc+BlockWidthY/2]),
            ( BlockWidthX, 0): lambda xc,yc: ([xc+BlockWidthX/2]*2,
                                            [yc-BlockWidthY/2, yc+BlockWidthY/2]),
            (0,-BlockWidthY): lambda xc,yc: ([xc-BlockWidthX/2, xc+BlockWidthX/2],
                                            [yc-BlockWidthY/2]*2),
            (0, BlockWidthY): lambda xc,yc: ([xc-BlockWidthX/2, xc+BlockWidthX/2],
                                            [yc+BlockWidthY/2]*2),
        }

        lookup = {(r.x, r.y): r for _, r in FaseBanco.iterrows()}

        for (xc,yc), row in lookup.items():
            clu = row[column_hue]
            for (dx,dy), segment in vecinos.items():
                vecino = lookup.get((xc+dx, yc+dy))
                if vecino is None or vecino[column_hue] != clu:
                    xs, ys = segment(xc,yc)
                    ax.plot(xs, ys, color='black', lw=cluster_lw)

    if show_legend:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label(column_hue, rotation=270, labelpad=15)

    plt.tight_layout()

    if save_as_image:
        filename = f'fase_{fase}_banco_{banco}_hue_{column_hue}.png'
        if path_to_save == '':
            path_to_save = filename
        else:
            path_to_save = os.path.join(path_to_save, filename)

        os.makedirs(os.path.dirname(path_to_save) or '.', exist_ok=True)
        fig.savefig(path_to_save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_mina_3D(mina, column_hue='tipomineral', params=dict()):
    """
    Visualiza en 3D la mina y elementos asociados como conos, elipses, curvas, puntos y bloques destacados.

    Utiliza Plotly para crear una figura interactiva que muestra los bloques de la mina coloreados por una propiedad
    (column_hue) y permite agregar otros elementos opcionales como conos, curvas (rampas), puntos y bloques destacados.

    Parámetros
    ----------
    mina : pandas.DataFrame
        DataFrame con columnas 'x', 'y', 'z', 'fase', 'id' y la columna especificada en `column_hue`.

    column_hue : str, opcional
        Columna del DataFrame `mina` utilizada para colorear los bloques. Por defecto es 'tipomineral'.

    params : dict, opcional
        Diccionario con parámetros adicionales para la visualización. Claves posibles:
            - 'puntos': lista de puntos (x, y, z) a marcar.
            - 'curvas': lista de curvas, cada una como (X_curve, Y_curve, Z_curve).
            - 'elipses': lista de elipses, cada una como (a, b, x_c, alpha, y_c, z_c).
            - 'conos': lista de conos, cada uno como (a, b, h, alpha, x_c, y_c, z_sup).
            - 'id_bloques': set de IDs de bloques a resaltar en rojo.
            - 'mina_rellena': DataFrame alternativo con bloques adicionales.
            - 'opacity_blocks': float entre 0 y 1 para la opacidad de los bloques.
            - 'z_ratio': proporción entre el eje z y los ejes x, y.
            - 'width', 'height': dimensiones del gráfico.

    Retorna
    -------
    None
        Muestra un gráfico interactivo en Plotly.
    """
    fig = go.Figure()

    params.setdefault('puntos', [])
    params.setdefault('curvas', [])
    params.setdefault('elipses', [])
    params.setdefault('conos', [])
    params.setdefault('id_bloques', set())
    params.setdefault('mina_rellena', pd.DataFrame())
    params.setdefault('opacity_blocks', 1)
    params.setdefault('z_ratio', 1)
    params.setdefault('width', 900)
    params.setdefault('height', 800)

    puntos = params['puntos']
    curva = params['curvas']
    elipses = params['elipses']
    cono = params['conos']
    id_bloques = params['id_bloques']
    mina_rellena = params['mina_rellena']
    opacity_blocks = params['opacity_blocks']
    z_ratio = params['z_ratio']
    width = params['width']
    height = params['height']


    if not mina.empty:
        required_columns = ['x', 'y', 'z', 'fase', 'id', column_hue]
        if not all(col in mina.columns for col in required_columns):
            raise ValueError("Faltan columnas requeridas en el DataFrame 'mina'.")
        
        fases_mina = sorted(mina['fase'].unique())
        for f in fases_mina:
            mini_mina = mina[mina['fase']==f]
            fig.add_trace(go.Scatter3d(
                x=mini_mina['x'],
                y=mini_mina['y'],
                z=mini_mina['z'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=mini_mina[column_hue],
                    colorscale='rainbow',
                    colorbar=dict(title=column_hue,
                                x=1.15,
                                y=0.5,
                                len=0.75),
                    opacity=opacity_blocks
                ),
                hovertemplate=(
                    "ID: %{customdata[0]}<br>" +
                    f"{column_hue}:"+"%{marker.color:.3f}<br>"
                ),
                customdata=mini_mina[['id', column_hue]],
                name=f'Bloques fase {f}'
            ))

    theta = np.linspace(0, 2*np.pi, 50)
    i = 1
    for elipse in elipses:
        a, b, x_centro, alpha, y_centro, z_centro = elipse
        X_elipse = a*np.cos(theta)*np.cos(alpha) - b*np.sin(theta)*np.sin(alpha) + x_centro
        Y_elipse = a*np.cos(theta)*np.sin(alpha) + b*np.sin(theta)*np.cos(alpha) + y_centro
        Z_elipse = np.full_like(theta, z_centro)

        fig.add_trace(go.Scatter3d(
                x = X_elipse, y = Y_elipse, z = Z_elipse,
                mode='lines',
                line=dict(color='black', width=2),
                name=f'Elipse {i}'
            ))
        i += 1

    colors = ['Viridis', 'Turbo', 'Hot', 'Ice', 'Plasma', 'Agsunset', 'Mint', 'Emrld']
    if len(cono)>0:
        cono_indices = []
        # if isinstance(cono, tuple) and len(cono)==7:
        #     a, b, h, x_centro, y_centro, alpha, z_sup = cono
        #     Z = np.linspace(z_sup - h, z_sup, 50)
        #     Theta, Z = np.meshgrid(Theta, Z)
        #     X_cono = ((a/h)*np.cos(Theta)*np.cos(alpha) - (b/h)*np.sin(Theta)*np.sin(alpha))*(h-z_sup+Z) + x_centro
        #     Y_cono = ((a/h)*np.cos(Theta)*np.sin(alpha) + (b/h)*np.sin(Theta)*np.cos(alpha))*(h-z_sup+Z) + y_centro
            
        #     fig.add_trace(go.Surface(
        #         x=X_cono,
        #         y=Y_cono,
        #         z=Z,
        #         colorscale='viridis',
        #         name='Cono',
        #         opacity=0.8,
        #         showscale=False
        #     ))
        #     cono_indices.append((1, len(fig.data)-1))
        # else:
        i = 1
        for a, b, h, alpha, x_centro, y_centro, z_sup in cono:
            Z = np.linspace(z_sup - h, z_sup, 50)
            Theta, Z = np.meshgrid(Theta, Z)
            X_cono = ((a/h)*np.cos(Theta)*np.cos(alpha) - (b/h)*np.sin(Theta)*np.sin(alpha))*(h-z_sup+Z) + x_centro
            Y_cono = ((a/h)*np.cos(Theta)*np.sin(alpha) + (b/h)*np.sin(Theta)*np.cos(alpha))*(h-z_sup+Z) + y_centro
            
            fig.add_trace(go.Surface(
                x=X_cono,
                y=Y_cono,
                z=Z,
                colorscale=colors[(i - 1) % len(colors)],
                name=f'Cono {i}',
                opacity=0.8,
                showscale=False
            ))
            cono_indices.append((i, len(fig.data)-1))
            i+=1

    if len(curva)>0:
        colors = ['black', 'red', 'green', 'blue']
        # if isinstance(curva, list) and len(curva)==3:
        #     X_curve, Y_curve, Z_curve = curva
        #     fig.add_trace(go.Scatter3d(
        #         x=X_curve, y=Y_curve, z=Z_curve,
        #         mode='lines',
        #         line=dict(color='black', width=5),
        #         name='Rampa'
        #     ))
        # else:
        i=1
        for X_curve, Y_curve, Z_curve in curva:
            fig.add_trace(go.Scatter3d(
                x=X_curve, y=Y_curve, z=Z_curve,
                mode='lines',
                line=dict(color=colors[(i-1)%len(colors)], width=5),
                name=f'Rampa {i}'
            ))
            i+=1
    
    if len(puntos)>0:
        X = [p[0] for p in puntos]
        Y = [p[1] for p in puntos]
        Z = [p[2] for p in puntos]
        fig.add_trace(go.Scatter3d(
            x=X, y=Y, z=Z,
            mode='markers',
            marker=dict(color='red', size=10),
            name='Puntos Iniciales',
            opacity=opacity_blocks**4
        ))
    
    if len(id_bloques)>0:
        if not mina_rellena.empty:
            bloques = mina_rellena[mina_rellena['id'].isin(id_bloques)]
            X = list(bloques['x'])
            Y = list(bloques['y'])
            Z = list(bloques['z'])
            fig.add_trace(go.Scatter3d(
                x=X, y=Y, z=Z,
                mode='markers',
                marker=dict(color='red', size=1),
                name='Bloques destacados',
                opacity=opacity_blocks**(1/2),
                hovertemplate=(
                "ID: %{customdata[0]}<br>"
            ),
            customdata=bloques['id'],
            ))
        else:
            bloques = mina[mina['id'].isin(id_bloques)]
            if not bloques.empty:
                X = list(bloques['x'])
                Y = list(bloques['y'])
                Z = list(bloques['z'])
                fig.add_trace(go.Scatter3d(
                    x=X, y=Y, z=Z,
                    mode='markers',
                    marker=dict(color='red', size=1),
                    name='Bloques destacados',
                    opacity=opacity_blocks**(1/2)
                ))

    if not mina.empty:
        x_all = mina['x']
        y_all = mina['y']
        z_all = mina['z']

        xrange = x_all.max() - x_all.min()
        yrange = y_all.max() - y_all.min()
        zrange = z_all.max() - z_all.min()
        max_range = max(xrange, yrange, zrange)

        aspectratio = dict(
            x=xrange / max_range,
            y=yrange / max_range,
            z=(zrange / max_range)*z_ratio
        )

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='manual',
                aspectratio=aspectratio
            ),
            title=f'Puntos 3D con Color según {column_hue}',
            width=width,
            height=height
        )

    
    buttons = []
    if len(cono)>0:
        # Mostrar todo
        buttons.append(dict(
            label='Mostrar todo',
            method='update',
            args=[{'visible': [True] * len(fig.data)}]
        ))

        # Ocultar todos los conos
        visible_hide_all_cones = [True] * len(fig.data)
        for _, idx in cono_indices:
            visible_hide_all_cones[idx] = False

        buttons.append(dict(
            label='Ocultar todos los conos',
            method='update',
            args=[{'visible': visible_hide_all_cones}]
        ))

        # Botones por cono individual
        for i, idx in cono_indices:
            visible_show_i = [True] * len(fig.data)
            for j, other_idx in cono_indices:
                if j != i:
                    visible_show_i[other_idx] = False


            buttons.append(dict(
                label=f'Mostrar Cono {i}',
                method='update',
                args=[{'visible': visible_show_i}]
            ))

    # Añadir menú
    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            showactive=True,
            x=0.15,
            y=1,
            xanchor='left',
            yanchor='top',
            buttons=buttons
        )]
    )

    fig.show()


def plot_precedencias_verticales_3D(
        fases_bancos, column_hue='cluster', 
        precedence_option=True, arcs={}, 
        width=1000, height=800, z_ratio=5):
    """
    Gráfico 3D de dos fases-banco con relaciones de precedencia vertical entre clusters.

    Esta función visualiza dos capas (fase-banco inferior y superior) en 3D con Plotly, y opcionalmente
    dibuja flechas (arcos) entre clusters relacionados según una estructura de precedencias verticales.

    Parámetros
    ----------
    fases_bancos : list of pandas.DataFrame
        Lista con dos DataFrames: [fase_banco_lower, fase_banco_upper], representando capas consecutivas.

    column_hue : str, opcional
        Nombre de la columna utilizada para colorear los bloques. Por defecto 'cluster'.

    precedence_option : bool, opcional
        Si es True, se espera una estructura `arcs` que define las precedencias verticales.

    arcs : dict, opcional
        Diccionario generado por `Global_Vertical_Arc_Calculation`, donde las llaves son tuplas (fase, banco, cluster_id)
        y los valores son listas de tuplas similares que indican dependencias.

    width : int, opcional
        Ancho del gráfico. Por defecto 1000.

    height : int, opcional
        Alto del gráfico. Por defecto 800.

    z_ratio : float, opcional
        Relación entre el eje Z y los ejes X,Y. Útil para enfatizar la profundidad vertical. Por defecto 5.

    Retorna
    -------
    None
        Muestra un gráfico interactivo en Plotly.
    """

    fase_banco_lower, fase_banco_upper = fases_bancos
    banco_lower = fase_banco_lower['banco'].values[0]
    banco_upper = fase_banco_upper['banco'].values[0]

    fase = fase_banco_lower['fase'].values[0]

    if precedence_option:
        if not arcs:
            raise Exception('Debe adjuntar arcos si precedence_option=True')

        arcs_lower = [k for k in arcs.keys() if k[0] == fase and k[1] == banco_lower]

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=fase_banco_lower['x'],
            y=fase_banco_lower['y'],
            z=fase_banco_lower['z'],
            mode='markers',
            marker=dict(
                size=5,
                color=fase_banco_lower[column_hue],
                colorscale='rainbow',
                cmin=fase_banco_lower[column_hue].min(),
                cmax=fase_banco_lower[column_hue].max(),
                opacity=0.9
            ),
            name='Capa inferior',
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter3d(
            x=fase_banco_upper['x'],
            y=fase_banco_upper['y'],
            z=fase_banco_upper['z'],
            mode='markers',
            marker=dict(
                size=5,
                color=fase_banco_upper[column_hue],
                colorscale='twilight',
                cmin=fase_banco_upper[column_hue].min(),
                cmax=fase_banco_upper[column_hue].max(),
                opacity=0.9
            ),
            name='Capa superior',
            hoverinfo='skip'
        ))

        buttons = []
        trace_index = 2
        Centros_lower = Centros_Clusters(fase_banco_lower)
        Centros_upper = Centros_Clusters(fase_banco_upper)
        z_lower = fase_banco_lower['z'].values[0]
        z_upper = fase_banco_upper['z'].values[0]

        for i, (cluster_id, (x,y)) in enumerate(Centros_lower.items()):
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z_lower],
                mode='markers',
                marker=dict(
                    size=10,
                    color='black'
                ),
                customdata=[[cluster_id, z_lower]],
                hovertemplate='Cluster: %{customdata[0]}<br>Z: %{customdata[1]}<extra></extra>'
            ))
            trace_index+=1

        for i, (cluster_id, (x,y)) in enumerate(Centros_upper.items()):
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z_upper],
                mode='markers',
                marker=dict(
                    size=10,
                    color='black'
                ),
                customdata=[[cluster_id, z_upper]],
                hovertemplate='Cluster: %{customdata[0]}<br>Z: %{customdata[1]}<extra></extra>'
            ))
            trace_index+=1
        
        arrow_traces = []

        for cluster in arcs_lower:
            source_id = cluster[2]
            for c in arcs[cluster]:
                destiny_id = c[2]
                fig.add_trace(go.Scatter3d(
                    x=[Centros_lower[source_id][0], Centros_upper[destiny_id][0]],
                    y=[Centros_lower[source_id][1], Centros_upper[destiny_id][1]],
                    z=[z_lower, z_upper],
                    mode='lines',
                    line=dict(color='black', width=4),
                    showlegend=False,
                    hoverinfo='skip',
                    visible=True
                ))
                trace_index+=1

                fig.add_trace(go.Scatter3d(
                    x=[Centros_upper[destiny_id][0]],
                    y=[Centros_upper[destiny_id][1]],
                    z=[z_upper],
                    mode='markers',
                    marker=dict(size=4, color='red'),
                    showlegend=False,
                    hoverinfo='skip',
                    visible=True
                ))
                trace_index +=1
                arrow_traces.append((source_id, trace_index-2, trace_index-1))
        
        all_visible = [True]*trace_index
        buttons.append(dict(
            label='Mostrar todos',
            method='update',
            args=[{'visible': all_visible}]
        ))

        unique_sources = sorted(set([sid for sid, _, _ in arrow_traces]))
        for sid in unique_sources:
            visibles = [True] * 2
            visibles += [True] * len(Centros_lower)
            visibles += [True] * len(Centros_upper)
            for s, line_idx, point_idx in arrow_traces:
                if s == sid:
                    visibles += [True, True]
                else:
                    visibles += [False, False]
            
            buttons.append(dict(
                label=f'Centro {sid}',
                method='update',
                args=[{'visible': visibles}]
            ))
        
        fig.update_layout(
            updatemenus=[dict(
                buttons=buttons,
                direction='down',
                showactive=True,
                x=1.1,
                y=0.8
            )]
        )
    
    else:
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x = fase_banco_lower['x'],
            y = fase_banco_lower['y'],
            z = fase_banco_lower['z'],
            mode='markers',
            marker=dict(
                size=5,
                color=fase_banco_lower[column_hue],
                colorscale='rainbow',
                cmin=fase_banco_lower[column_hue].min(),
                cmax=fase_banco_lower[column_hue].max(),
                opacity=0.9
            ),
            name='Capa inferior',
            hovertemplate='Z: %{z}<br>'+f'{column_hue}:'+'%{marker.color:.3f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter3d(
            x = fase_banco_upper['x'],
            y = fase_banco_upper['y'],
            z = fase_banco_upper['z'],
            mode='markers',
            marker=dict(
                size=5,
                color=fase_banco_upper[column_hue],
                colorscale='twilight',
                cmin=fase_banco_upper[column_hue].min(),
                cmax=fase_banco_upper[column_hue].max(),
                opacity=0.9
            ),
            name='Capa superior',
            hovertemplate='Z: %{z}<br>'+f'{column_hue}:'+'%{marker.color:.3f}<extra></extra>'
        ))

    x_all = pd.concat([fase_banco_lower['x'], fase_banco_upper['x']])
    y_all = pd.concat([fase_banco_lower['y'], fase_banco_upper['y']])
    z_all = pd.concat([fase_banco_lower['z'], fase_banco_upper['z']])

    xrange = x_all.max() - x_all.min()
    yrange = y_all.max() - y_all.min()
    zrange = z_all.max() - z_all.min()
    max_range = max(xrange, yrange, zrange)

    aspectratio = dict(
        x=xrange / max_range,
        y=yrange / max_range,
        z=(zrange / max_range)*z_ratio
    )
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='manual',
            aspectratio=aspectratio
        ),
        title=f'Fases bancos up:{banco_upper} - down:{banco_lower}',
        width=width,
        height=height
    )
    fig.show()



######################################################
################# Metrics Functions ##################
######################################################

def Rock_Unity(FaseBanco):
    """
    Calcula la homogeneidad del tipo de roca en cada cluster de la fase-banco.

    El Rock Unity (RU) mide la proporción del tipo de mineral predominante dentro de cada cluster.
    La función devuelve el promedio de RU entre todos los clusters y la distribución (lista)
    de RU para cada cluster individualmente.

    Parámetros
    ----------
    FaseBanco : pandas.DataFrame
        DataFrame que debe contener al menos las columnas 'cluster' y 'tipomineral'.
        Cada fila representa un bloque con su asignación a cluster y tipo de mineral.

    Retorna
    -------
    RU : float
        Valor promedio del Rock Unity entre todos los clusters.
        Si no existe la columna 'tipomineral', devuelve 1.

    RU_distribution : list de floats
        Lista con el Rock Unity calculado para cada cluster individual.
        Si no existe la columna 'tipomineral', devuelve [1].

    """
    if 'tipomineral' in FaseBanco.columns:
        ID_Clusters = np.sort(FaseBanco['cluster'].unique())
        num_clusters = len(ID_Clusters)
        sum_rock_unity = 0
        RU_distribution = []

        for id in ID_Clusters:
            Cluster = FaseBanco.loc[FaseBanco['cluster']==id]
            n_cluster = len(Cluster)
            max_rock = Cluster['tipomineral'].value_counts().max()
            ru = max_rock/n_cluster
            sum_rock_unity += ru
            RU_distribution.append(ru)

        RU = sum_rock_unity/num_clusters
    else:
        RU = 1
        RU_distribution = [1]
    return RU, RU_distribution


def Destination_Dilution_Factor(FaseBanco):
    """
    Calcula la homogeneidad del destino de los clusters de la fase-banco, ponderada por el tonelaje de cada bloque.

    El Destination Dilution Factor (DDF) mide la proporción del tonelaje del destino más frecuente dentro de cada cluster
    respecto al tonelaje total del cluster. La función devuelve el promedio del DDF entre todos los clusters y la distribución
    (lista) de DDF para cada cluster.

    Parámetros
    ----------
    FaseBanco : pandas.DataFrame
        DataFrame que debe contener al menos las columnas 'cluster', 'destino' y 'density'.
        Cada fila representa un bloque con su asignación a cluster, destino y tonelaje (density).

    Retorna
    -------
    DDF : float
        Valor promedio del Destination Dilution Factor entre todos los clusters.

    DDF_distribution : list de floats
        Lista con el DDF calculado para cada cluster individual.

    """
    ID_Clusters = np.sort(FaseBanco['cluster'].unique())
    num_clusters = len(ID_Clusters)
    sum_ddf = 0
    DDF_distribution = []

    for id in ID_Clusters:
        Cluster = FaseBanco.loc[FaseBanco['cluster']==id]
        max_destino = Cluster['destino'].value_counts().idxmax()
        ton_destino = Cluster.loc[Cluster['destino']==max_destino]['density'].sum()
        ton_total = Cluster['density'].sum()
        ddf = ton_destino/ton_total
        sum_ddf += ddf
        DDF_distribution.append(ddf)
    
    DDF = sum_ddf/num_clusters
    return DDF, DDF_distribution


def Coefficient_Variation(FaseBanco):
    """
    Calcula el coeficiente de variación (CV) de la ley (cut) para cada cluster en la fase-banco.

    El CV representa la desviación estándar relativa respecto a la media de la ley de los bloques dentro de cada cluster.
    La función devuelve el promedio del CV entre todos los clusters y la distribución (lista) de CV para cada cluster individual.

    Parámetros
    ----------
    FaseBanco : pandas.DataFrame
        DataFrame que debe contener las columnas 'cluster' y 'cut'.
        Cada fila representa un bloque con su asignación a cluster y su ley (cut).

    Retorna
    -------
    CV : float
        Promedio del coeficiente de variación de la ley en todos los clusters.

    CV_distribution : list de floats
        Lista con el coeficiente de variación calculado para cada cluster individual.

    """
    ID_Clusters = np.sort(FaseBanco['cluster'].unique())
    num_clusters = len(ID_Clusters)
    sum_cv = 0
    CV_distribution = []

    for id in ID_Clusters:
        Cluster = FaseBanco.loc[FaseBanco['cluster']==id]

        if len(Cluster)==1:
            cv = 0

        else:
            std = Cluster['cut'].std()
            mean = Cluster['cut'].mean()

            if mean == 0:
                cv = 0
            else:
                cv = std/mean

        sum_cv += cv
        CV_distribution.append(cv)

    CV = sum_cv/num_clusters

    return CV, CV_distribution



###############################################################
################# Shape Refinement Functions ##################
###############################################################

def Find_Corner_Blocks_Tabesh(FaseBanco, AdjencyMatrix):
    """
    Identifica los bloques esquina en la fase-banco según el criterio de Tabesh (2013).

    Un bloque esquina es aquel que tiene a lo sumo un vecino en el mismo cluster y más de un vecino en otros clusters,
    con al menos un cluster vecino repetido.

    Parámetros
    ----------
    FaseBanco : pandas.DataFrame
        DataFrame que contiene al menos las columnas 'id' y 'cluster'.
    
    AdjencyMatrix : scipy.sparse matrix
        Matriz de adyacencia que indica vecindad entre bloques en la fase-banco.

    Retorna
    -------
    corner_blocks : dict
        Diccionario donde las llaves son los IDs de los bloques esquina y los valores son arrays numpy
        con los IDs de los clusters vecinos, ordenados.

    Notas
    -----
    Esta función es utilizada internamente en Shape_Refinement_Tabesh.
    """
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
            corner_blocks.update({FaseBanco.iloc[i]['id']: np.sort(Clusters_Vecinos)})

    return corner_blocks


def Shape_Refinement_Tabesh(FaseBanco, AdjencyMatrix, Min_Cluster_Length = 10, Iterations_PostProcessing=5, Reset_Clusters_Index=False):
    """
    Identifica los bloques esquina en la fase-banco según el criterio de Tabesh (2013).

    Un bloque esquina es aquel que tiene a lo sumo un vecino en el mismo cluster y más de un vecino en otros clusters,
    con al menos un cluster vecino repetido.

    Parámetros
    ----------
    FaseBanco : pandas.DataFrame
    DataFrame que contiene al menos las columnas 'id' y 'cluster'.

    AdjencyMatrix : scipy.sparse matrix
    Matriz de adyacencia que indica vecindad entre bloques en la fase-banco.

    Retorna
    -------
    corner_blocks : dict
    Diccionario donde las llaves son los IDs de los bloques esquina y los valores son arrays numpy
    con los IDs de los clusters vecinos, ordenados.

    Notas
    -----
    Esta función es utilizada internamente en Shape_Refinement_Tabesh.
    """
    fase_banco = FaseBanco.copy()
    execution_time = 0
    ID_Small_Clusters = list(fase_banco['cluster'].value_counts().loc[fase_banco['cluster'].value_counts() < Min_Cluster_Length].index)
    max_i_cluster = fase_banco['cluster'].max() + 1
    t1 = time.time()

    for i in ID_Small_Clusters:
        blocks = list(fase_banco.loc[fase_banco['cluster'] == i, 'id'])

        for j in blocks:
            fase_banco.loc[fase_banco['id']==j, 'cluster'] = max_i_cluster
            max_i_cluster += 1

    for iterations in range(Iterations_PostProcessing):
        Corner_Blocks = Find_Corner_Blocks_Tabesh(fase_banco, AdjencyMatrix)

        if len(Corner_Blocks) == 0:
            break

        for i in Corner_Blocks.keys():
            if len(Corner_Blocks[i]) == 2:
                Cluster_to_insert = Corner_Blocks[i][0]
            else:
                # Cluster_to_insert = np.unique(array, return_counts=True)
                # Cluster_to_insert = np.unique_counts(Corner_Blocks[i]).values[np.unique_counts(Corner_Blocks[i]).counts.argmax()]
                vals, counts = np.unique(Corner_Blocks[i], return_counts=True)
                Cluster_to_insert = vals[np.argmax(counts)]

            fase_banco.loc[fase_banco['id']==i, 'cluster'] = Cluster_to_insert

    t2 = time.time()
    execution_time = t2-t1
    N_Clusters = len(fase_banco['cluster'].unique())

    if Reset_Clusters_Index:
        fase_banco['cluster'] = fase_banco['cluster'].map(lambda x: np.array(range(1,N_Clusters+1))[np.where(fase_banco['cluster'].unique() == x)[0][0]] if x in fase_banco['cluster'].unique() else x)

    print(f"========PostProcessing Results========")
    print(f"Total de clusters: {N_Clusters}")
    print(f'Tiempo: {execution_time}')

    return fase_banco, execution_time, N_Clusters


def Shape_Refinement_Mod(FaseBanco, AdjencyMatrix, Min_Cluster_Length = 10, Iterations_PostProcessing=5, Reset_Clusters_Index=False):
    '''
    Refinamiento modificado de clusters en la fase-banco basado en Tabesh (2013) con criterios adicionales.

    Parámetros
    ----------
    FaseBanco : pandas.DataFrame
        DataFrame que contiene información de los bloques, incluyendo 'cluster' y 'destino'.

    AdjencyMatrix : scipy.sparse matrix
        Matriz de adyacencia que indica vecindad entre bloques.

    Min_Cluster_Length : int, opcional
        Tamaño mínimo para considerar un cluster como válido. Clusters más pequeños se separan en clusters individuales.

    Iterations_PostProcessing : int, opcional
        Número de iteraciones del proceso de refinamiento secuencial.

    Reset_Clusters_Index : bool, opcional
        Si es True, reasigna índices de clusters a un rango consecutivo.

    Retorna
    -------
    fase_banco : pandas.DataFrame
        DataFrame modificado con clusters refinados.

    execution_time : float
        Tiempo total de ejecución del refinamiento.

    N_Clusters : int
        Número total de clusters después del refinamiento.
    '''
    fase_banco = FaseBanco.copy().reset_index()
    execution_time = 0
    ID_Small_Clusters = fase_banco['cluster'].value_counts().loc[fase_banco['cluster'].value_counts() < Min_Cluster_Length].index.tolist()
    max_i_cluster = fase_banco['cluster'].max() + 1
    t1 = time.time()

    for i in ID_Small_Clusters:
        blocks = fase_banco.loc[fase_banco['cluster'] == i, 'cluster'].index
        for j in blocks:
            fase_banco.loc[j, 'cluster'] = max_i_cluster
            max_i_cluster += 1

    for iterations in range(Iterations_PostProcessing):
        rows, cols = AdjencyMatrix.nonzero()
        Clusters = np.sort(fase_banco['cluster'].unique())

        for cluster in Clusters:
            Blocks = fase_banco.loc[fase_banco['cluster'] == cluster].index

            for b in Blocks:
                Mismo_Cluster = 0
                Distinto_Cluster = 0
                Clusters_Vecinos = []
                b_is_row = np.where(rows == b)[0]

                for j in b_is_row:
                    if fase_banco.iloc[b]['cluster'] == fase_banco.iloc[cols[j]]['cluster']:
                        Mismo_Cluster += 1
                    else:
                        Distinto_Cluster += 1
                        Clusters_Vecinos.append(fase_banco.iloc[cols[j]]['cluster'].astype(int))

                if Mismo_Cluster <=1:
                    # Criterio Tabesh
                    if (Distinto_Cluster >= 2) and len(np.unique(Clusters_Vecinos)) < len(Clusters_Vecinos):
                        vals, counts = np.unique(Clusters_Vecinos, return_counts=True)
                        Cluster_to_insert = vals[np.argmax(counts)]
                        fase_banco.loc[b, 'cluster'] = Cluster_to_insert

                    # Criterio Modificado. Considera los casos en que el bloque tiene 3 vecinos de distintos clusters o sólo 1 vecino de otro cluster.
                    elif ((Distinto_Cluster == 3) and (len(np.unique(Clusters_Vecinos)) == len(Clusters_Vecinos))) or ((Distinto_Cluster == 1)):
                        choose = 0
                        Cluster_to_insert = fase_banco.iloc[b]['cluster']
                        disim = np.inf
                        for cluster_v in np.unique(Clusters_Vecinos):
                            destino_medio = fase_banco.loc[fase_banco['cluster']==cluster_v]['destino'].mean()
                            if np.abs(destino_medio - fase_banco.iloc[b]['destino']) < disim:
                                choose = cluster_v
                                disim = np.abs(destino_medio - fase_banco.iloc[b]['destino'])
                        if disim < 0.5:
                            Cluster_to_insert = choose
                        fase_banco.loc[b, 'cluster'] = Cluster_to_insert


    t2 = time.time()
    execution_time = t2-t1
    N_Clusters = len(fase_banco['cluster'].unique())
    if Reset_Clusters_Index:
        fase_banco['cluster'] = fase_banco['cluster'].map(lambda x: np.array(range(1,N_Clusters+1))[np.where(fase_banco['cluster'].unique() == x)[0][0]] if x in fase_banco['cluster'].unique() else x)
    print(f"========PostProcessing Results========")
    print(f"Total de clusters: {N_Clusters}")
    print(f'Tiempo: {execution_time}')
    
    fase_banco = fase_banco.set_index('index')
    fase_banco.index.name = None

    return fase_banco, execution_time, N_Clusters


def animate_fase_banco(history, column_hue='cluster', params=dict(), save_path=None, fps=2, real_steps=None):
    """
    Anima la evolución de una clusterización en una fase-banco.

    Parámetros
    ----------
    history : list of pandas.DataFrame
        Lista de DataFrames con la evolución del clusterizado en cada paso.
    column_hue : str, opcional
        Nombre de la columna para colorear los clusters. Por defecto 'cluster'.
    params : dict, opcional
        Parámetros para la función de graficado `plot_fase_banco`. Se añaden valores por defecto.
    save_path : str o None, opcional
        Ruta para guardar la animación. Si es None, se muestra la animación en pantalla.
    fps : int, opcional
        Fotogramas por segundo de la animación guardada. Por defecto 2.
    real_steps : list o None, opcional
        Lista con los números reales de pasos para mostrar en el título. Si None, usa índice + 1.

    Retorna
    -------
    matplotlib.animation.FuncAnimation
        Objeto de la animación generada.
    """
    if not history:
        raise ValueError("La historia de clusterización está vacía.")
    
    plot_params = deepcopy(params)
    plot_params.setdefault('xsize', 10)
    plot_params.setdefault('ysize', 10)
    plot_params.setdefault('dpi', 100)
    plot_params['guardar_imagen'] = False  # Nunca guardar dentro de plot

    fig, ax = plt.subplots(figsize=(plot_params['xsize'], plot_params['ysize']), dpi=plot_params['dpi'])

    def update(frame):
        ax.clear()
        df = history[frame]
        plot_fase_banco(df, column_hue=column_hue, text_hue=None, params=plot_params, ax=ax)

        paso = real_steps[frame] if real_steps else frame + 1
        ax.set_title(f'Paso {paso} - Clusters: {df[column_hue].nunique()}')

        return ax,

    ani = FuncAnimation(fig, update, frames=len(history), blit=False, repeat=False)

    if save_path:
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps)
        else:
            raise ValueError("El archivo debe tener extensión '.mp4' o '.gif'")
    else:
        plt.show()

    return ani



###########################################################################################
################# Automatic Creation of Horizontal Cluster's Precedences ##################
###########################################################################################

def Clusters_Vecinos(FaseBanco, Cluster, AdjencyMatrix):
    """
    Obtiene los clusters vecinos de un cluster dado en la fase-banco, según la matriz de adyacencia.

    Parámetros
    ----------
    FaseBanco : pandas.DataFrame
        DataFrame con las columnas 'cluster' y el índice correspondiente a los bloques.

    Cluster : int
        ID del cluster para el cual se buscan clusters vecinos.

    AdjencyMatrix : scipy.sparse matrix
        Matriz de adyacencia que indica vecindad entre bloques.

    Retorna
    -------
    clusters_vecinos : list of int
        Lista (posiblemente con repeticiones) de IDs de clusters vecinos distintos al actual.
    """
    fase_banco = FaseBanco.copy().reset_index()

    Blocks = fase_banco.loc[fase_banco['cluster'] == Cluster].index
    rows, cols = AdjencyMatrix.nonzero()

    clusters_vecinos = []
    for b in Blocks:
            b_is_row = np.where(rows == b)[0]
            for j in b_is_row:
                if fase_banco.iloc[b]['cluster'] != fase_banco.iloc[cols[j]]['cluster']:
                    clusters_vecinos.append(fase_banco.iloc[cols[j]]['cluster'].astype(int))

    return clusters_vecinos


def Centros_Clusters(FaseBanco):
    """
    Calcula el centroide de cada cluster en la fase-banco.

    Parámetros
    ----------
    FaseBanco : pandas.DataFrame
        DataFrame que debe contener al menos las columnas 'cluster', 'x' y 'y'.

    Retorna
    -------
    Centers : dict
        Diccionario donde las llaves son los IDs de los clusters y los valores son tuplas (x, y)
        con los centroides de cada cluster.
    """
    ID_Clusters = FaseBanco['cluster'].unique()
    Centers = {}
    for cluster_id in sorted(ID_Clusters):
        Cluster = FaseBanco.loc[FaseBanco['cluster']==cluster_id]
        P_center = (Cluster['x'].mean(), Cluster['y'].mean())
        Centers[id] = P_center
    return Centers


def Precedencias_Clusters_Agend(FaseBanco, vector_mineria, 
                                BlockWidthX=10, BlockWidthY=10, Distance_Option=True):
    """
    Calcula el orden de precedencia entre clusters en una fase-banco, considerando un vector de minería.

    Este procedimiento asigna un orden a los clusters según su cercanía al punto de inicio,
    y construye un grafo de precedencias basado en la vecindad entre clusters.

    Parámetros
    ----------
    FaseBanco : pandas.DataFrame
        DataFrame con al menos las columnas 'cluster', 'x', 'y'.
    vector_mineria : list or tuple
        Par de puntos [(x0, y0), (x1, y1)] que define el vector de dirección de minería.
    BlockWidthX : float, opcional
        Ancho de bloque en X, usado para calcular adyacencias. Por defecto 10.
    BlockWidthY : float, opcional
        Alto de bloque en Y, usado para calcular adyacencias. Por defecto 10.
    Distance_Option : bool, opcional
        Si es True, se proyectan los centros de clusters sobre el vector de minería. Si es False, se usa distancia euclidiana.

    Retorna
    -------
    Precedencias_Clusters : dict
        Diccionario donde las llaves son los nuevos IDs de clusters y los valores son listas de clusters predecesores.
    fase_banco : pandas.DataFrame
        Copia modificada del DataFrame original con clusters reindexados según el nuevo orden.
    New_centers : dict
        Diccionario con los centros (x, y) de cada cluster ordenado por nuevo índice.
    execution_time : float
        Tiempo de ejecución del algoritmo en segundos.
    """
    fase_banco = FaseBanco.copy()
    ID_Clusters = fase_banco['cluster'].unique()
    Num_Clusters = len(ID_Clusters)
    
    t1 = time.time()
    Centers = Centros_Clusters(FaseBanco)
    distancias_al_inicio = {}
    Dic_Precedencias = {}
    adjency_matrix = Calculate_Adjency_Matrix(fase_banco, BlockWidthX, BlockWidthY)

    if len(vector_mineria) == 0:
        return dict(), fase_banco, Centers, 0
    else:
        P_inicio, P_final = vector_mineria

    # Calculo de las distancias de los clusters al punto de inicio
    # Distance_Option = True, calcula la distancia proyectando a la recta de dirección de minería
    # Distance_Option = False, calcula la distancia euclideana al punto de inicio.
    for id in Centers.keys():
        if Distance_Option:
            di = np.sqrt( (P_inicio[0] - Centers[id][0])**2 + (P_inicio[1] - Centers[id][1])**2 )
            df = np.sqrt( (P_final[0] - Centers[id][0])**2 + (P_final[1] - Centers[id][1])**2 )
            L = np.sqrt( (P_inicio[0] - P_final[0])**2 + (P_inicio[1] - P_final[1])**2 )
            d = (di**2 + L**2 - df**2)/(2*L)
            distancias_al_inicio[id] = d

        else:
            d = np.sqrt( (Centers[id][0] - P_inicio[0])**2 + (Centers[id][1] - P_inicio[1])**2 )
            distancias_al_inicio[id] = d
        
    distancias_al_inicio = sorted(distancias_al_inicio.items(), key=lambda item: item[1])
    distancias_al_inicio = dict(distancias_al_inicio)
    
    # El primer cluster escogido es el de menor distancia
    Dic_Precedencias[list(distancias_al_inicio.keys())[0]] = []
    Clusters_Candidatos = set()

    # En cada iteración, calcula Clusters_Candidatos, que son los clusters vecinos a los clusters ya "minados" (ordenados).
    # Luego, de entre los candidatos, escoge el que tiene menor distancia y lo añade a los clusters ordenados.
    # Después calcula las precedencias del nuevo cluster, los cuales son los clusters vecinos que ya fuerons "minados".
    for iterations in range(Num_Clusters-1):
        prev_cluster = list(Dic_Precedencias.keys())[iterations]
        Clusters_Vecinos_prev_cluster = Clusters_Vecinos(fase_banco, prev_cluster, adjency_matrix)
        Clusters_Candidatos.update(Clusters_Vecinos_prev_cluster)

        for c in Dic_Precedencias.keys():
            Clusters_Candidatos.discard(c)

        sub_distancias = {k: distancias_al_inicio[k] for k in Clusters_Candidatos}

        if not sub_distancias:
            resto = set(ID_Clusters).difference(set(Dic_Precedencias.keys()))
            sub_distancias = {k: distancias_al_inicio[k] for k in resto}
        
        next_cluster = min(sub_distancias, key=sub_distancias.get)
        Precedencias = []
        Clusters_Vecinos_next_cluster = Clusters_Vecinos(fase_banco, next_cluster, adjency_matrix)

        for cluster in Dic_Precedencias.keys():
            if (cluster in list(Dic_Precedencias.keys())) and (cluster in Clusters_Vecinos_next_cluster):
                Precedencias.append(cluster)

        Dic_Precedencias[next_cluster] = Precedencias
    
    # Finalmente, luego de ordenar y calcular las precedencias, reasigna la etiqueta de cada cluster de acuerdo al orden obtenido,
    # tanto en el dataframe de la fase banco como en el diccionario de precedencias.
    ord_cluster = np.array(list(Dic_Precedencias.keys()))
    Precedencias_Clusters = {}
    New_centers = {}
    contador = 1

    for key in Dic_Precedencias.keys():
        new_value = []

        for value in Dic_Precedencias[key]:
            new_value.append(np.where(ord_cluster==value)[0][0]+1)

        Precedencias_Clusters[contador] = new_value
        New_centers[contador] = Centers[key]
        contador += 1
    
    fase_banco['cluster'] = fase_banco['cluster'].apply(lambda x: np.where(ord_cluster==x)[0][0]+1)
    t2 = time.time()
    execution_time = t2-t1

    return Precedencias_Clusters, fase_banco, New_centers, execution_time


def Precedencias_Clusters_Angle(FaseBanco, vector_mineria, 
                                BlockWidthX=10, BlockWidthY=10, Distance_Option=True, Angle=0):
    """
    Calcula las precedencias entre clusters considerando la dirección de minería y un ángulo de tolerancia.

    A diferencia del enfoque clásico, esta versión incorpora la dirección relativa entre clusters y el vector de minería,
    filtrando los predecesores según un ángulo mínimo respecto a la dirección opuesta al vector de minería.

    Parámetros
    ----------
    FaseBanco : pandas.DataFrame
        DataFrame que contiene información de los bloques, incluyendo columnas 'cluster', 'x' e 'y'.
    vector_mineria : list or tuple
        Par de puntos [(x0, y0), (x1, y1)] que definen la dirección de minería.
    BlockWidthX : float, opcional
        Ancho de los bloques en dirección X. Default = 10.
    BlockWidthY : float, opcional
        Ancho de los bloques en dirección Y. Default = 10.
    Distance_Option : bool, opcional
        Si es True, calcula distancia proyectada sobre el vector de minería. Si es False, usa distancia euclidiana.
    Angle : float, opcional
        Ángulo mínimo (en radianes) respecto a la dirección opuesta al vector de minería para aceptar predecesores.

    Retorna
    -------
    Precedencias_Clusters : dict
        Diccionario donde cada clave es un ID de cluster (reindexado) y su valor es una lista de clusters predecesores.
    fase_banco : pandas.DataFrame
        Copia del DataFrame original con los nuevos índices de clusters.
    New_centers : dict
        Diccionario con los centros de cada cluster, usando los nuevos IDs.
    execution_time : float
        Tiempo de ejecución del procedimiento (en segundos).
    """
    fase_banco = FaseBanco.copy()
    ID_Clusters = fase_banco['cluster'].unique()
    Num_Clusters = len(ID_Clusters)

    t1 = time.time()
    Centers = Centros_Clusters(FaseBanco)
    distancias_al_inicio = {}
    Dic_Precedencias = {}
    adjency_matrix = Calculate_Adjency_Matrix(fase_banco, BlockWidthX, BlockWidthY)

    if len(vector_mineria) == 0:
        return dict(), fase_banco, Centers, 0
    else:
        P_inicio, P_final = vector_mineria

    # Calculo de las distancias de los clusters al punto de inicio
    # Distance_Option = True, calcula la distancia proyectando a la recta de dirección de minería
    # Distance_Option = False, calcula la distancia euclideana al punto de inicio.
    for id in Centers.keys():
        if Distance_Option:
            di = np.sqrt( (P_inicio[0] - Centers[id][0])**2 + (P_inicio[1] - Centers[id][1])**2 )
            df = np.sqrt( (P_final[0] - Centers[id][0])**2 + (P_final[1] - Centers[id][1])**2 )
            L = np.sqrt( (P_inicio[0] - P_final[0])**2 + (P_inicio[1] - P_final[1])**2 )
            d = (di**2 + L**2 - df**2)/(2*L)
            distancias_al_inicio[id] = d

        else:
            d = np.sqrt( (Centers[id][0] - P_inicio[0])**2 + (Centers[id][1] - P_inicio[1])**2 )
            distancias_al_inicio[id] = d
        
    distancias_al_inicio = sorted(distancias_al_inicio.items(), key=lambda item: item[1])
    distancias_al_inicio = dict(distancias_al_inicio)

    # El primer cluster escogido es el de menor distancia
    Dic_Precedencias[list(distancias_al_inicio.keys())[0]] = []
    Clusters_Candidatos = set()

    # Angle = Angle*np.pi/180
    vector_dir_min = np.array(( P_final[0]-P_inicio[0], P_final[1]-P_inicio[1] ))
    vector_dir_min = vector_dir_min/np.linalg.norm(vector_dir_min, 2)

    # En cada iteración, calcula Clusters_Candidatos, que son los clusters vecinos a los clusters ya "minados" (ordenados).
    # Luego, de entre los candidatos, escoge el que tiene menor distancia y lo añade a los clusters ordenados.
    # Después calcula las precedencias del nuevo cluster, los cuales son los clusters vecinos que ya fuerons "minados", y que tienen dirección contraria a la dirección de minería, con tolerancia Angle.
    for iterations in range(Num_Clusters-1):
        prev_cluster = list(Dic_Precedencias.keys())[iterations]
        Clusters_Vecinos_prev_cluster = Clusters_Vecinos(fase_banco, prev_cluster, adjency_matrix)

        Clusters_Candidatos.update(Clusters_Vecinos_prev_cluster)

        for c in Dic_Precedencias.keys():
            Clusters_Candidatos.discard(c)

        sub_distancias = {k: distancias_al_inicio[k] for k in Clusters_Candidatos}

        if not sub_distancias:
            resto = set(ID_Clusters).difference(set(Dic_Precedencias.keys()))
            sub_distancias = {k: distancias_al_inicio[k] for k in resto}
        next_cluster = min(sub_distancias, key=sub_distancias.get)
        Precedencias = []
        Clusters_Vecinos_next_cluster = Clusters_Vecinos(fase_banco, next_cluster, adjency_matrix)

        for cluster in Dic_Precedencias.keys():
            v1 = np.array(( Centers[cluster][0]-Centers[next_cluster][0], Centers[cluster][1]-Centers[next_cluster][1] ))
            v1 = v1/np.linalg.norm(v1,2)
            product = vector_dir_min[0]*v1[0] + vector_dir_min[1]*v1[1]
            # if (np.round(np.arccos(product),6) >= (np.pi/2+Angle) and (cluster in Clusters_Vecinos_next_cluster)) or (cluster in Clusters_Vecinos_next_cluster and cluster==list(Dic_Precedencias.keys())[0]):
            #     Precedencias.append(cluster)
    
            if ( np.round(np.arccos(product),6) >= (np.pi/2+Angle) or cluster==list(Dic_Precedencias.keys())[0] ) and (cluster in Clusters_Vecinos_next_cluster):
                Precedencias.append(cluster)

        Dic_Precedencias[next_cluster] = np.array(Precedencias, dtype=int)

    # Finalmente, luego de ordenar y calcular las precedencias, reasigna la etiqueta de cada cluster de acuerdo al orden obtenido,
    # tanto en el dataframe de la fase banco como en el diccionario de precedencias.
    ord_cluster = np.array(list(Dic_Precedencias.keys()))
    Precedencias_Clusters = {}
    New_centers = {}
    contador = 1

    for key in Dic_Precedencias.keys():
        new_value = []

        for value in Dic_Precedencias[key]:
            new_value.append(np.where(ord_cluster==value)[0][0]+1)

        Precedencias_Clusters[contador] = new_value
        New_centers[contador] = Centers[key]

        contador += 1
    
    fase_banco['cluster'] = fase_banco['cluster'].apply(lambda x: np.where(ord_cluster==x)[0][0]+1)
    t2 = time.time()
    execution_time = t2-t1

    return Precedencias_Clusters, fase_banco, New_centers, execution_time


def Precedencia_Fase_Banco(df):
    """
    Calcula relaciones de precedencia vertical entre pares de (fase, banco) en un DataFrame.

    La precedencia se determina cuando un par (f1, b1) está justo sobre otro (f2, b2) en nivel Z y sus proyecciones en X e Y se intersectan.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame con columnas 'fase', 'banco', 'z', 'x' y 'y'.

    Retorna
    -------
    resultados : list of tuple
        Lista de tuplas ((fase_superior, banco_superior), (fase_inferior, banco_inferior)) que cumplen la relación de precedencia vertical.
    """
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


def Puntos_Iniciales(mina, rampa, 
                     z_min=None, debug=False):
    X_curve, Y_curve, Z_curve = rampa

    if not z_min:
        z_min = mina['z'].min()

    alturas = mina['z'].unique()
    alturas = sorted(alturas[alturas>=z_min])
    
    positions = []

    for h in alturas:
        counter = 0
        for z in Z_curve:
            if h > z:
                if counter == 0:
                    positions.append(counter)
                else:
                    positions.append(counter-1)
                break
            counter+=1

    if debug:
        print(alturas)
        print(positions)

    puntos_iniciales = []
    counter = 0

    for c in positions:
        puntos_iniciales.append((float(X_curve[c]), float(Y_curve[c]), float(alturas[counter])))
        counter+=1
    
    return puntos_iniciales


###############################################################
####################### Extra Functions #######################
###############################################################

def Hadamard_Product_Sparse(A, B):
    """
    Calcula el producto de Hadamard (componente a componente) entre una matriz dispersa y una densa.

    La operación se realiza únicamente sobre las posiciones no nulas de la matriz dispersa `A`,
    lo que permite una implementación eficiente tanto en memoria como en tiempo. El resultado se
    entrega como una matriz dispersa en formato CSR (Compressed Sparse Row).

    Parámetros
    ----------
    A : scipy.sparse.spmatrix
        Matriz dispersa (sparse) de SciPy. Debe tener el mismo tamaño que `B`.
        Se recomienda que esté en formato CSR o CSC para mayor eficiencia.
    B : numpy.ndarray
        Matriz densa de NumPy, del mismo tamaño que `A`.

    Retorna
    -------
    resultado_sparse : scipy.sparse.csr_matrix
        Matriz dispersa en formato CSR que representa el producto de Hadamard `A ∘ B`.
        Sólo contiene valores en las posiciones originalmente no nulas de `A`.

    Errores
    -------
    ValueError
        Si las dimensiones de `A` y `B` no coinciden.

    Notas
    -----
    - Esta función no modifica las matrices originales.
    - Las posiciones donde `A` tiene ceros son ignoradas, incluso si `B` tiene un valor distinto de cero.
    - Útil para aplicar máscaras o escalamiento por bloques en estructuras dispersas.

    Ejemplo
    -------
    >>> from scipy.sparse import csr_matrix
    >>> import numpy as np
    >>> A = csr_matrix([[1, 0], [0, 3]])
    >>> B = np.array([[10, 20], [30, 40]])
    >>> Hadamard_Product_Sparse(A, B).toarray()
    array([[10,  0],
           [ 0, 120]])
    """

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
    resultado_sparse.eliminate_zeros()
    return resultado_sparse


def Z_min(mina, cono, min_area=1e6, debug=False):
    """
    Calcula el nivel mínimo `z` a partir del cual la proyección del cono supera un área mínima.

    Se evalúan secciones horizontales del cono desde la base (`z_c`) hacia abajo. La primera sección con área ≥ `min_area`
    define el valor `z_min`.

    Parámetros
    ----------
    mina : pandas.DataFrame
        DataFrame que contiene una columna 'z' con los niveles verticales disponibles.

    cono : tuple
        Tupla con los parámetros del cono: (a, b, h, alpha, x_c, y_c, z_c), donde:
            - a, b : semi-ejes en la base
            - h    : altura del cono
            - alpha: orientación angular (no se usa en esta función)
            - x_c, y_c : centro en la base (no se usa)
            - z_c  : nivel z superior del cono

    min_area : float, opcional
        Área mínima deseada para la sección horizontal (por defecto 1e6).

    debug : bool, opcional
        Si es True, imprime las áreas correspondientes a cada nivel `z`.

    Retorna
    -------
    z_min : float
        El valor mínimo de `z` donde la sección horizontal del cono alcanza al menos `min_area`.
    """
    a, b, h, alpha, x_c, y_c, z_c = cono
    Z_values = sorted(mina['z'].unique())

    z_min = z_c
    for z in Z_values:
        z_rel = (h-z_c+z)/h
        A = a*z_rel
        B = b*z_rel
        if debug:
            print((z, np.pi*A*B))
        if np.pi*A*B >= min_area:
            z_min = z
            break

    return z_min


