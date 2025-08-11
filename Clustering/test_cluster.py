import cluster_tools as ct
import numpy as np
import pandas as pd
import time


from multiprocessing import Pool, cpu_count
from pathlib import Path

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
    return ct.Clustering_mina(minimina, cm=cm, cr=cr, cp=cp, P=P, R=R, alpha_ley_corte=alpha_ley_corte, options=options)


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


if __name__ == "__main__":
    R = 0.85
    P = 4
    cm = 2
    cp = 10
    cr = 0.25
    FTL = 2204.62
    alpha_ley_corte = 0 # 0 -> Ley Corte Marginal ... 1 -> Ley Corte Critica

    BlockWidthX=10
    BlockWidthY=10
    BlockHeightZ=16

    area_minima_operativa = np.pi*80*80
    ancho_rampa = 30

    ley_marginal = cp/((P-cr)*FTL*R)
    ley_critica = (cp+cm)/((P-cr)*FTL*R)

    ley_corte = ((1-alpha_ley_corte)*ley_marginal + alpha_ley_corte*(ley_critica))*100

    mina_cp = pd.read_csv(Path('Rampas_Final/Params/CP_fases.txt'), sep=r"\s+")
    mina_cp.rename(columns={'0': 'id'}, inplace=True)

    num_fases = mina_cp['fase'].unique()
    fases_new = []
    for f in num_fases:
        fase = mina_cp[mina_cp['fase']==f].copy()
        z_sorted = np.sort(fase['z'].unique())[::-1]
        bancos = np.array(range(1,len(z_sorted)+1))
        z_to_banco = dict(zip(z_sorted, bancos))
        fase['banco'] = fase['z'].map(z_to_banco)
        fases_new.append(fase)

    mina = pd.concat(fases_new)
    mina = mina.sort_index()

    mina['destino'] = [1 if mina.iloc[i]['cut']>= ley_corte else 0 for i in range(len(mina))]

    # path_save = 

    options = {
    'BlockWidthX': 10,
    'BlockWidthY': 10,
    'BlockHeightZ': 16,
    'area_minima_operativa': np.pi*80*80,

    'peso_distancia': 2,
    'peso_ley': 0,
    'tolerancia_ley': 0.001,
    'peso_direccion_mineria': 0.25,
    'tolerancia_direccion_mineria': 0.001,
    'penalizacion_destino': 0.9,
    'penalizacion_roca': 0,
    'penalizacion_c': 0,

    'tamaño_maximo_cluster': 60,
    'tamaño_promedio_cluster': 50,
    'tamaño_minimo_cluster': 5,
    'tolerancia_tamaño_maximo_cluster': 10,
    'tolerancia_tamaño_minimo_cluster': 5,

    'Shape_Refinement': 'Modificado',  # None, 'Tabesh' o 'Modificado'
    'Iteraciones_Shape_Refinement': 5,

    'save': True,
    'save_images': True,
    'path_save': 'test_clustering/Resultados/',
    'path_save_images': 'test_clustering/Imagenes/',
    'path_params': 'test_clustering/Params/',
    'cmap': 'jet', 
    'show_block_label': False
    }

    t1 = time.time()
    mina_clusterizada = Clustering_parallel(mina, workers=4, cm=cm, cr=cr, cp=cp, P=P, R=R, alpha_ley_corte=alpha_ley_corte, options=options)
    t2 = time.time()

    print(f'\nTiempo total de clusterizacion: {t2-t1}')

