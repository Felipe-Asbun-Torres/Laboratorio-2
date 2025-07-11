import numpy as np
import numpy.matlib as matlib
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors 
import time



import plotly.graph_objects as go

from skopt.utils import use_named_args
from skopt.space import Real
from itertools import product


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%}




def plot_fase_3D(fase, width=900, height=800, column_hue='cluster', elipses=[], cone=(), curve=[], opacity_blocks=1, z_ratio=1, points=[]):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=fase['x'],
        y=fase['y'],
        z=fase['z'],
        mode='markers',
        marker=dict(
            size=5,  # tamaño del punto
            color=fase[column_hue],  # color según columna 'cut'
            colorscale='rainbow',
            colorbar=dict(title=column_hue),
            opacity=opacity_blocks
        ),
        hovertemplate=(
            "ID: %{customdata[0]}<br>" +
            f"{column_hue}:"+"%{marker.color:.3f}<br>"
        ),
        customdata=fase[['id', column_hue]],
        name='Bloques'
    ))

    Theta = np.linspace(0, 2*np.pi, 100)
    for elipse in elipses:
        if len(elipse) == 5:
            a, b, x_centro, y_centro, z_centro = elipse
            X_elipse = a*np.cos(Theta) + x_centro
            Y_elipse = b*np.sin(Theta) + y_centro
            Z_elipse = np.full_like(Theta, z_centro)

            fig.add_trace(go.Scatter3d(
                x = X_elipse, y = Y_elipse, z = Z_elipse,
                mode='lines',
                line=dict(color='black', width=2)
            ))
        if len(elipse) == 3:
            X_elipse, Y_elipse, Z_elipse = elipse
            fig.add_trace(go.Scatter3d(
                x=X_elipse, y=Y_elipse, z=Z_elipse,
                mode='lines',
                line=dict(color='blue', width=2)
            ))

    if cone:
        a, b, h, x_centro, y_centro, alpha, z_centro = cone
        Z = np.linspace(z_centro - h, z_centro, 100)
        Theta, Z = np.meshgrid(Theta, Z)
        X_cono = ((a/h)*np.cos(Theta)*np.cos(alpha) - (b/h)*np.sin(Theta)*np.sin(alpha))*(h-z_centro+Z) + x_centro
        Y_cono = ((a/h)*np.cos(Theta)*np.sin(alpha) + (b/h)*np.sin(Theta)*np.cos(alpha))*(h-z_centro+Z) + y_centro
        
        fig.add_trace(go.Surface(
            x=X_cono,
            y=Y_cono,
            z=Z,
            colorscale='viridis',
            opacity=0.8,
            showscale=False
        ))

    if curve:
        X_curve, Y_curve, Z_curve = curve
        fig.add_trace(go.Scatter3d(
            x=X_curve, y=Y_curve, z=Z_curve,
            mode='lines',
            line=dict(color='black', width=5),
            name='Rampa'
        ))
    
    if len(points)>0:
        X = [p[0] for p in points]
        Y = [p[1] for p in points]
        Z = [p[2] for p in points]
        fig.add_trace(go.Scatter3d(
            x=X, y=Y, z=Z,
            mode='markers',
            marker=dict(color='red', size=10),
            name='Puntos Iniciales'
        ))

    x_all = fase['x']
    y_all = fase['y']
    z_all = fase['z']

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

    fig.show()


def plot_2_fase_banco_3D(fases_bancos, column_hue='cluster', precedence_option=True, arcs={}, width=1000, height=800, z_ratio=1):
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


# Función para graficar fases-bancos en 2D.
def plot_fase_banco(
        FaseBanco, 
        block_width=10, 
        block_height=10, 
        column_hue='cut', 
        text_hue=None, 
        cmap='plasma', 
        show_block_label=True, 
        show_grid=True, 
        dpi = 100,
        xsize = 10, 
        ysize = 10, 
        highlight_blocks=[], 
        points=[], 
        arrows=[], 
        sectors=[],
        precedences={},
        centers={},
        elipse=[],
        save_as_image=False,
        path_to_save=''
        ):
    if FaseBanco.empty:
        print("El DataFrame 'FaseBanco' está vacío. No se puede graficar.")
        return
    if not text_hue:
        text_hue = column_hue

    fig, ax = plt.subplots(figsize=(xsize, ysize), dpi=dpi)
    norm = None
    colormap = None
    color_map_discrete = {}
    variables_continuas = FaseBanco.select_dtypes(include='float64').columns.tolist()
    
    fase = FaseBanco['fase'].values[0]
    z = FaseBanco['z'].values[0]

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
        block_width = block_width
        block_height = block_height

        x_corner = x_center - block_width / 2
        y_corner = y_center - block_height / 2

        if is_continuous:
            color = colormap(norm(block_value))
        else:
            color = color_map_discrete.get(block_value, 'gray')
        # print(color)
        rect = patches.Rectangle((x_corner, y_corner), block_width, block_height, linewidth=0.5, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        if show_block_label:
            block_text = row[text_hue]
            if text_hue in variables_continuas:
                block_text = np.trunc(block_text*10)/10
            else:
                block_text = int(block_text)
            ax.text(x_center, y_center, str(block_text), ha='center', va='center', fontsize=8, color='black')
    
    if is_continuous:
        x_min = FaseBanco['x'].min() - 5*block_width
        x_max = FaseBanco['x'].max() + 5*block_width
        # ax.set_xlim(x_min, x_max)
    else:
        x_min = FaseBanco['x'].min() - 5*block_width
        x_max = FaseBanco['x'].max() + 5*block_width
        # ax.set_xlim(x_min, x_max)
    y_min = FaseBanco['y'].min() - 5*block_height
    y_max = FaseBanco['y'].max() + 5*block_height
    # ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Fase {fase} - Banco (Z={z}) - Hue {column_hue}')
    ax.grid(show_grid, color='gray', linestyle='--', linewidth=0.5)


    for block in highlight_blocks:
        ax.plot(FaseBanco['x'][block], FaseBanco['y'][block], 'ro', markersize=10)

    for p in points:
        ax.plot(p[0], p[1], 'bo', markersize=5)

    for a in arrows:
        P1, P2 = a
        ax.annotate('', xy=P2, xytext=P1, arrowprops=dict(arrowstyle='->', color='black', lw=2, mutation_scale=15))
    
    n_clusters = len(precedences.keys())
    for cluster in precedences.keys():
        list_precedences = precedences[cluster]
        for cluster_precedente in list_precedences:
            P_prec = centers[cluster_precedente]
            P = centers[cluster]
            # ax.annotate('', xy=P, xytext=P_prec, arrowprops=dict(arrowstyle='->', color=color_map_discrete[n_clusters - cluster+1], lw=1.5, mutation_scale=15))
            ax.annotate('', xy=P, xytext=P_prec, arrowprops=dict(arrowstyle='->', color='azure', lw=1.5, mutation_scale=15))

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
    
    for params in elipse:
        a, b, x_barra, y_barra = params
        Theta = np.linspace(0, 2*np.pi, 100)
        X_elipse = a*np.cos(Theta) + x_barra
        Y_elipse = b*np.sin(Theta) + y_barra

        ax.plot(X_elipse, Y_elipse, color='black')

    if is_continuous:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label(column_hue, rotation=270, labelpad=15)
    else:
        legend_patches = [patches.Patch(color=color, label=str(value)) for value, color in color_map_discrete.items()]
        ax.legend(handles=legend_patches, title=column_hue, loc='upper right', fontsize=8, title_fontsize=10)
    plt.tight_layout()

    if save_as_image:
        if path_to_save=='':
            path_to_save = f'fase_{fase}_banco_{z}.png'
        else:
            path_to_save = path_to_save + f'/fase_{fase}_banco_{z}.png'
        fig.savefig(path_to_save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Clustering Implementation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Calculate_Adjency_Matrix(FaseBanco, BlockWidth, BlockHeight, Sectores=[]):
    '''
    Crea la matriz de adyacencia de los bloques de la fase-banco respecto a sus coordenadas x e y.
    Considera las dimensiones de los bloques con BlockWidth ('x') y BlockHeight ('y').
    Permite definir sectores, que son listas de coordenadas (P1,P2,P3,P4) que definen los límites de cada sector para
    clusterizar con fronteras (clustering within boundaries).
    Devuelve una matriz sparse (CSR).
    En caso de definir sectores, depende de Hadamard_Product_Sparse.
    '''
    x = FaseBanco['x'].values
    y = FaseBanco['y'].values

    X1 = matlib.repmat(x.reshape(len(x),1), 1, len(x))
    X2 = X1.T
    Y1 = matlib.repmat(y.reshape(len(y),1), 1, len(y))
    Y2 = Y1.T

    D = np.sqrt((1/BlockWidth**2)*(X1 - X2)**2 + (1/BlockHeight**2)*(Y1 - Y2)**2)

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
        return adjency_matrix, fase_banco, sector_matrix
    else:
        return adjency_matrix

def Calculate_Similarity_Matrix(
        FaseBanco, 
        peso_distancia = 2, 
        peso_ley = 0, 
        penalizacion_destino = 0.9, 
        penalizacion_roca = 0.9,
        peso_directional_mining = 0.25,
        tol_ley = 0.01,
        tol_directional_mining = 0.001,
        P_inicio = (-1,-1),
        P_final = (1,1)
        ):
    '''
    Calcula la similaridad entre los bloques de la fase-banco, de acuerdo a distancia, ley, destino, tipo de roca 
    y/o dirección de minería.
    Para no usar un criterio, basta con asignar el valor 0 en el peso o penalización correspondiente.
    Devuelve una matriz densa.
    '''

    n = len(FaseBanco)

    similarity_matrix = np.zeros((n, n), dtype=float)

    x = FaseBanco['x'].values
    y = FaseBanco['y'].values

    # Distancia
    X1 = matlib.repmat(x.reshape(len(x),1), 1, len(x))
    X2 = X1.T
    Y1 = matlib.repmat(y.reshape(len(y),1), 1, len(y))
    Y2 = Y1.T

    D = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)

    ND = D / np.max(D)

    # Ley
    g = FaseBanco['cut'].values
    G1 = matlib.repmat(g.reshape(len(g),1), 1, len(g))
    G2 = G1.T

    G = np.maximum(np.abs(G1 - G2), tol_ley)

    NG = G / np.max(G)

    # Destino
    t = FaseBanco['destino'].values
    T1 = matlib.repmat(t.reshape(len(t),1), 1, len(t))
    T2 = T1.T
    T = T1 != T2

    T = np.ones((n,n)) - (1-penalizacion_destino)*T*np.ones((n,n))

    # Tipo Roca/Material
    if 'tipomineral' in FaseBanco.columns:
        rock = FaseBanco['tipomineral'].values
        R1 = matlib.repmat(rock.reshape(len(rock),1), 1, len(rock))
        R2 = R1.T
        R = R1 != R2

        R = np.ones((n,n)) - (1-penalizacion_roca)*R*np.ones((n,n))
    else:
        R = 1

    # Dirección de minería
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
    resultado_sparse.eliminate_zeros()
    return resultado_sparse


def Find_Most_Similar_Adjacent_Clusters(AdjencyMatrix, SimilarityMatrix):
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


def Find_Corner_Blocks_Tabesh(FaseBanco, AdjencyMatrix):
    '''
    Encuentra los bloques esquinas de la fase-banco. Utiliza el criterio de Tabesh (2013) para definir los bloques esquinas.
    Devuelve un diccionario cuyas llaves son los bloques esquina y cuyos valores son los clusters con los que es vecino.
    Sólo se utiliza en Shape_Refinement_Tabesh.
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
    

def Clustering_Tabesh(
        FaseBanco,
        AdjencyMatrix,
        SimilarityMatrix,
        Average_Desired_Length_Cluster = 30, 
        Max_Cluster_Length = 35, 
        Reset_Clusters_Index = True,
        Debug = False
        ):
    '''
    Realiza un clustering jerárquico y agregativo de la fase-banco.
    Average_Desired_Length_Cluster y Max_Cluster_Length son restricciones del tamaño de los clusters.
    Depende de la funcion Find_Most_Similar_Adjacent_Clusters.
    Devuelve un DataFrame de la fase-banco incluyendo una nueva columna llamada 'cluster' que indica el cluster al que pertenece cada bloque.
    '''
    fase_banco = FaseBanco.copy()
    adj_matrix_sparse = AdjencyMatrix
    sim_matrix = SimilarityMatrix
    execution_time = 0
    if Debug:
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
        
        if Debug:
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
    print(f"========PreProcessing Results========")
    print(f"Tamaño Fase-Banco: {n}")
    print(f"Clusters objetivo: {Max_N_Clusters}")
    print(f"Clusters eliminados: {Clusters_Eliminados}")
    print(f"Total de clusters: {N_Clusters}")
    print(f'Tiempo: {execution_time}')

    return fase_banco, execution_time, N_Clusters

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Shape Refinement %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



def Shape_Refinement_Tabesh(FaseBanco, AdjencyMatrix, Min_Cluster_Length = 10, Iterations_PostProcessing=5, Reset_Clusters_Index=False):
    '''
    Refina la forma de los clusters de la fase-banco, utilizando el criterio de Tabesh (2013).
    Depende de la función Find_Corner_Blocks_Tabesh.
    Devuelve el DataFrame de la fase-banco.
    '''
    
    fase_banco = FaseBanco.copy()
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
        Corner_Blocks = Find_Corner_Blocks_Tabesh(fase_banco, AdjencyMatrix)
        if len(Corner_Blocks) == 0:
            break
        for i in Corner_Blocks.keys():
            if len(Corner_Blocks[i]) == 2:
                Cluster_to_insert = Corner_Blocks[i][0]
            else:
                Cluster_to_insert = np.unique_counts(Corner_Blocks[i]).values[np.unique_counts(Corner_Blocks[i]).counts.argmax()]
            fase_banco.loc[i, 'cluster'] = Cluster_to_insert
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
    Modificación del refinamiento de formas. Además de considerar los corners según Tabesh (2013),
    también considera los bloques que tienen 3 vecinos de distintos clusters o sólo 1 vecino de otro cluster.
    A diferencia del refinamiento de Tabesh, el cual trabaja por etapas, este refinamiento se realiza de forma
    secuencial.
    '''

    fase_banco = FaseBanco.copy()
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
        # corner_blocks = []
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
                        Cluster_to_insert = np.unique_counts(Clusters_Vecinos).values[np.unique_counts(Clusters_Vecinos).counts.argmax()]
                        fase_banco.loc[b, 'cluster'] = Cluster_to_insert
                        # corner_blocks.append(b)
                        # print(f'Caso 1: {cluster}, {b}, {Clusters_Vecinos}')

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
                        # corner_blocks.append(b)
                        # print(f'Caso 2: {cluster}, {b}, {Clusters_Vecinos}')

    t2 = time.time()
    execution_time = t2-t1
    N_Clusters = len(fase_banco['cluster'].unique())
    if Reset_Clusters_Index:
        fase_banco['cluster'] = fase_banco['cluster'].map(lambda x: np.array(range(1,N_Clusters+1))[np.where(fase_banco['cluster'].unique() == x)[0][0]] if x in fase_banco['cluster'].unique() else x)
    print(f"========PostProcessing Results========")
    print(f"Total de clusters: {N_Clusters}")
    print(f'Tiempo: {execution_time}')
    
    return fase_banco, execution_time, N_Clusters

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Automatic Creation of Cluster's Precedences %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def Clusters_Vecinos(FaseBanco, Cluster, AdjencyMatrix):
    '''
    Calcula los clusters vecinos al cluster especificado.
    '''
    Blocks = FaseBanco.loc[FaseBanco['cluster'] == Cluster].index
    rows, cols = AdjencyMatrix.nonzero()
    Clusters_Vecinos = []
    for b in Blocks:
            b_is_row = np.where(rows == b)[0]
            for j in b_is_row:
                if FaseBanco.iloc[b]['cluster'] != FaseBanco.iloc[cols[j]]['cluster']:
                    Clusters_Vecinos.append(FaseBanco.iloc[cols[j]]['cluster'].astype(int))
    return Clusters_Vecinos

def Centros_Clusters(FaseBanco):
    ID_Clusters = FaseBanco['cluster'].unique()
    Centers = {}
    for id in ID_Clusters:
        Cluster = FaseBanco.loc[FaseBanco['cluster']==id]
        P_center = (Cluster['x'].mean(), Cluster['y'].mean())
        Centers[id] = P_center
    return Centers

def Precedencias_Clusters_Agend(FaseBanco, P_inicio, P_final, BlockWidth, BlockHeight, Distance_Option=True):
    '''
    Calcula las precedencias de los clusters. Actualiza el dataframe FaseBanco con el nuevo orden entre clusters y también devuelve
    un diccionario con las precedencias junto con los centros de los clusters.
    Depende de Calculate_Adjency_Matrix.
    '''
    fase_banco = FaseBanco.copy()
    ID_Clusters = fase_banco['cluster'].unique()
    Num_Clusters = len(ID_Clusters)
    
    t1 = time.time()
    Centers = Centros_Clusters(FaseBanco)
    # # Calculo de centros de los clusters
    # for id in ID_Clusters:
    #     Cluster = fase_banco.loc[fase_banco['cluster']==id]
    #     P_center = (Cluster['x'].mean(), Cluster['y'].mean())
    #     Centers[id] = P_center
    
    distancias_al_inicio = {}
    Dic_Precedencias = {}
    adjency_matrix = Calculate_Adjency_Matrix(fase_banco, BlockWidth, BlockHeight)

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
        
        # if not Clusters_Candidatos:
        #     list1 = np.array(list(distancias_al_inicio.keys()))
        #     list2 = np.array(list(Dic_Precedencias.keys()))
        #     list3 = []
        #     for element in list2:
        #         if element in list1:
        #             list3.append(element)
        #     list3 = np.array(list3)

        #     next_cluster = np.delete(np.array(list(distancias_al_inicio.keys())), np.array(list(Dic_Precedencias.keys())))[0]
        #     Dic_Precedencias[next_cluster] = [prev_cluster]

        sub_distancias = {k: distancias_al_inicio[k] for k in Clusters_Candidatos}
        if not sub_distancias:
            resto = set(ID_Clusters).difference(set(Dic_Precedencias.keys()))
            sub_distancias = {k: distancias_al_inicio[k] for k in resto}
            next_cluster = min(sub_distancias, key=sub_distancias.get)
        else:
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


def Precedencias_Clusters_Angle(FaseBanco, P_inicio, P_final, BlockWidth, BlockHeight, Distance_Option=True, Angle=0):
    '''
    Calcula las precedencias de los clusters. Actualiza el dataframe FaseBanco con el nuevo orden entre clusters y también devuelve
    un diccionario con las precedencias junto con los centros de los clusters.
    Depende de Calculate_Adjency_Matrix.
    '''
    fase_banco = FaseBanco.copy()
    ID_Clusters = fase_banco['cluster'].unique()
    Num_Clusters = len(ID_Clusters)
    Centers = {}
    t1 = time.time()
    # Cálculo de centros de los clusters
    for id in ID_Clusters:
        Cluster = fase_banco.loc[fase_banco['cluster']==id]
        P_center = (Cluster['x'].mean(), Cluster['y'].mean())
        Centers[id] = P_center
    
    distancias_al_inicio = {}
    Dic_Precedencias = {}
    adjency_matrix = Calculate_Adjency_Matrix(fase_banco, BlockWidth, BlockHeight)

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

    Angle = Angle*np.pi/180
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
        else:
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



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Metrics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def Rock_Unity(FaseBanco):
    '''
    Calcula la homogeneidad del tipo de roca de los clusters de la fase-banco.
    Devuelve el promedio del Rock Unity y la distribución del Rock Unity por cluster.
    '''
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
    '''
    Calcula la homogeneidad del destino de los clusters de la fase-banco, pesado por el tonelaje de cada bloque.
    Devuelve el promedio del DDF y la distribución del DDF por cluster.
    '''
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
    '''
    Calcula la variación porcentual en torno a la media en la ley de los bloques de cada cluster de la fase-banco.
    Devuelve el promedio del CV de los clusters y la distribución del CV por cluster.
    '''
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Best Cone %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    

# # Demasiado caro de evaluar
# def profit(mina, params_cono, 
#            Minimum_base_area, 
#            BlockWidth, BlockHeight,
#            cm, cr, cp, P, FTL, R,
#            default_density,
#            ley_corte
#            ):
#     print('Calculando profit... \n')
#     a, b, h, x_cone, y_cone, alpha = params_cono
#     z_c = mina['z'].max()
    
#     Z = mina['z'].unique()

#     Profit = 0

#     for z in Z:
#         mina_a_altura_z = mina[mina['z']==z]

#         A_z = (a/h)*(h - z_c + z)
#         B_z = (b/h)*(h - z_c + z)
#         if (A_z <= 0):
#             continue
#         if np.pi*A_z*B_z < Minimum_base_area:
#             continue

#         M = np.array([[1/A_z**2, 0],
#                         [0, 1/B_z**2]])
        
#         Rot = np.array([[np.cos(alpha), -np.sin(alpha)],
#                       [np.sin(alpha), np.cos(alpha)]])
#         Rot_inv = np.transpose(Rot)

#         E = Rot_inv @ M @ Rot

#         X_values = []
#         Y_values = []
#         P_0 = []
#         for id in mina_a_altura_z['id'].values:
#             bloque = mina[mina['id']==id]
#             x_bloque = bloque['x'].values[0]
#             y_bloque = bloque['y'].values[0]
#             P_bloque = np.array([x_bloque, y_bloque])
#             if 1 - (P_bloque-np.array((x_cone, y_cone))) @ E @ (P_bloque-np.array((x_cone, y_cone))) > 0:
#                 P_0 = P_bloque
#                 break
        
#         if len(P_0)==0:
#             continue
        
#         X_values.append(P_0[0])
#         Y_values.append(P_0[1])

#         x_new = P_0[0] - BlockWidth
#         while x_new > x_cone - A_z:
#             X_values.append(x_new)
#             x_new -= BlockWidth
#         x_new = P_0[0] + BlockWidth
#         while x_new < x_cone + A_z:
#             X_values.append(x_new)
#             x_new += BlockWidth

#         y_new = P_0[1] - BlockHeight
#         while x_new > y_cone - B_z:
#             Y_values.append(y_new)
#             y_new -= BlockHeight
#         y_new = P_0[1] + BlockHeight
#         while y_new < y_cone + B_z:
#             Y_values.append(y_new)
#             y_new += BlockHeight
        
#         X_values = sorted(X_values)
#         Y_values = sorted(Y_values)

#         N_x = len(X_values)
#         N_y = len(Y_values)
#         X_values, Y_values = np.meshgrid(X_values, Y_values)

#         for i in range(N_x):
#             for j in range(N_y):
#                 P_bloque = (X_values[j][i], Y_values[j][i])

#                 if 1 - (P_bloque-np.array((x_cone, y_cone))) @ E @ (P_bloque-np.array((x_cone, y_cone))) > 0:
#                     Bloque =  mina_a_altura_z[
#                         (mina_a_altura_z['x']==P_bloque[0]) & 
#                         (mina_a_altura_z['y']==P_bloque[1])]

#                     if Bloque.empty: #Bloque no está en DataFrame

#                         Profit = Profit - cm*default_density
#                     else:
#                         ley_bloque = Bloque['cut'].values[0]
#                         densidad_bloque = Bloque['density'].values[0]

#                         if ley_bloque >= ley_corte:
#                             Profit = Profit + (P-cr)*FTL*R*ley_bloque*densidad_bloque - (cm+cp)*densidad_bloque
#                         else:
#                             Profit = Profit - cm*densidad_bloque
#     return Profit

# def continuous_profit(mina, params_cono, 
#            Minimum_base_area, 
#            BlockWidth, BlockHeight, BlockHeightZ,
#            cm, cr, cp, P, FTL, R,
#            default_density,
#            ley_corte,
#            aproximator='uniform'
#            ):
#     a, b, h, x_cone, y_cone, alpha = params_cono
#     z_c = mina['z'].max()

#     Profit = 0

#     mina_df = mina.copy()

#     X_values = sorted(mina['x'].unique())+BlockWidth/2
#     Y_values = sorted(mina['y'].unique())+BlockHeight/2
#     Z_values = sorted(mina['z'].unique())+BlockHeightZ/2

#     def densidad_profit(posicion):
#         x0, y0, z0 = posicion
#         for x in X_values:
#             if x>x0:
#                 X_Bloque = x-BlockWidth/2
#         for y in Y_values:
#             if y>y0:
#                 Y_Bloque = y-BlockHeight/2
#         for z in Z_values:
#             if z>z0:
#                 Z_Bloque = z-BlockHeightZ/2
#         Bloque = mina[
#                     mina['x']==X_Bloque &
#                     mina['y']==Y_Bloque &
#                     mina['z']==Z_Bloque
#         ]


# # Primer try: No sirve pues Profit no es diferenciable
# def Best_Cone_by_Profit_first_try(mina, P=4, cr=0.25, cm=2, cp=10, R=0.85, BlockWidth=10, BlockHeight=10, block_tolerance=4, max_global_angle=45, alpha_ley_corte=0, Minimum_base_area=0, default_density=0.25):
#     FTL = 2204.62
#     ley_marginal = cp/((P-cr)*FTL*R)
#     ley_critica = (cp+cm)/((P-cr)*FTL*R)

#     ley_corte = ((1-alpha_ley_corte)*ley_marginal + alpha_ley_corte*(ley_critica))*100
    
#     x_c = mina['x'].mean()
#     y_c = mina['y'].mean()
#     z_c = mina['z'].max()

#     points = np.array(mina[['x', 'y', 'z']])

#     a_guess = (2/3)*(mina['x'].max() - mina['x'].min())
#     b_guess = (2/3)*(mina['y'].max() - mina['y'].min())
#     h_guess = (4/3)*(mina['z'].max() - mina['z'].min())

#     def objective(params):
#         a, b, h, x_cone, y_cone = params
#         varepsilon = 1e-2
#         if (a <= 0) or (b <= 0) or (h <= 0):
#             return np.inf
#         return -profit(mina, params, Minimum_base_area, BlockWidth, BlockHeight, cm, cr, cp, P, FTL, R, default_density, ley_corte) + varepsilon*abs(x_cone-x_c)**2 + varepsilon*abs(y_cone-y_c)**2
    
#     def constrainst_gen(point):
#         z_c = mina['z'].max()
#         def constraint(params):
#             a, b, h, x_cone, y_cone = params
#             A = (a/h)*(h-z_c+point[2])
#             B = (b/h)*(h-z_c+point[2])
#             if A < 0:
#                 A = 1e-12
#             if B < 0:
#                 B = 1e-12
#             A += block_tolerance*BlockWidth
#             B += block_tolerance*BlockHeight
#             M = np.array([[1/A**2, 0],
#                         [0, 1/B**2]])
#             return 1 - (point[0:2]-np.array((x_cone, y_cone))) @ M @ (point[0:2]-np.array((x_cone, y_cone)))
#         return constraint
    
#     constraints = [{'type': 'ineq', 'fun': constrainst_gen(p)} for p in points]

#     constraints.append({'type': 'ineq',
#                         'fun': lambda params: max_global_angle - np.arctan(params[2]/params[0])*180/np.pi})
    
#     constraints.append({'type': 'ineq',
#                         'fun': lambda params: max_global_angle - np.arctan(params[2]/params[1])*180/np.pi})
    
#     initial_guess = (a_guess, b_guess, h_guess, x_c, y_c)
#     result = sp.optimize.minimize(objective, initial_guess, constraints=constraints, method='SLSQP')

#     print(result)
#     return result

# Second Try: Demasiado tiempo de cómputo



###########################################################################################
########################### Relleno de mina + Cálculo de profit ###########################
###########################################################################################


# def relleno_mina(mina, default_density,
#                  BlockWidth, BlockHeight, BlockHeightZ,
#                  cm, cr, cp, P, FTL, R, ley_corte,
#                  relleno_lateral=30):
#     mina_rellenada = mina.copy()
#     new_blocks = pd.DataFrame({
#         col: pd.Series(dtype=mina[col].dtype) for col in mina.columns
#     })

#     x_min = mina['x'].min()
#     x_max = mina['x'].max()
#     y_min = mina['y'].min()
#     y_max = mina['y'].max()

#     X_values = mina['x'].unique()
#     Y_values = mina['y'].unique()
#     Z_values = mina['z'].unique()

#     for i in range(1, relleno_lateral+1):
#         x_inf_new = x_min - i*BlockWidth
#         x_sup_new = x_max + i*BlockWidth
#         y_inf_new = y_min - i*BlockHeight
#         y_sup_new = y_max + i*BlockHeight

#         X_values = np.append(X_values, [x_inf_new, x_sup_new])
#         Y_values = np.append(Y_values, [y_inf_new, y_sup_new])

#     X_values = sorted(set(X_values))
#     Y_values = sorted(set(Y_values))

#     for x in X_values:
#         for y in Y_values:
#             in_mina = mina[(mina['x']==x) & (mina['y']==y)]
#             if in_mina.empty:
#                 to_add = pd.DataFrame(data=0, columns=new_blocks.columns, index=range(len(Z_values)))
                
#                 to_add['x'] = [x]*len(Z_values)
#                 to_add['y'] = [y]*len(Z_values)
#                 to_add['z'] = Z_values

#                 to_add['density'] = [default_density]*len(Z_values)

#                 new_blocks = pd.concat([new_blocks, to_add], ignore_index=True)
#             else:
#                 z_in_mina = in_mina['z'].unique()

#                 z_to_add = list(set(Z_values) - set(z_in_mina))
#                 z_max_loc = np.array(z_in_mina).max()

#                 to_add = pd.DataFrame(data=0, columns=new_blocks.columns, index=range(len(z_to_add)))
#                 to_add['x'] = [x]*len(z_to_add)
#                 to_add['y'] = [y]*len(z_to_add)
#                 to_add['z'] = z_to_add

#                 to_add['density'] = np.where(to_add['z']<z_max_loc, default_density, 0)

#                 new_blocks = pd.concat([new_blocks, to_add], ignore_index=True)


#     mina_rellenada = pd.concat([mina_rellenada, new_blocks], ignore_index=True)


#     Block_Vol = BlockWidth * BlockHeight * BlockHeightZ

#     mina_rellenada['value'] = np.where(
#         mina_rellenada['cut']<ley_corte, 
#         -cm*mina_rellenada['density']*Block_Vol, 
#         (P-cr)*FTL*R*mina_rellenada['cut']*mina_rellenada['density']*Block_Vol - (cp+cm)*mina_rellenada['density']*Block_Vol)

#     return mina_rellenada

def relleno_mina(mina, default_density,
                 BlockWidth, BlockHeight, BlockHeightZ,
                 cm, cr, cp, P, FTL, R, ley_corte,
                 relleno_lateral=30):
    
    mina_copy = mina.copy()

    mina_copy['value'] = 0 

    x_min, x_max = mina['x'].min(), mina['x'].max()
    y_min, y_max = mina['y'].min(), mina['y'].max()

    X_values = set(mina['x'].unique())
    Y_values = set(mina['y'].unique())
    Z_values = mina['z'].unique()

    # Relleno lateral: agrega extremos
    for i in range(1, relleno_lateral + 1):
        X_values.update([x_min - i * BlockWidth, x_max + i * BlockWidth])
        Y_values.update([y_min - i * BlockHeight, y_max + i * BlockHeight])

    # Convertimos los sets a listas ordenadas
    X_values = sorted(X_values)
    Y_values = sorted(Y_values)

    columns = list(mina_copy.columns)
    Coords = list(product(X_values, Y_values, Z_values))

    new_mina = pd.DataFrame(0, index=range(len(Coords)), columns=columns)
    new_mina[['x','y','z']] = pd.DataFrame(Coords, columns=['x','y','z'])
    new_mina['density'] = [default_density]*len(new_mina)


    min_z_por_xy = mina_copy.groupby(['x', 'y'])['z'].max().to_dict()

    def calcular_density(row):
        key = (row['x'], row['y'])
        if key in min_z_por_xy and row['z'] < min_z_por_xy[key]:
            return 0
        else:
            return default_density
    def calcular_tipomineral(row):
        key = (row['x'], row['y'])
        if key in min_z_por_xy and row['z'] < min_z_por_xy[key]:
            return -1
        else:
            return -2

    new_mina['density'] = new_mina.apply(calcular_density, axis=1)
    new_mina['tipomineral'] = new_mina.apply(calcular_tipomineral, axis=1)

    claves = ['x','y','z']

    mina_rellena = mina_copy.set_index(claves).combine_first(new_mina.set_index(claves)).reset_index()

    claves_B = set(map(tuple, mina[claves].values))
    mascara_solo_new_mina = ~mina_rellena[claves].apply(tuple, axis=1).isin(claves_B)

    inicio_id = len(mina) + 1
    mina_rellena.loc[mascara_solo_new_mina, 'id'] = range(inicio_id, inicio_id + mascara_solo_new_mina.sum())

    mina_rellena['id'] = mina_rellena['id'].astype(int)

    Block_Vol = BlockWidth * BlockHeight * BlockHeightZ

    mina_rellena['value'] = np.where(
        mina_rellena['cut']<ley_corte, 
        -cm*mina_rellena['density']*Block_Vol, 
        (P-cr)*FTL*R*mina_rellena['cut']*mina_rellena['density']*Block_Vol - (cp+cm)*mina_rellena['density']*Block_Vol)

        
    return mina_rellena





def isin_cone(mina_rellena, params_cono, Minimum_base_area=0, horizontal_tolerance=0):
    a, b, h, x_cone, y_cone, alpha = params_cono
    z_c = mina_rellena['z'].max()

    a += horizontal_tolerance
    b += horizontal_tolerance


    M = np.array([[1/a**2, 0],
                        [0, 1/b**2]])
        
    Rot = np.array([[np.cos(alpha), -np.sin(alpha)],
                    [np.sin(alpha), np.cos(alpha)]])
    Rot_inv = np.transpose(Rot)

    E = Rot_inv @ M @ Rot

    x = np.asarray(mina_rellena['x'])
    y = np.asarray(mina_rellena['y'])
    z = np.asarray(mina_rellena['z'])

    z_rel = (h-z_c+z)/h
    norm_rel = E[0,0]*(x-x_cone)**2 + 2*E[0,1]*(x-x_cone)*(y-y_cone) + E[1,1]*(y-y_cone)**2

    points_in = pd.Series(norm_rel<=z_rel**2, name='isin_cone')

    if Minimum_base_area > 0:
        Z = sorted(mina_rellena['z'].unique())
        Z_rel = (h-z_c+Z)/h

        A_z = a*Z_rel
        B_z = b*Z_rel

        Areas_z = np.pi*A_z*B_z
        if Minimum_base_area >= Areas_z.max():
            Warning('Valor de Minimum_base_area demasiado alto.')
            return pd.Series()
        
        index_z_min = np.where(1 - (Areas_z<Minimum_base_area))[0].min()

        z_min = Z[index_z_min]

        points_in = points_in*(mina_rellena['z']>=z_min)

    return points_in

def profit(mina_rellena, params_cono, Minimum_base_area=0, horizontal_tolerance=0):
    points_in_cone = mina_rellena[isin_cone(mina_rellena, params_cono, 
                                            Minimum_base_area=Minimum_base_area, 
                                            horizontal_tolerance=horizontal_tolerance)]

    return points_in_cone['value'].sum()


    




#########################################################################################
############################# Optimización para mejor cono ##############################
#########################################################################################

def Best_Cone_by_Volume(fase, max_global_angle=45):
    x_c = fase['x'].mean()
    y_c = fase['y'].mean()
    z_c = fase['z'].max()

    points = np.array(fase[['x', 'y', 'z']])

    a_guess = (2/3)*(fase['x'].max() - fase['x'].min())
    b_guess = (2/3)*(fase['y'].max() - fase['y'].min())
    h_guess = (4/3)*(fase['z'].max() - fase['z'].min())
    alpha_guess = 0

    def objective(params):
        a, b, h, x_cone, y_cone, alpha = params
        varepsilon = 1e-2
        if (a <= 0) or (b <= 0) or (h <= 0):
            return 1e20
        return (1/3)*np.pi*a*b*h + varepsilon*abs(x_cone-x_c)**2 + varepsilon*abs(y_cone-y_c)**2
    
    def constrainst_gen(point):
        z_c = fase['z'].max()
        def constraint(params):
            a, b, h, x_cone, y_cone, alpha = params
            A = (a/h)*(h-z_c+point[2])
            B = (b/h)*(h-z_c+point[2])
            if A < 0:
                A = 1e-12
            if B < 0:
                B = 1e-12
            M = np.array([[1/A**2, 0],
                        [0, 1/B**2]])
            Rot = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
            Rot_inv = np.transpose(Rot)
            E = Rot_inv @ M @ Rot
            return 1 - (point[0:2]-np.array((x_cone, y_cone))) @ E @ (point[0:2]-np.array((x_cone, y_cone)))
        return constraint
    
    constraints = [{'type': 'ineq', 'fun': constrainst_gen(p)} for p in points]
    
    constraints.append({'type': 'ineq',
                        'fun': lambda params: max_global_angle - np.arctan(params[2]/params[0])*180/np.pi})
    
    constraints.append({'type': 'ineq',
                        'fun': lambda params: max_global_angle - np.arctan(params[2]/params[1])*180/np.pi})

    constraints.append({'type': 'ineq',
                        'fun': lambda params: np.pi/2 - params[5]})
    
    constraints.append({'type': 'ineq',
                        'fun': lambda params: params[5] + np.pi/2})

    initial_guess = (a_guess, b_guess, h_guess, x_c, y_c, alpha_guess)
    result = sp.optimize.minimize(objective, initial_guess, constraints=constraints, method='SLSQP')

    print(result)
    return result

def Best_Cone_by_Profit_sp_minimize(mina, max_global_angle=45, Minimum_base_area=0, method='L-BFGS-B', fd_step=1e-1, ftol=0.01):
    x_c = mina['x'].mean()
    y_c = mina['y'].mean()
    z_c = mina['z'].max()
    
    a_low = (0.01)*(mina['x'].max() - mina['x'].min())
    a_up = (0.75)*(mina['x'].max() - mina['x'].min())

    b_low = (0.01)*(mina['y'].max() - mina['y'].min())
    b_up = (0.75)*(mina['y'].max() - mina['y'].min())

    h_low = (0.5)*(mina['z'].max() - mina['z'].min())
    h_up = (2)*(mina['z'].max() - mina['z'].min())

    lambda_x = 0.45
    x_low = (1-lambda_x)*mina['x'].min() + (lambda_x)*mina['x'].max()
    x_up = (lambda_x)*mina['x'].min() + (1-lambda_x)*mina['x'].max()

    lambda_y = 0.45
    y_low = (1-lambda_y)*mina['y'].min() + (lambda_y)*mina['y'].max()
    y_up = (lambda_y)*mina['y'].min() + (1-lambda_y)*mina['y'].max()


    a_guess = (0.5)*(mina['x'].max() - mina['x'].min())
    b_guess = (0.5)*(mina['y'].max() - mina['y'].min())
    x_guess = mina['x'].median()
    y_guess = mina['y'].median()
    h_guess = (0.75)*(mina['z'].max() - mina['z'].min())
    alpha_guess = 0

    x0 = [a_guess, b_guess, h_guess, x_guess, y_guess, alpha_guess]

    def objective(params):
        a, b, h, x_cone, y_cone, alpha = params
        varepsilon = 1e-2
        if np.arctan(h/a)*180/np.pi > max_global_angle:
            return 1e20
        if np.arctan(h/b)*180/np.pi > max_global_angle:
            return 1e20
        if (a <= 0) or (b <= 0) or (h <= 0):
            return 1e20
        return -profit(mina, params, Minimum_base_area=Minimum_base_area) + varepsilon*abs(x_cone-x_c)**2 + varepsilon*abs(y_cone-y_c)**2

    bounds = [(a_low, a_up),
              (b_low, b_up),
              (h_low, h_up),
              (x_low, x_up),
              (y_low, y_up),
              (-np.pi/2, np.pi/2)]

    result = sp.optimize.minimize(objective, x0,
                                  bounds=bounds,
                                  method=method,
                                  jac='3-point',
                                  options={'disp': True, 'finite_diff_rel_step': fd_step, 'ftol': ftol})
    

    return result


def Best_Cone_by_Profit_diff_evol(mina, max_global_angle=45, Minimum_base_area=0, maxiter=1000, popsize=15, polish=True, init='latinhypercube', strategy='best1bin', mutation=(0.5, 1), recombination=0.7, rng=None):
    
    x_c = mina['x'].mean()
    y_c = mina['y'].mean()
    z_c = mina['z'].max()

    a_low = (1/3)*(mina['x'].max() - mina['x'].min())
    a_up = (2)*(mina['x'].max() - mina['x'].min())

    b_low = (1/3)*(mina['y'].max() - mina['y'].min())
    b_up = (2)*(mina['y'].max() - mina['y'].min())

    h_low = (1/3)*(mina['z'].max() - mina['z'].min())
    h_up = (2)*(mina['z'].max() - mina['z'].min())

    lambda_x = 0
    x_low = (1-lambda_x)*mina['x'].min() + (lambda_x)*mina['x'].max()
    x_up = (lambda_x)*mina['x'].min() + (1-lambda_x)*mina['x'].max()

    lambda_y = 0
    y_low = (1-lambda_y)*mina['y'].min() + (lambda_y)*mina['y'].max()
    y_up = (lambda_y)*mina['y'].min() + (1-lambda_y)*mina['y'].max()


    def objective(params):
        a, b, h, x_cone, y_cone, alpha = params
        varepsilon = 1e-2

        if np.arctan(h/a)*180/np.pi > max_global_angle:
            return 1e20
        if np.arctan(h/b)*180/np.pi > max_global_angle:
            return 1e20
        if (a <= 0) or (b <= 0) or (h <= 0):
            return 1e20
        
        return -profit(mina, params, Minimum_base_area) + varepsilon*abs(x_cone-x_c)**2 + varepsilon*abs(y_cone-y_c)**2
    
    bounds = [(a_low, a_up), (b_low, b_up), (h_low, h_up), (x_low, x_up), (y_low, y_up), (-np.pi/2, np.pi/2)]

    result = sp.optimize.differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, polish=polish, disp=True, workers=1, init=init, strategy=strategy, mutation=mutation, recombination=recombination, rng=rng)

    return result


def Best_Cone_by_Profit_basinhopping(mina, max_global_angle=45, Minimum_base_area=0, niter=1000, T=1, stepsize=0.5):
    
    x_c = mina['x'].mean()
    y_c = mina['y'].mean()
    z_c = mina['z'].max()

    norm_a = mina['x'].max() - mina['x'].min()
    norm_b = mina['y'].max() - mina['y'].min()
    norm_h = mina['z'].max() - mina['z'].min()
    diam = (norm_a+norm_b)/2 # "diametro de la mina"

    a_guess = 4/3
    b_guess = 4/3
    h_guess = 1
    x_guess = 0
    y_guess = 0
    alpha_guess = 0

    x0 = np.array([a_guess, b_guess, h_guess, x_guess, y_guess, alpha_guess])

    def objective(params):
        a, b, h, x_cone, y_cone, alpha = params
        varepsilon = 1e-2

        a = norm_a*a
        b = norm_b*b
        h = norm_h*h
        x_cone = x_c + diam*x_cone
        y_cone = y_c + diam*y_cone
        alpha = (np.pi/2)*alpha

        if np.arctan(h/a)*180/np.pi > max_global_angle:
            return 1e20
        if np.arctan(h/b)*180/np.pi > max_global_angle:
            return 1e20
        if (a <= 0) or (b <= 0) or (h <= 0):
            return 1e20
        
        return -profit(mina, [a,b,h,x_cone,y_cone,alpha], Minimum_base_area) + varepsilon*abs(x_cone-x_c)**2 + varepsilon*abs(y_cone-y_c)**2


    result = sp.optimize.basinhopping(objective, x0, niter=niter, T=T, stepsize=stepsize, disp=True)

    return result


# Enfoque Scikit
def make_search_space(amin, amax, bmin, bmax, hmin, hmax, xmin, xmax, ymin, ymax, alphamin, alphamax):
    return [
        Real(amin, amax, name="a"),
        Real(bmin, bmax, name="b"),
        Real(hmin, hmax, name="h"),
        Real(xmin, xmax, name="x_cone"),
        Real(ymin, ymax, name="y_cone"),
        Real(alphamin, alphamax, name='alpha')
    ]

def make_objective(mina,
                   Minimum_base_area,
                   max_global_angle,
                   horizontal_tolerance,
                   x_c, y_c,
                   varepsilon,
                   search_space):

    @use_named_args(search_space)
    def objective(a, b, h, x_cone, y_cone, alpha):
        try:
            if np.arctan(h/a)*180/np.pi > max_global_angle:
                return 1e20
            if np.arctan(h/b)*180/np.pi > max_global_angle:
                return 1e20
            val = profit(mina, [a, b, h, x_cone, y_cone, alpha], Minimum_base_area=Minimum_base_area, horizontal_tolerance=horizontal_tolerance)
            penalty = varepsilon * ((x_cone - x_c)**2 + (y_cone - y_c)**2)
            return -val + penalty
        except Exception as e:
            print(f"Error: {e}")
            return 1e20  # Penalización si falla

    return objective


#########################################################################################
####################### Rampa y puntos iniciales para clustering ########################
#########################################################################################



# def Rampa_old(fase, Cone=[], Angulo_Descenso=np.arcsin(0.10), theta_0=0, t_final=2*5*np.pi, n=1000, orientation=1):
#     if not Cone:
#         raise Exception('Debe adjuntar cono')
#     if orientation >= 0:
#         orientation = 1
#     else:
#         orientation = -1


#     z_c = fase['z'].max()
#     p = -np.sin(Angulo_Descenso)
#     a_opt, b_opt, h_opt, x_opt, y_opt, alpha_opt = Cone

#     T = np.linspace(0, t_final, n)
#     X_curve = []
#     Y_curve = []
#     Z_curve = []

#     z = z_c
#     theta = theta_0

#     x = a_opt*np.cos(theta_0) + x_opt
#     y = b_opt*np.sin(theta_0) + y_opt

#     X_curve.append(x)
#     Y_curve.append(y)
#     Z_curve.append(z_c)

#     for t in T:
#         if t == 0:
#             t0 = 0
#         else:
#             ra = (a_opt/h_opt)*(h_opt - z_c + z)
#             rb = (b_opt/h_opt)*(h_opt - z_c + z)
#             A = p**2 * (b_opt/h_opt)**2 * np.sin(theta) + p**2 * (b_opt/h_opt)**2 * np.cos(theta) + p**2 - 1
#             B = (2*p**2 * (b_opt/h_opt)*rb*np.sin(theta)*np.cos(theta) - 2*p**2 * (a_opt/h_opt)*ra*np.sin(theta)*np.cos(theta))
#             C = p**2 * rb**2 * np.cos(theta)**2 + p**2 * ra**2 * np.sin(theta)**2

#             dz = ((-B) + np.sqrt( B**2 - 4*A*C ))/(2*A)

#             theta_new = theta_0 + orientation*t
#             z_new = z + (t-t0)*dz

#             x = (a_opt/h_opt)*(h_opt-z_c+z_new)*np.cos(theta_new) + x_opt
#             y = (b_opt/h_opt)*(h_opt-z_c+z_new)*np.sin(theta_new) + y_opt

#             X_curve.append(x)
#             Y_curve.append(y)
#             Z_curve.append(z)

#             t0 = t
#             z = z_new
#             theta = theta_new

#     return X_curve, Y_curve, Z_curve

def Rampa(fase, Cone=None, Angulo_Descenso=np.arcsin(0.10), theta_0=0, n=1000, orientation=1, z_min=None, max_vueltas=5):
    if Cone is None or len(Cone)==0:
        raise Exception('Debe adjuntar cono')
    if orientation >= 0:
        orientation = 1
    else:
        orientation = -1
    
    t_final=2*max_vueltas*np.pi

    z_c = fase['z'].max()
    p = -np.sin(Angulo_Descenso)
    a, b, h, x_cone, y_cone, alpha_cone = Cone

    T = np.linspace(0, t_final, n)
    X_curve = []
    Y_curve = []
    Z_curve = []

    z = z_c
    theta = theta_0

    x = (a*np.cos(theta_0)*np.cos(alpha_cone) - b*np.sin(theta_0)*np.sin(alpha_cone)) + x_cone
    y = (a*np.cos(theta_0)*np.sin(alpha_cone) + b*np.sin(theta_0)*np.cos(alpha_cone)) + y_cone

    X_curve.append(x)
    Y_curve.append(y)
    Z_curve.append(z)
    
    stop = False

    for t in T:
        if t == 0:
            t0 = 0
        else:
            R_1 = a*np.cos(theta)*np.cos(alpha_cone) - b*np.sin(theta)*np.sin(alpha_cone)
            dR_1 = (-a*np.sin(theta)*np.cos(alpha_cone) - b*np.cos(theta)*np.sin(alpha_cone))*orientation
            R_2 = a*np.cos(theta)*np.sin(alpha_cone) + b*np.sin(theta)*np.cos(alpha_cone)
            dR_2 = (-a*np.sin(theta)*np.sin(alpha_cone) + b*np.cos(theta)*np.cos(alpha_cone))*orientation

            A = p**2 - 1 + p**2 * R_1**2/h**2 + p**2 * R_2**2/h**2
            B = (2*p**2/h)*(R_1*dR_1 + R_2*dR_2)*(1-z_c/h + z/h)
            C = p**2*(dR_1**2 + dR_2**2)*(1 - z_c/h + z/h)**2

            dz = ((-B) + np.sqrt( B**2 - 4*A*C ))/(2*A)

            theta_new = theta_0 + orientation*t
            z_new = z + (t-t0)*dz

            x = (a*np.cos(theta_new)*np.cos(alpha_cone) - b*np.sin(theta_new)*np.sin(alpha_cone))*(1-z_c/h+z_new/h) + x_cone
            y = (a*np.cos(theta_new)*np.sin(alpha_cone) + b*np.sin(theta_new)*np.cos(alpha_cone))*(1-z_c/h+z_new/h) + y_cone

            X_curve.append(x)
            Y_curve.append(y)
            Z_curve.append(z)

            t0 = t
            z = z_new
            theta = theta_new

            if stop:
                break

            if z_min:
                if z_new < z_min:
                    stop = True
            

    return X_curve, Y_curve, Z_curve

def Puntos_Iniciales(mina, rampa, z_min=None, debug=False):
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


def Z_min(mina, cono, Minimum_Area=1e6, debug=False):
    a, b, h, x_c, y_c, alpha = cono
    Z_values = sorted(mina['z'].unique())
    z_c = mina['z'].max()

    z_min = z_c
    for z in Z_values:
        z_rel = (h-z_c+z)/h
        A = a*z_rel
        B = b*z_rel
        if debug:
            print((z, np.pi*A*B))
        if np.pi*A*B >= Minimum_Area:
            z_min = z
            break

    return z_min


def isin_rampa(mina_rellena, rampa, ancho_rampa):
    id_bloques_in_rampa = set()
    X_curve, Y_curve, Z_curve = rampa
    alturas = np.array(sorted(mina_rellena['z'].unique()))

    for c in range(len(Z_curve)):
        x, y, z = (X_curve[c], Y_curve[c], Z_curve[c])
        z_in_mina = np.abs((alturas - z)).argmin()
        z_in_mina = alturas[z_in_mina]

        mask = ((mina_rellena['x']-x)**2 + (mina_rellena['y']-y)**2 <= (ancho_rampa/2)**2) & (mina_rellena['z']==z_in_mina)

        id_bloques_in_rampa.update(list(mina_rellena[mask]['id'].unique()))
    
    return list(id_bloques_in_rampa)

def block_precedence(mina_rellena, block_coords,
                     BlockWidth, BlockHeight, BlockHeightZ,
                     config='5-points'):
    if (not config=='5-points') or (not config=='9-points'):
        print('Solo use congif=5-points o config=9-points')
        return None
    x, y, z = block_coords
    precedences = []
    if z == mina_rellena['z'].max():
        return precedences
    
    if config=='5-points':
        offsets = [
            (0, 0),                     # Centro arriba
            (0, BlockHeight),           # Norte
            (0, -BlockHeight),          # Sur
            (BlockWidth, 0),            # Este
            (-BlockWidth, 0),           # Oeste
        ]
    elif config=='9-points':
        offsets = [
            (0, 0),                     # Centro arriba
            (0, BlockHeight),           # Norte
            (0, -BlockHeight),          # Sur
            (BlockWidth, 0),            # Este
            (-BlockWidth, 0),           # Oeste
            (BlockWidth, BlockHeight),  # Noreste
            (-BlockWidth, BlockHeight), # Noroeste
            (-BlockWidth, -BlockHeight),# Suroeste
            (BlockWidth, -BlockHeight), # Sureste
        ]

    z_p = z + BlockHeightZ
    vecinos = pd.DataFrame([
        {'x': x + dx, 'y': y + dy, 'z': z_p}
        for dx, dy in offsets
    ])

    coincidencias = vecinos.merge(mina_rellena[['x', 'y', 'z', 'id']],
                                    on=['x', 'y', 'z'], how='inner')

    precedences = coincidencias['id'].tolist()

    return precedences
        
def expand_all_precedences(mina_rellena, block_coords,
                           BlockWidth, BlockHeight, BlockHeightZ,
                           config='5-points'):
    if (not config=='5-points') or (not config=='9-points'):
        print('Solo use congif=5-points o config=9-points')
        return None

    visited = set()
    queue = [block_coords]
    all_precedences = set()

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        prec_ids = block_precedence(
            mina_rellena, current,
            BlockWidth, BlockHeight, BlockHeightZ,
            config=config
        )

        # Buscar coordenadas correspondientes a estos IDs
        prec_coords_df = mina_rellena[mina_rellena['id'].isin(prec_ids)][['x', 'y', 'z']]
        new_coords = [tuple(row) for row in prec_coords_df.to_numpy()]

        # Agregamos los IDs nuevos
        all_precedences.update(prec_ids)

        # Añadimos los nuevos bloques a la cola si no han sido visitados
        for coord in new_coords:
            if coord not in visited:
                queue.append(coord)

    return list(all_precedences)


# def total_additional_blocks(mina_rellena, rampa, z_min,
#                             ancho_rampa):
#     id_in_rampa = isin_rampa(mina_rellena, rampa, ancho_rampa)

def expand_group_precedences(mina_rellena, block_coords_list,
                              BlockWidth, BlockHeight, BlockHeightZ,
                              config='5-points'):
    if (not config=='5-points') or (not config=='9-points'):
        print('Solo use congif=5-points o config=9-points')
        return None

    visited_coords = set()
    # Convertir todos los elementos de entrada a tuplas
    queue = [tuple(coord) for coord in block_coords_list]
    all_precedences = set()

    while queue:
        current = queue.pop(0)
        if current in visited_coords:
            continue
        visited_coords.add(current)

        prec_ids = block_precedence(
            mina_rellena, current,
            BlockWidth, BlockHeight, BlockHeightZ,
            config=config
        )

        all_precedences.update(prec_ids)

        prec_coords_df = mina_rellena[mina_rellena['id'].isin(prec_ids)][['x', 'y', 'z']]
        new_coords = [tuple(row) for row in prec_coords_df.to_numpy()]

        for coord in new_coords:
            if coord not in visited_coords:
                queue.append(coord)

    return list(all_precedences)

def all_precedences(mina_rellena, block_coords,
                           BlockWidth, BlockHeight, BlockHeightZ,
                           config='5-points', angulo_apertura=45):
    if (not config=='5-points') and (not config=='9-points'):
        print('Solo use config=5-points o config=9-points')
        return None

    angulo_apertura = angulo_apertura*np.pi/180

    all_precedences = set()
    block_coords = [tuple(coord) for coord in block_coords]
    alturas = mina_rellena['z'].unique()
    

    for block in block_coords:
        x_b, y_b, z_b = block
        alturas_up_block = sorted(alturas[alturas>=z_b])
        new_ids = set()

        dx = (mina_rellena['x']-x_b)/BlockWidth
        dy = (mina_rellena['y']-y_b)/BlockHeight
        if config=='5-points':
            dist = np.abs(dx) + np.abs(dy)
        elif config=='9-points':
            dist = np.maximum(np.abs(dx), np.abs(dy))

        counter = 0
        rel = 0.5
        for z in alturas_up_block:
            rel = ((z - z_b)/BlockHeightZ)*np.arctan(angulo_apertura)
            mask = (dist <= rel) & (mina_rellena['z']==z)
            new_ids.update(list(mina_rellena[mask]['id']))
            # counter+=1
        all_precedences.update(new_ids)
    
    return all_precedences