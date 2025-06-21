import numpy as np
import numpy.matlib as matlib
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors 
import time

from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

import plotly.graph_objects as go

from skopt.utils import use_named_args
from skopt.space import Real
from skopt.space import Categorical
from skopt import gp_minimize
from itertools import product
from functools import partial

from pathlib import Path


###############################################################
##################### Plotting Functions ######################
###############################################################

'''
Gráfico de mina/fase en 3D.
Formato de datos:
# puntos = [ (x1, y1, z1), (x2, y2, z2), ... ]
# curva = [ (x1, x2, x3, ...), (y1, y2, y3, ...), (z1, z2, z3, ...) ]
# elipses = [ (a1, b1, x1_c, y1_c, alpha1, z1_c), (a2, b2, x2_c, y2_c, alpha2, z2_c), ... ]
# cono = [(a, b, h, alpha, x_centro, y_centro, , z_sup), ...]
'''
def plot_mina_3D(mina, column_hue='tipomineral', params=dict()):
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

    Theta = np.linspace(0, 2*np.pi, 50)
    i = 1
    for elipse in elipses:
        a, b, x_centro, alpha, y_centro, z_centro = elipse
        X_elipse = a*np.cos(Theta)*np.cos(alpha) - b*np.sin(Theta)*np.sin(alpha) + x_centro
        Y_elipse = a*np.cos(Theta)*np.sin(alpha) + b*np.sin(Theta)*np.cos(alpha) + y_centro
        Z_elipse = np.full_like(Theta, z_centro)

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
            customdata=mina['id'],
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


'''
Gráifco 3D de dos fases bancos con precedencias verticales
Formato de datos:
# fases_bancos = [ fase_banco_lower, fase_banco_upper ] (Lista de DataFrames)
# arcs: estructura entregada por Global_Vertical_Arc_Calculation de aux_tools
# precedence_option: habilita el uso de arcs
'''
def plot_precedencias_verticales_3D(
        fases_bancos, column_hue='cluster', 
        precedence_option=True, arcs={}, 
        width=1000, height=800, z_ratio=5):
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


'''
Gráfico de fases_bancos en 2D
Formato de datos:
# FaseBanco: DataFrame
# highlight_blocks = [ id1, id2, id3, ... ] (Lista con id de bloques)
# points = [ (x1, y1), (x2, y2), (x3, y3), ... ]
# arrows = [ ( (x11, y11), (x12, y12) ), ( (x21, y21), (x22, y22) ), ...]
# precedences: diccionario entregado por Precedencias_Clusters
# centers: centros de clusters entregado por Centros_Clusters
# elipse = [ (a1, b1, x1_centro, y1_centro, alpha1), (a2, b2, x2_centro, y2_centro, alpha2), ... ]
# save_as_image: True si, en vez de mostrar el gráfico, se guarde en un archivo .png. Requiere de un path_file entregado por path_to_save.
'''
def plot_fase_banco(FaseBanco, column_hue='cut', text_hue=None, params=dict(), ax=None):
    if FaseBanco.empty:
        raise ValueError("El DataFrame 'FaseBanco' está vacío. No se puede graficar.")

    params.setdefault('BlockWidthX', 10)
    params.setdefault('BlockWidthY', 10)
    params.setdefault('xsize', 10)
    params.setdefault('ysize', 10)
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

    if not text_hue:
        text_hue = column_hue
    
    xsize=int(xsize)
    ysize=int(ysize)
    dpi=float(dpi)

    # fig, ax = plt.subplots(figsize=(xsize, ysize), dpi=dpi)
    if ax is None:
        fig, ax = plt.subplots(figsize=(xsize, ysize), dpi=dpi)
    else:
        fig = ax.figure
        ax.clear()
    norm = None
    colormap = None
    color_map_discrete = {}
    variables_continuas = FaseBanco.select_dtypes(include='float64').columns.tolist()
    
    fase = FaseBanco['fase'].values[0]
    z = FaseBanco['z'].values[0]
    banco = FaseBanco['banco'].values[0]

    col_data = FaseBanco[column_hue]
    if column_hue in variables_continuas:
        is_continuous = True
    else:
        is_continuous = False
    
    is_continuous = True
    if is_continuous:
        vmin = np.min(col_data)
        vmax = np.max(col_data)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.get_cmap(cmap)
    else:
        if len(col_data.unique())<=1:
            colors = plt.get_cmap('tab20', len(col_data.unique()))
            color_map_discrete = {val: colors(i) for i, val in enumerate(col_data.unique())}
        else:
            colors = plt.get_cmap(cmap, len(col_data.unique()))
            color_map_discrete = {val: colors(i) for i, val in enumerate(col_data.unique())}

    for i, row in FaseBanco.iterrows():
        x_center = row['x']
        y_center = row['y']
        block_value = row[column_hue]

        x_corner = x_center - BlockWidthX / 2
        y_corner = y_center - BlockWidthY / 2

        if is_continuous:
            color = colormap(norm(block_value))
        else:
            color = color_map_discrete.get(block_value, 'gray')
        # print(color)
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

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Fase {fase} - Banco (Z={z}) - Hue {column_hue}')
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
        a, b, alpha, x_centro, y_centro = params
        Theta = np.linspace(0, 2*np.pi, 100)
        X_elipse = a*np.cos(Theta)*np.cos(alpha) - b*np.sin(Theta)*np.sin(alpha) + x_centro
        Y_elipse = a*np.cos(Theta)*np.sin(alpha) + b*np.sin(Theta)*np.cos(alpha) + y_centro

        ax.plot(X_elipse, Y_elipse, color='black')

    if show_legend:
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
            path_to_save = Path(f'fase_{fase}_banco_{banco}.png')
        else:
            path_to_save = Path(path_to_save + f'fase_{fase}_banco_{banco}.png')

        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_to_save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


from copy import deepcopy
def animate_fase_banco(history, column_hue='cluster', params=dict(), save_path=None, fps=2, real_steps=None):
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




###############################################################
#################### Clustering Functions #####################
###############################################################


# Herramienta de utilidad
'''
Calcula el producto de Hadamard (componente a componente) entre una matriz sparse (A) y una densa (B).
Devuelve una matriz sparse (CSR).
'''
def Hadamard_Product_Sparse(A, B):
    
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


'''
Crea la matriz de adyacencia de los bloques de la fase-banco respecto a sus coordenadas x e y.
Considera las dimensiones de los bloques con BlockWidthX ('x') y BlockWidthY ('y').
Permite definir sectores, que son listas de coordenadas (P1,P2,P3,P4) que definen los límites de cada sector para
clusterizar con fronteras (clustering within boundaries).
Devuelve una matriz sparse (CSR).
En caso de definir sectores, depende de Hadamard_Product_Sparse.
'''
def Calculate_Adjency_Matrix(FaseBanco, BlockWidthX=10, BlockWidthY=10, Sectores=[]):
    
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


'''
Calcula la similaridad entre los bloques de la fase-banco, de acuerdo a distancia, ley, destino, tipo de roca 
y/o dirección de minería.
Para no usar un criterio, basta con asignar el valor 0 en el peso o penalización correspondiente.
Devuelve una matriz densa.
'''
def Calculate_Similarity_Matrix(FaseBanco, params = dict()):
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


'''
Encuentra los dos clusters adyacentes más similares entre sí, de acuerdo a la matriz de similaridad y la matriz de adyacencia.
Devuelve una lista con los índices de los clusters más similares. Si no encuentra pares similares, devuelve None.
Depende de la función Hadamard_Product_Sparse.
'''
def Find_Most_Similar_Adjacent_Clusters(AdjencyMatrix, SimilarityMatrix):
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


'''
Realiza un clustering jerárquico y agregativo de la fase-banco.
Average_Desired_Length_Cluster y Max_Cluster_Length son restricciones del tamaño de los clusters.
Depende de la funcion Find_Most_Similar_Adjacent_Clusters.
Devuelve un DataFrame de la fase-banco incluyendo una nueva columna llamada 'cluster' que indica el cluster al que pertenece cada bloque.
'''
def Clustering_Tabesh(FaseBanco, AdjencyMatrix, SimilarityMatrix, params=dict(), animation_mode=False):
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





###############################################################
################# Shape Refinement Functions ##################
###############################################################

'''
Encuentra los bloques esquinas de la fase-banco. Utiliza el criterio de Tabesh (2013) para definir los bloques esquinas.
Devuelve un diccionario cuyas llaves son el id de los bloques esquina y cuyos valores son los clusters con los que es vecino.
Sólo se utiliza en Shape_Refinement_Tabesh.
'''
def Find_Corner_Blocks_Tabesh(FaseBanco, AdjencyMatrix):
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


'''
Refina la forma de los clusters de la fase-banco utilizando el criterio de Tabesh (2013).
Depende de la función Find_Corner_Blocks_Tabesh.
Devuelve el DataFrame de la fase-banco.
'''
def Shape_Refinement_Tabesh(FaseBanco, AdjencyMatrix, Min_Cluster_Length = 10, Iterations_PostProcessing=5, Reset_Clusters_Index=False):
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



'''
Modificación del refinamiento de formas. Además de considerar los corners según Tabesh (2013),
también considera los bloques que tienen 3 vecinos de distintos clusters o sólo 1 vecino de otro cluster.
A diferencia del refinamiento de Tabesh, el cual trabaja por etapas, este refinamiento se realiza de forma
secuencial, lo que lo hace depender del orden de refinado.
'''
def Shape_Refinement_Mod(FaseBanco, AdjencyMatrix, Min_Cluster_Length = 10, Iterations_PostProcessing=5, Reset_Clusters_Index=False):
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
                        # Cluster_to_insert = np.unique_counts(Clusters_Vecinos).values[np.unique_counts(Clusters_Vecinos).counts.argmax()]
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

######################################################
################# Metrics Functions ##################
######################################################

'''
Calcula la homogeneidad del tipo de roca de los clusters de la fase-banco.
Devuelve el promedio del Rock Unity y la distribución del Rock Unity por cluster.
'''

def Rock_Unity(FaseBanco):
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


'''
Calcula la homogeneidad del destino de los clusters de la fase-banco, pesado por el tonelaje de cada bloque.
Devuelve el promedio del DDF y la distribución del DDF por cluster.
'''
def Destination_Dilution_Factor(FaseBanco):
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

'''
Calcula la variación porcentual en torno a la media en la ley de los bloques de cada cluster de la fase-banco.
Devuelve el promedio del CV de los clusters y la distribución del CV por cluster.
'''
def Coefficient_Variation(FaseBanco):
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

###########################################################################################
################# Automatic Creation of Horizontal Cluster's Precedences ##################
###########################################################################################

'''
Calcula los clusters vecinos al cluster especificado.
'''
def Clusters_Vecinos(FaseBanco, Cluster, AdjencyMatrix):
    fase_banco = FaseBanco.copy().reset_index()

    Blocks = fase_banco.loc[fase_banco['cluster'] == Cluster].index
    rows, cols = AdjencyMatrix.nonzero()
    Clusters_Vecinos = []

    for b in Blocks:
            b_is_row = np.where(rows == b)[0]
            for j in b_is_row:
                if fase_banco.iloc[b]['cluster'] != fase_banco.iloc[cols[j]]['cluster']:
                    Clusters_Vecinos.append(fase_banco.iloc[cols[j]]['cluster'].astype(int))

    return Clusters_Vecinos


'''
Calcula los centros geométricos de los clusters.
Retorna un diccionario donde la llave es el ID del cluster y su valor son las coordenadas (x,y) de su centro.
'''
def Centros_Clusters(FaseBanco):
    ID_Clusters = FaseBanco['cluster'].unique()
    Centers = {}
    for id in ID_Clusters:
        Cluster = FaseBanco.loc[FaseBanco['cluster']==id]
        P_center = (Cluster['x'].mean(), Cluster['y'].mean())
        Centers[id] = P_center
    return Centers


'''
Calcula las precedencias de los clusters. Actualiza el dataframe FaseBanco con el nuevo orden entre clusters y también devuelve
un diccionario con las precedencias junto con los centros de los clusters.
Depende de Calculate_Adjency_Matrix.
'''
def Precedencias_Clusters_Agend(FaseBanco, vector_mineria, 
                                BlockWidthX=10, BlockWidthY=10, Distance_Option=True):
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


'''
Calcula las precedencias de los clusters. Actualiza el dataframe FaseBanco con el nuevo orden entre clusters y también devuelve
un diccionario con las precedencias junto con los centros de los clusters.
Depende de Calculate_Adjency_Matrix.
'''
def Precedencias_Clusters_Angle(FaseBanco, vector_mineria, 
                                BlockWidthX=10, BlockWidthY=10, Distance_Option=True, Angle=0):
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



#################################################
################# Fitting Cone ##################
#################################################

# Útil para cálculos de beneficio
'''
Rellena la mina de estéril artificial (tipomineral=-2) de densidad default_density, extendiendo a 100*BlockWidthX (o BlockWidthY) metros hacia los lados de la mina, salvo directamente arriba de la mina, donde rellena con aire (tipomineral=-1) de densidad 0.
'''
def relleno_mina(mina, default_density,
                 BlockWidthX, BlockWidthY, BlockHeightZ,
                 cm, cr, cp, P, FTL, R, ley_corte,
                 relleno_lateral=100):
    
    mina_copy = mina.copy()
    mina_copy['value'] = 0 

    x_min, x_max = mina['x'].min(), mina['x'].max()
    y_min, y_max = mina['y'].min(), mina['y'].max()

    X_values = set(mina['x'].unique())
    Y_values = set(mina['y'].unique())
    Z_values = mina['z'].unique()

    for i in range(1, relleno_lateral + 1):
        X_values.update([x_min - i * BlockWidthX, x_max + i * BlockWidthX])
        Y_values.update([y_min - i * BlockWidthY, y_max + i * BlockWidthY])

    X_values = sorted(X_values)
    Y_values = sorted(Y_values)

    columns = list(mina_copy.columns)
    Coords = list(product(X_values, Y_values, Z_values))

    new_mina = pd.DataFrame(0, index=range(len(Coords)), columns=columns)
    new_mina[['x','y','z']] = pd.DataFrame(Coords, columns=['x','y','z'])
    new_mina['density'] = [default_density]*len(new_mina)

    max_z_por_xy = mina_copy.groupby(['x', 'y'])['z'].max().to_dict()

    def calcular_density(row):
        key = (row['x'], row['y'])

        if key in max_z_por_xy and row['z'] > max_z_por_xy[key]:
            return 0
        else:
            return default_density
        
    def calcular_tipomineral(row):
        key = (row['x'], row['y'])

        if key in max_z_por_xy:
            if row['z'] > max_z_por_xy[key]:
                return -1
            else:
                return -2
        else:
            return -2


    new_mina['density'] = new_mina.apply(calcular_density, axis=1)
    new_mina['tipomineral'] = new_mina.apply(calcular_tipomineral, axis=1)

    claves = ['x','y','z']
    mina_rellena = mina_copy.set_index(claves).combine_first(new_mina.set_index(claves)).reset_index()

    claves_B = set(map(tuple, mina[claves].values))
    mascara_solo_new_mina = ~mina_rellena[claves].apply(tuple, axis=1).isin(claves_B)

    inicio_id = mina['id'].max() + 1
    mina_rellena.loc[mascara_solo_new_mina, 'id'] = range(inicio_id, inicio_id + mascara_solo_new_mina.sum())
    mina_rellena['id'] = mina_rellena['id'].astype(int)

    Block_Vol = BlockWidthX * BlockWidthY * BlockHeightZ

    mina_rellena['value'] = np.where(
        mina_rellena['cut']<ley_corte, 
        -cm*mina_rellena['density']*Block_Vol, 
        (P-cr)*FTL*R*mina_rellena['cut']*(mina_rellena['density']/100.0)*Block_Vol - (cp+cm)*mina_rellena['density']*Block_Vol)

    return mina_rellena



def relleno_mina_angular(mina, default_density,
                 BlockWidthX, BlockWidthY, BlockHeightZ,
                 cm, cr, cp, P, FTL, R, ley_corte,
                 relleno_lateral=100, angulo_cono_deg=10):
    pendiente = np.tan(np.radians(angulo_cono_deg))

    mina_copy = mina.copy()
    mina_copy['value'] = 0 

    x_min, x_max = mina['x'].min(), mina['x'].max()
    y_min, y_max = mina['y'].min(), mina['y'].max()

    X_values = set(mina['x'].unique())
    Y_values = set(mina['y'].unique())
    Z_values = mina['z'].unique()

    for i in range(1, relleno_lateral + 1):
        X_values.update([x_min - i * BlockWidthX, x_max + i * BlockWidthX])
        Y_values.update([y_min - i * BlockWidthY, y_max + i * BlockWidthY])

    X_values = sorted(X_values)
    Y_values = sorted(Y_values)

    columns = list(mina_copy.columns)
    Coords = list(product(X_values, Y_values, Z_values))

    new_mina = pd.DataFrame(0, index=range(len(Coords)), columns=columns)
    new_mina[['x','y','z']] = pd.DataFrame(Coords, columns=['x','y','z'])
    new_mina['density'] = default_density
    new_mina['tipomineral'] = -2

    aire_mask = np.full(len(new_mina), False)

    for _, bloque in mina.iterrows():
        x0, y0, z0 = bloque['x'], bloque['y'], bloque['z']
        dx = new_mina['x'] - x0
        dy = new_mina['y'] - y0
        dz = new_mina['z'] - z0

        dist = np.sqrt(dx**2 + dy**2)

        in_cono = (dz > 0) & (dist <= dz*pendiente)
        aire_mask |= in_cono

    new_mina.loc[aire_mask, 'density'] = 0
    new_mina.loc[aire_mask, 'tipomineral'] = -1
    
    claves = ['x', 'y', 'z']
    mina_rellena = mina_copy.set_index(claves).combine_first(new_mina.set_index(claves)).reset_index()

    claves_B = set(map(tuple, mina[claves].values))
    mascara_solo_new_mina = ~mina_rellena[claves].apply(tuple, axis=1).isin(claves_B)

    inicio_id = mina['id'].max() + 1
    mina_rellena.loc[mascara_solo_new_mina, 'id'] = range(inicio_id, inicio_id + mascara_solo_new_mina.sum())
    mina_rellena['id'] = mina_rellena['id'].astype(int)

    Block_Vol = BlockWidthX * BlockWidthY * BlockHeightZ

    mina_rellena['value'] = np.where(
        mina_rellena['cut']<ley_corte, 
        -cm*mina_rellena['density']*Block_Vol, 
        (P-cr)*FTL*R*mina_rellena['cut']*(mina_rellena['density']/100.0)*Block_Vol - (cp+cm)*mina_rellena['density']*Block_Vol)
    
    return mina_rellena

'''
Identifica los puntos de la mina rellena que están dentro del cono especificado.
Minimum_Area es el área mínima inferior de la mina necesario para operar. En caso de que Minimum_Area>0, no se cuentan los bloques que están en elipses del cono que violan esta restricción.
horizontal_tolerance es un parámetro de tolerancia horizontal de bloques, de modo que se expanden las elipses del cono en horizontal_tolerance metros.
'''
def isin_cone(mina_rellena, params_cono, min_area=0):
    a, b, h, alpha, x_cone, y_cone, z_c  = params_cono
    # z_c = mina_rellena['z'].max()

    index_mina_rellena = mina_rellena.index

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

    points_in = pd.Series(norm_rel<=z_rel**2, name='isin_cone', index=index_mina_rellena)

    if min_area > 0:
        Z = sorted(mina_rellena['z'].unique())
        Z_rel = (h-z_c+Z)/h

        Areas_z = []
        for z in Z_rel:
            if z < 0:
                Areas_z.append(-1e20)
            else:
                Areas_z.append(np.pi*a*b*z**2)

        Areas_z = np.array(Areas_z)

        if min_area >= Areas_z.max():
            Warning('Valor de Minimum_base_area demasiado alto.')
            return pd.Series()
        
        index_z_min = np.where(1 - (Areas_z<min_area))[0].min()

        z_min = Z[index_z_min]

        mask_z = pd.Series(mina_rellena['z']>=z_min, name='z_filter', index=index_mina_rellena)

        points_in = points_in & mask_z

    return points_in


'''
Calcula el beneficio esperado (sin tiempo) del cono especificado
'''
def profit(mina_rellena, params_cono, min_area=0):
    in_cone = isin_cone(mina_rellena, params_cono, min_area=min_area)
    if in_cone.empty:
        return 0
    
    points_in_cone = mina_rellena[in_cone]

    return points_in_cone['value'].sum()


'''
Busca los paráemtros del cono que mejor contiene a la mina, minimizando volumen.
'''
def Best_Cone_by_Volume(mina, max_global_angle=45):
    x_c = mina['x'].mean()
    y_c = mina['y'].mean()
    z_c = mina['z'].max()

    points = np.array(mina[['x', 'y', 'z']])

    a_guess = (2/3)*(mina['x'].max() - mina['x'].min())
    b_guess = (2/3)*(mina['y'].max() - mina['y'].min())
    h_guess = (4/3)*(mina['z'].max() - mina['z'].min())
    x_guess = mina['x'].median()
    y_guess = mina['y'].median()
    alpha_guess = 0

    def objective(params):
        a, b, h, alpha, x_cone, y_cone, z_cone = params
        varepsilon = 1e-2
        if (a <= 0) or (b <= 0) or (h <= 0):
            return 1e20
        return (1/3)*np.pi*a*b*h + varepsilon*abs(x_cone-x_c)**2 + varepsilon*abs(y_cone-y_c)**2
    
    def constrainst_gen(point):
        def constraint(params):
            a, b, h, alpha, x_cone, y_cone, z_cone = params
            A = (a/h)*(h-z_cone+point[2])
            B = (b/h)*(h-z_cone+point[2])
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

    bounds = [(None, None),
          (None, None),
          (None, None),
          (-np.pi/2, np.pi/2),
          (None, None),
          (None, None),
          (z_c, z_c)
    ]
    

    initial_guess = (a_guess, b_guess, h_guess, alpha_guess, x_guess, y_guess, z_c)
    result = sp.optimize.minimize(objective, initial_guess, bounds=bounds,constraints=constraints, method='SLSQP', options={'ftol':1e-9})

    print(result)
    return result


'''
Busca el mejor cono que se ajusta a la mina maximizando profit utilizando gradiente descendente de scipy.
Este cono no necesariamente contiene a toda la mina.
'''
def Best_Cone_by_Profit_sp_minimize(mina, mina_rellena, max_global_angle=45, 
                                    min_area=0, 
                                    method='L-BFGS-B', fd_step=1e-1, ftol=0.01,
                                    x0=[]):
    x_c = mina['x'].mean()
    y_c = mina['y'].mean()
    z_c = mina['z'].max()
    
    a_low = (0.1)*(mina['x'].max() - mina['x'].min())
    a_up = (1)*(mina['x'].max() - mina['x'].min())

    b_low = (0.1)*(mina['y'].max() - mina['y'].min())
    b_up = (1)*(mina['y'].max() - mina['y'].min())

    h_low = (0.1)*(mina['z'].max() - mina['z'].min())
    h_up = (2)*(mina['z'].max() - mina['z'].min())

    lambda_x = 0.4
    x_low = (1-lambda_x)*mina['x'].min() + (lambda_x)*mina['x'].max()
    x_up = (lambda_x)*mina['x'].min() + (1-lambda_x)*mina['x'].max()

    lambda_y = 0.4
    y_low = (1-lambda_y)*mina['y'].min() + (lambda_y)*mina['y'].max()
    y_up = (lambda_y)*mina['y'].min() + (1-lambda_y)*mina['y'].max()


    a_guess = (0.5)*(mina['x'].max() - mina['x'].min())
    b_guess = (0.5)*(mina['y'].max() - mina['y'].min())
    x_guess = mina['x'].median()
    y_guess = mina['y'].median()
    h_guess = (0.75)*(mina['z'].max() - mina['z'].min())
    alpha_guess = 0

    if len(x0)==0:
        x0 = [a_guess, b_guess, h_guess, alpha_guess, x_guess, y_guess, z_c]

    print(x0)
    def objective(params):
        a, b, h, alpha, x_cone, y_cone, z_cone = params
        varepsilon = 1e-2
        if np.arctan(h/a)*180/np.pi > max_global_angle:
            return 1e20
        if np.arctan(h/b)*180/np.pi > max_global_angle:
            return 1e20
        if (a <= 0) or (b <= 0) or (h <= 0):
            return 1e20
        return -profit(mina_rellena, params, min_area=min_area) + varepsilon*abs(x_cone-x_c)**2 + varepsilon*abs(y_cone-y_c)**2

    bounds = [(a_low, a_up),
              (b_low, b_up),
              (h_low, h_up),
              (-np.pi/2, np.pi/2),
              (x_low, x_up),
              (y_low, y_up),
              (z_c, z_c)
              ]

    result = sp.optimize.minimize(objective, x0,
                                  bounds=bounds,
                                  method=method,
                                  jac='3-point',
                                  options={'disp': True, 'finite_diff_rel_step': fd_step, 'ftol': ftol})

    print(result)
    return result

def Best_Cone_by_Profit_diff_evol(mina, mina_rellena, max_global_angle=45, min_area=0, options=dict()):

    options.setdefault('maxiter', 1000)
    options.setdefault('popsize', 15)
    options.setdefault('init', 'latinhypercube')
    options.setdefault('strategy', 'best1bin')
    options.setdefault('mutation', (0.5, 1))
    options.setdefault('recombination', 0.7)
    options.setdefault('x0', [])

    maxiter = options['maxiter']
    popsize = options['popsize']
    init = options['init']
    strategy = options['strategy']
    mutation = options['mutation']
    recombination = options['recombination']
    x0 = options['x0']

    
    x_c = mina['x'].mean()
    y_c = mina['y'].mean()
    z_c = mina['z'].max()

    a_low = (0.5)*(mina['x'].max() - mina['x'].min())
    a_up = (2)*(mina['x'].max() - mina['x'].min())

    b_low = (0.5)*(mina['y'].max() - mina['y'].min())
    b_up = (2)*(mina['y'].max() - mina['y'].min())

    h_low = (0.3)*(mina['z'].max() - mina['z'].min())
    h_up = (2)*(mina['z'].max() - mina['z'].min())

    lambda_x = 0.1
    x_low = (1-lambda_x)*mina['x'].min() + (lambda_x)*mina['x'].max()
    x_up = (lambda_x)*mina['x'].min() + (1-lambda_x)*mina['x'].max()

    lambda_y = 0.1
    y_low = (1-lambda_y)*mina['y'].min() + (lambda_y)*mina['y'].max()
    y_up = (lambda_y)*mina['y'].min() + (1-lambda_y)*mina['y'].max()

    a_guess = (0.75)*(mina['x'].max() - mina['x'].min())
    b_guess = (0.75)*(mina['y'].max() - mina['y'].min())
    x_guess = mina['x'].median()
    y_guess = mina['y'].median()
    h_guess = (1)*(mina['z'].max() - mina['z'].min())
    alpha_guess = 0

    if len(x0)==0:
        x0 = [a_guess, b_guess, h_guess, alpha_guess, x_guess, y_guess, z_c]

    def objective(params):
        a, b, h, alpha, x_cone, y_cone, z_cone = params
        varepsilon = 1e-2
        if np.arctan(h/a)*180/np.pi > max_global_angle:
            return 1e20
        if np.arctan(h/b)*180/np.pi > max_global_angle:
            return 1e20
        if (a <= 0) or (b <= 0) or (h <= 0):
            return 1e20
        return -profit(mina_rellena, params, min_area=min_area) + varepsilon*abs(x_cone-x_c)**2 + varepsilon*abs(y_cone-y_c)**2
    
    bounds = [(a_low, a_up),
              (b_low, b_up),
              (h_low, h_up),
              (-np.pi/2, np.pi/2),
              (x_low, x_up),
              (y_low, y_up),
              (z_c, z_c)
              ]

    result = sp.optimize.differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, disp=True, init=init, strategy=strategy, mutation=mutation, recombination=recombination, x0=x0)

    print(result)
    return result


'''
Identificación de altura mínima de la mina de acuerdo a un criterio de área mínima.
'''
def Z_min(mina, cono, min_area=1e6, debug=False):
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


def Longitud_Rampa(rampa):
    X = np.array(rampa[0])
    Y = np.array(rampa[1])
    Z = np.array(rampa[2])


    dX = X[1:] - X[:-1]
    dY = Y[1:] - Y[:-1]
    dZ = Z[1:] - Z[:-1]


    length = np.sum(np.sqrt(dX**2 + dY**2 + dZ**2))

    return length


'''
Creación de rampa embebida en el cono con cierta pendiente de descenso y ángulo inicial.
'''
def Rampa(cono, params_rampa, n=500, max_vueltas=5, ancho_rampa=30, z_min=None, separacion_switchback=2, return_final_theta=False):
    
    theta_0, descenso, orientacion, z_switchback, switchback_mode, lambda_switchback = params_rampa

    if orientacion >= 0:
        orientacion = 1
    else:
        orientacion = -1

    t_final = 2*max_vueltas*np.pi

    p = -descenso
    a, b, h, alpha_cone, x_cone, y_cone, z_c = cono

    if switchback_mode:
        a_ext = a + (1-lambda_switchback)*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback/h)**(-1))
        b_ext = b + (1-lambda_switchback)*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback/h)**(-1))
        a_int = a - (lambda_switchback)*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback/h)**(-1))
        b_int = b - (lambda_switchback)*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback/h)**(-1))


    T = np.linspace(0, t_final, n)
    X_curve = []
    Y_curve = []
    Z_curve = []

    z = z_c
    theta = theta_0

    if switchback_mode:
        x = (a_ext*np.cos(theta_0)*np.cos(alpha_cone) - b_ext*np.sin(theta_0)*np.sin(alpha_cone)) + x_cone
        y = (a_ext*np.cos(theta_0)*np.sin(alpha_cone) + b_ext*np.sin(theta_0)*np.cos(alpha_cone)) + y_cone
    else:
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
            if switchback_mode:
                R_1 = a_ext*np.cos(theta)*np.cos(alpha_cone) - b_ext*np.sin(theta)*np.sin(alpha_cone)
                dR_1 = (-a_ext*np.sin(theta)*np.cos(alpha_cone) - b_ext*np.cos(theta)*np.sin(alpha_cone))*orientacion
                R_2 = a_ext*np.cos(theta)*np.sin(alpha_cone) + b_ext*np.sin(theta)*np.cos(alpha_cone)
                dR_2 = (-a_ext*np.sin(theta)*np.sin(alpha_cone) + b_ext*np.cos(theta)*np.cos(alpha_cone))*orientacion
            else:
                R_1 = a*np.cos(theta)*np.cos(alpha_cone) - b*np.sin(theta)*np.sin(alpha_cone)
                dR_1 = (-a*np.sin(theta)*np.cos(alpha_cone) - b*np.cos(theta)*np.sin(alpha_cone))*orientacion
                R_2 = a*np.cos(theta)*np.sin(alpha_cone) + b*np.sin(theta)*np.cos(alpha_cone)
                dR_2 = (-a*np.sin(theta)*np.sin(alpha_cone) + b*np.cos(theta)*np.cos(alpha_cone))*orientacion

            A = p**2 - 1 + p**2 * R_1**2/h**2 + p**2 * R_2**2/h**2
            B = (2*p**2/h)*(R_1*dR_1 + R_2*dR_2)*(1-z_c/h + z/h)
            C = p**2*(dR_1**2 + dR_2**2)*(1 - z_c/h + z/h)**2

            dz = ((-B) + np.sqrt( B**2 - 4*A*C ))/(2*A)

            z_new = z + (t-t0)*dz
            theta_new = theta_0 + orientacion*t

            if switchback_mode:
                if z_new<z_switchback:
                    orientacion *= -1

                    S = np.linspace(0, 1, int(np.sqrt(n)))[:-1]
                    x_before = (a_ext*np.cos(theta_new)*np.cos(alpha_cone) - b_ext*np.sin(theta_new)*np.sin(alpha_cone))*(1-z_c/h+z_new/h) + x_cone
                    y_before = (a_ext*np.cos(theta_new)*np.sin(alpha_cone) + b_ext*np.sin(theta_new)*np.cos(alpha_cone))*(1-z_c/h+z_new/h) + y_cone
                    x = (a_int*np.cos(theta_new)*np.cos(alpha_cone) - b_int*np.sin(theta_new)*np.sin(alpha_cone))*(1-z_c/h+z_new/h) + x_cone
                    y = (a_int*np.cos(theta_new)*np.sin(alpha_cone) + b_int*np.sin(theta_new)*np.cos(alpha_cone))*(1-z_c/h+z_new/h) + y_cone

                    X_p = list(x_before + (x - x_before)*S)
                    Y_p = list(y_before + (y - y_before)*S)
                    Z_p = [z_new]*len(S)

                    X_curve = X_curve + X_p
                    Y_curve = Y_curve + Y_p
                    Z_curve = Z_curve + Z_p
                    
                    t0 = t
                    z = z_new
                    theta = theta_new

                    if stop:
                        break

                    if z_min:
                        if z_new < z_min:
                            stop = True
                    
                    cono_sw = [a_int*(1-z_c/h+z_new/h), b_int*(1-z_c/h+z_new/h), h-(z_c-z_new), alpha_cone, x_cone, y_cone, z_new]
                    params_rampa_sw = [theta_new, descenso, orientacion, None, False, 0]

                    if return_final_theta:
                        X, Y, Z, theta = Rampa(cono_sw, params_rampa_sw, n=n, z_min=z_min, max_vueltas=3, return_final_theta=True)
                    else:
                        X, Y, Z = Rampa(cono_sw, params_rampa_sw, n=n, z_min=z_min, max_vueltas=3)
                    X_curve = X_curve + X
                    Y_curve = Y_curve + Y
                    Z_curve = Z_curve + Z
                    break

            if switchback_mode:
                x = (a_ext*np.cos(theta_new)*np.cos(alpha_cone) - b_ext*np.sin(theta_new)*np.sin(alpha_cone))*(1-z_c/h+z_new/h) + x_cone
                y = (a_ext*np.cos(theta_new)*np.sin(alpha_cone) + b_ext*np.sin(theta_new)*np.cos(alpha_cone))*(1-z_c/h+z_new/h) + y_cone
            else:
                x = (a*np.cos(theta_new)*np.cos(alpha_cone) - b*np.sin(theta_new)*np.sin(alpha_cone))*(1-z_c/h+z_new/h) + x_cone
                y = (a*np.cos(theta_new)*np.sin(alpha_cone) + b*np.sin(theta_new)*np.cos(alpha_cone))*(1-z_c/h+z_new/h) + y_cone

            X_curve.append(x)
            Y_curve.append(y)
            Z_curve.append(z_new)

            t0 = t
            z = z_new
            theta = theta_new

            if stop:
                break

            if z_min:
                if z_new < z_min:
                    stop = True
    
    if return_final_theta:
        return X_curve, Y_curve, Z_curve, theta

    return X_curve, Y_curve, Z_curve


def Rampa_2(cono, params_rampa, n=500, max_vueltas=5, ancho_rampa=30, z_min=None, separacion_switchback=2, return_final_theta=False):
    
    theta_0, descenso, orientacion, z_switchback, switchback_mode, lambda_switchback = params_rampa

    if orientacion >= 0:
        orientacion = 1
    else:
        orientacion = -1

    t_final = 2*max_vueltas*np.pi

    p = -descenso
    a, b, h, alpha_cone, x_cone, y_cone, z_c = cono

    if switchback_mode:
        a_ext = a + (1-lambda_switchback)*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback/h)**(-1))
        b_ext = b + (1-lambda_switchback)*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback/h)**(-1))
        a_int = a - (lambda_switchback)*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback/h)**(-1))
        b_int = b - (lambda_switchback)*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback/h)**(-1))


    T = np.linspace(0, t_final, n)
    X_curve = []
    Y_curve = []
    Z_curve = []

    z = z_c
    theta = theta_0

    if switchback_mode:
        x = (a_ext*np.cos(theta_0)*np.cos(alpha_cone) - b_ext*np.sin(theta_0)*np.sin(alpha_cone)) + x_cone
        y = (a_ext*np.cos(theta_0)*np.sin(alpha_cone) + b_ext*np.sin(theta_0)*np.cos(alpha_cone)) + y_cone
    else:
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
            if switchback_mode:
                R_1 = a_ext*np.cos(theta)*np.cos(alpha_cone) - b_ext*np.sin(theta)*np.sin(alpha_cone)
                dR_1 = (-a_ext*np.sin(theta)*np.cos(alpha_cone) - b_ext*np.cos(theta)*np.sin(alpha_cone))*orientacion
                R_2 = a_ext*np.cos(theta)*np.sin(alpha_cone) + b_ext*np.sin(theta)*np.cos(alpha_cone)
                dR_2 = (-a_ext*np.sin(theta)*np.sin(alpha_cone) + b_ext*np.cos(theta)*np.cos(alpha_cone))*orientacion
            else:
                R_1 = a*np.cos(theta)*np.cos(alpha_cone) - b*np.sin(theta)*np.sin(alpha_cone)
                dR_1 = (-a*np.sin(theta)*np.cos(alpha_cone) - b*np.cos(theta)*np.sin(alpha_cone))*orientacion
                R_2 = a*np.cos(theta)*np.sin(alpha_cone) + b*np.sin(theta)*np.cos(alpha_cone)
                dR_2 = (-a*np.sin(theta)*np.sin(alpha_cone) + b*np.cos(theta)*np.cos(alpha_cone))*orientacion

            A = p**2 - 1 + p**2 * R_1**2/h**2 + p**2 * R_2**2/h**2
            B = (2*p**2/h)*(R_1*dR_1 + R_2*dR_2)*(1-z_c/h + z/h)
            C = p**2*(dR_1**2 + dR_2**2)*(1 - z_c/h + z/h)**2

            dz = ((-B) + np.sqrt( B**2 - 4*A*C ))/(2*A)

            z_new = z + (t-t0)*dz
            theta_new = theta_0 + orientacion*t

            if switchback_mode:
                if z_new<z_switchback:
                    orientacion *= -1
                    S = np.linspace(0, 1, int(np.sqrt(n)))[:-1]
                    x_before = (a_ext*np.cos(theta_new)*np.cos(alpha_cone) - b_ext*np.sin(theta_new)*np.sin(alpha_cone))*(1-z_c/h+z_new/h) + x_cone
                    y_before = (a_ext*np.cos(theta_new)*np.sin(alpha_cone) + b_ext*np.sin(theta_new)*np.cos(alpha_cone))*(1-z_c/h+z_new/h) + y_cone
                    x = (a_int*np.cos(theta_new)*np.cos(alpha_cone) - b_int*np.sin(theta_new)*np.sin(alpha_cone))*(1-z_c/h+z_new/h) + x_cone
                    y = (a_int*np.cos(theta_new)*np.sin(alpha_cone) + b_int*np.sin(theta_new)*np.cos(alpha_cone))*(1-z_c/h+z_new/h) + y_cone

                    X_p = list(x_before + (x - x_before)*S)
                    Y_p = list(y_before + (y - y_before)*S)
                    Z_p = [z_new]*len(S)

                    X_curve = X_curve + X_p
                    Y_curve = Y_curve + Y_p
                    Z_curve = Z_curve + Z_p
                    
                    t0 = t
                    z = z_new
                    theta = theta_new

                    if stop:
                        break

                    if z_min:
                        if z_new < z_min:
                            stop = True

                    
                    cono_sw = [a_int*(1-z_c/h+z_switchback/h), b_int*(1-z_c/h+z_switchback/h), h-(z_c-z_switchback), alpha_cone, x_cone, y_cone, z_switchback]
                    params_rampa_sw = [theta_new, descenso, orientacion, None, False, 0]


                    if return_final_theta:
                        X, Y, Z, theta = Rampa_2(cono_sw, params_rampa_sw, n=n, z_min=z_min, max_vueltas=3, return_final_theta=True)
                    else:
                        X, Y, Z = Rampa_2(cono_sw, params_rampa_sw, n=n, z_min=z_min, max_vueltas=3)
                    X_curve = X_curve + X
                    Y_curve = Y_curve + Y
                    Z_curve = Z_curve + Z
                    break

            if switchback_mode:
                x = (a_ext*np.cos(theta_new)*np.cos(alpha_cone) - b_ext*np.sin(theta_new)*np.sin(alpha_cone))*(1-z_c/h+z_new/h) + x_cone
                y = (a_ext*np.cos(theta_new)*np.sin(alpha_cone) + b_ext*np.sin(theta_new)*np.cos(alpha_cone))*(1-z_c/h+z_new/h) + y_cone
            else:
                x = (a*np.cos(theta_new)*np.cos(alpha_cone) - b*np.sin(theta_new)*np.sin(alpha_cone))*(1-z_c/h+z_new/h) + x_cone
                y = (a*np.cos(theta_new)*np.sin(alpha_cone) + b*np.sin(theta_new)*np.cos(alpha_cone))*(1-z_c/h+z_new/h) + y_cone

            X_curve.append(x)
            Y_curve.append(y)
            Z_curve.append(z_new)

            t0 = t
            z = z_new
            theta = theta_new

            if stop:
                break

            if z_min:
                if z_new < z_min:
                    stop = True
    
    if return_final_theta:
        return X_curve, Y_curve, Z_curve, theta

    return X_curve, Y_curve, Z_curve


# Descenso variable con solo 1 switchback
def Rampa_descenso_variable(mina, cono, params_rampa, min_area, n=500, ancho_rampa=30, max_vueltas=5, separacion_switchback=2, output_mode=True, debug=False):
    theta_0, descenso_vec, orientacion, z_switchback, switchback_mode, lambda_switchback = params_rampa

    z_steps = sorted(mina['z'].unique())[::-1]

    X_curve, Y_curve, Z_curve = [], [], []

    theta_actual = theta_0
    orientacion_actual = orientacion

    a, b, h, alpha_cone, x_cone, y_cone, z_c = cono

    z_actual = z_c
    rampas = []

    for i in range(len(descenso_vec)):
        cono_tramo = [a*(1-z_c/h+z_actual/h), b*(1-z_c/h+z_actual/h), h-(z_c-z_actual), alpha_cone, x_cone, y_cone, z_actual]

        z_min_area = Z_min(mina, cono_tramo, min_area=min_area)
        if debug:
            print(f'Area minima:{z_min_area}')

        params_tramo = [theta_actual, descenso_vec[i], orientacion_actual, z_switchback, switchback_mode, lambda_switchback]

        X_i, Y_i, Z_i, theta_actual = Rampa(
            cono_tramo,
            params_tramo,
            n=n,
            z_min=np.max([z_steps[i+1], z_min_area]),
            return_final_theta=True,
            ancho_rampa=ancho_rampa,
            max_vueltas=max_vueltas
        )

        z_actual =  Z_i[-1]
        if (z_actual < z_switchback) and (switchback_mode):
            if debug:
                print(f'Altura del switchback:{z_actual}')
            orientacion_actual *= -1
            switchback_mode = False
            a-= (lambda_switchback)*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback/h)**(-1))
            b-= (lambda_switchback)*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback/h)**(-1))

        rampas.append([X_i, Y_i, Z_i])
        X_curve += X_i
        Y_curve += Y_i
        Z_curve += Z_i
    
    if output_mode:
        return X_curve, Y_curve, Z_curve
    else:
        return rampas


# Descenso variable con switchback vectorial
def Rampa_descenso_variable_2(mina, cono, params_rampa, min_area, n=500, ancho_rampa=30, max_vueltas=5, separacion_switchback=2, output_mode=True, debug=False, correction_factor=1):
    theta_0, descenso_vec, orientacion, z_switchback_vec, switchback_mode, lambda_switchback = params_rampa


    z_steps = sorted(mina['z'].unique())[::-1]

    z_switchback_vec = sorted(z_switchback_vec)[::-1]
    z_switchback_vec = z_switchback_vec[0:switchback_mode]

    if len(z_switchback_vec)==0:
        z_switchback_vec = [0]

    X_curve, Y_curve, Z_curve = [], [], []

    theta_actual = theta_0
    orientacion_actual = orientacion

    a, b, h, alpha_cone, x_cone, y_cone, z_c = cono

    for z in z_switchback_vec[::-1][1:len(z_switchback_vec)]:
        a+= correction_factor*separacion_switchback*ancho_rampa*((1-z_c/h+z/h)**(-1))
        b+= correction_factor*separacion_switchback*ancho_rampa*((1-z_c/h+z/h)**(-1))


    z_actual = z_c
    rampas = []

    for i in range(len(descenso_vec)):
        cono_tramo = [a*(1-z_c/h+z_actual/h), b*(1-z_c/h+z_actual/h), h-(z_c-z_actual), alpha_cone, x_cone, y_cone, z_actual]

        z_min_area = Z_min(mina, cono_tramo, min_area=min_area)
        if debug:
            print(f'Area minima:{z_min_area}')

        params_tramo = [theta_actual, descenso_vec[i], orientacion_actual, z_switchback_vec[0], switchback_mode, lambda_switchback]

        X_i, Y_i, Z_i, theta_actual = Rampa_2(
            cono_tramo,
            params_tramo,
            n=n,
            z_min=np.max([z_steps[i+1], z_min_area]),
            return_final_theta=True,
            ancho_rampa=ancho_rampa,
            max_vueltas=max_vueltas
        )

        z_actual =  Z_i[-1]
        if (z_actual < z_switchback_vec[0]) and (switchback_mode):
            if debug:
                print(f'Altura del switchback:{z_actual}')
            orientacion_actual *= -1

            switchback_mode -= 1

            if switchback_mode>=1:
                a-= correction_factor*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback_vec[0]/h)**(-1))
                b-= correction_factor*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback_vec[0]/h)**(-1))
            else:
                a-= (lambda_switchback)*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback_vec[0]/h)**(-1))
                b-= (lambda_switchback)*separacion_switchback*ancho_rampa*((1-z_c/h+z_switchback_vec[0]/h)**(-1))

            z_switchback_vec = z_switchback_vec[1:len(z_switchback_vec)]
            if len(z_switchback_vec)==0:
                z_switchback_vec = [0]

        # if i>=1:
        #     xi, yi, zi = X_i[0], Y_i[0], Z_i[0]
        #     xj, yj, zj = X_curve[-1], Y_curve[-1], Z_curve[-1]
        #     norm = np.linalg.norm( np.array([xi, yi, zi]) - np.array([xj, yj, zj]) )
        #     print(f'{i}->{i+1}: {norm}')

        rampas.append([X_i, Y_i, Z_i])
        X_curve += X_i
        Y_curve += Y_i
        Z_curve += Z_i
    
    if output_mode:
        return X_curve, Y_curve, Z_curve
    else:
        return rampas


'''
Identifica los id de los bloques que cruzan la rampa, incluyendo precedencias horizontales de minado.
'''
def isin_rampa(mina, mina_rellena, rampa, cono, BlockHeightZ,
            ancho_rampa, ord=np.inf):
    
    id_bloques_in_rampa = set()

    X_curve, Y_curve, Z_curve = rampa
    X_curve = np.array(X_curve)
    Y_curve = np.array(Y_curve)
    Z_curve = np.array(Z_curve) + BlockHeightZ/2
    alturas = np.array(sorted(mina_rellena['z'].unique()))

    a, b, h, alpha, x_c, y_c, z_c = cono
    center = np.array([x_c, y_c])

    x_all = mina_rellena['x'].values
    y_all = mina_rellena['y'].values
    z_all = mina_rellena['z'].values
    id_all = mina_rellena['id'].values
    tm_all = mina_rellena['tipomineral'].values
    filtro_aire = tm_all != -1

    radio = ancho_rampa/2

    for x, y, z in zip(X_curve, Y_curve, Z_curve):
        z_cercano_en_mina = alturas[np.abs((alturas - z)).argmin()]
        z_mask = (z_all == z_cercano_en_mina)
        
        point = np.array([x,y])

        lenght_seg = np.linalg.norm(point - center)
        n_seg = int(np.trunc(2*lenght_seg/ancho_rampa) + 1)
        T = np.linspace(0, 1, n_seg).reshape(-1,1)
        segmento_in = center + (point-center)*T

        for P in segmento_in:
            x_p, y_p = P
            dx = x_all[z_mask] - x_p
            dy = y_all[z_mask] - y_p
            mask_aire = filtro_aire[z_mask]
            
            dist = np.linalg.norm(np.array([dx, dy]), axis=0, ord=ord)
            mask = (dist <= radio) & mask_aire

            ids = id_all[z_mask][mask]
            id_bloques_in_rampa.update(ids)
    
    id_mina = set(mina['id'])

    in_rampa = (id_bloques_in_rampa - id_mina)
    return in_rampa



def total_additional_blocks(mina_rellena, blocks_id, params=dict()):
    
    params.setdefault('BlockWidthX', 10)
    params.setdefault('BlockWidthY', 10)
    params.setdefault('BlockHeightZ', 16)
    params.setdefault('config', '9-points')
    params.setdefault('angulo_apertura_up', 20)
    params.setdefault('angulo_apertura_down', 20)

    BlockWidthX = params['BlockWidthX']
    BlockWidthY = params['BlockWidthY']
    BlockHeightZ = params['BlockHeightZ']
    config = params['config']
    angulo_apertura_up = params['angulo_apertura_up']
    angulo_apertura_down = params['angulo_apertura_down']

    if config not in ('5-points', '9-points'):
        raise ValueError('Solo use config=5-points o config=9-points')


    angulo_apertura_up_rad = np.radians(angulo_apertura_up)
    angulo_apertura_down_rad = np.radians(angulo_apertura_down)
    all_precedences = set()
    all_support = set()

    block_coords = mina_rellena[mina_rellena['id'].isin(blocks_id)][['x','y','z']].to_numpy()

    x_all = mina_rellena['x'].values
    y_all = mina_rellena['y'].values
    z_all = mina_rellena['z'].values
    ids_all = mina_rellena['id'].values
    tm_all = mina_rellena['tipomineral'].values
    filtro_aire = tm_all != -1

    for x_b, y_b, z_b in block_coords:
        dx = (x_all - x_b) / BlockWidthX
        dy = (y_all - y_b) / BlockWidthY

        if config == '5-points':
            dist = np.abs(dx) + np.abs(dy)
        else:
            dist = np.maximum(np.abs(dx), np.abs(dy))

        mask_precedence = (z_all >= z_b) & filtro_aire 
        z_rel_precedence = ((z_all[mask_precedence] - z_b) / BlockHeightZ)*np.tan(angulo_apertura_up_rad)
        dist_precedence = dist[mask_precedence]
        id_precedence = ids_all[mask_precedence]
        mask_final_precedence = dist_precedence <= z_rel_precedence
        all_precedences.update(id_precedence[mask_final_precedence])

        mask_support = (z_all < z_b) & filtro_aire
        z_rel_support = ((z_b - z_all[mask_support] ) / BlockHeightZ)*np.tan(angulo_apertura_down_rad)
        dist_support = dist[mask_support]
        id_support = ids_all[mask_support]
        mask_final_support = dist_support <= z_rel_support
        all_support.update(id_support[mask_final_support])

    return all_precedences, all_support


def Best_Ramp_sp(mina, mina_rellena, params_cono, min_area,
                ancho_rampa, x0=[],
                ref_rampa=300, angulo_apertura_up=20, angulo_apertura_down=20, config='9-points',
                BlockWidthX=10, BlockWidthY=10, BlockHeightZ=16,
                method='L-BFGS-B'):

    z_min = Z_min(mina, params_cono, min_area=min_area)

    bounds = [(-np.pi, np.pi), (0, 0.15), (-1, 1)]
    if len(x0)==0:
        x0 = [0, 0.1, 0.5]

    def objective(params):
        theta, descenso, orientation, z_switchback = params
        rampa = Rampa(params_cono, params, n=ref_rampa, ancho_rampa=ancho_rampa, z_min=z_min)

        id_in_rampa = isin_rampa(mina, mina_rellena, cono=params_cono, BlockHeightZ=BlockHeightZ, rampa=rampa, ancho_rampa=ancho_rampa, ord=np.inf)

        all_precedences, all_support = total_additional_blocks(mina_rellena, id_in_rampa,
                                                  BlockWidthX, BlockWidthY, BlockHeightZ,
                                                  config=config, angulo_apertura_up=angulo_apertura_up,
                                                  angulo_apertura_down=angulo_apertura_down)

        total_ids = (set(mina['id']) | all_precedences) - all_support
        mina_con_rampa = mina_rellena[mina_rellena['id'].isin(total_ids)]

        return -mina_con_rampa['value'].sum()


    result = sp.optimize.minimize(objective, x0, 
                                  bounds=bounds, method=method)

    print(result)
    return result


def Best_Ramp_diff_evol(mina, mina_rellena, params_cono, min_area, ancho_rampa, x0=[],
                        ref_rampa=500, angulo_apertura_up=20, angulo_apertura_down=20, config='9-points',
                        BlockWidthX=10, BlockWidthY=10, BlockHeightZ=16):
    
    z_min = Z_min(mina, params_cono, min_area=min_area)
    bounds = [(-np.pi, np.pi), (0, 0.15), (-1, 1)]

    def objective(params):
        theta, descenso, orientation, z_switchback = params
        rampa = Rampa(params_cono, params, n=ref_rampa, ancho_rampa=ancho_rampa, z_min=z_min)

        id_in_rampa = isin_rampa(mina, mina_rellena, cono=params_cono, BlockHeightZ=BlockHeightZ, rampa=rampa, ancho_rampa=ancho_rampa, ord=np.inf)

        all_precedences, all_support = total_additional_blocks(mina_rellena, id_in_rampa,
                                                  BlockWidthX, BlockWidthY, BlockHeightZ,
                                                  config=config, angulo_apertura_up=angulo_apertura_up,
                                                  angulo_apertura_down=angulo_apertura_down)

        total_ids = (set(mina['id']) | all_precedences) - all_support
        mina_con_rampa = mina_rellena[mina_rellena['id'].isin(total_ids)]

        return -mina_con_rampa['value'].sum()
    
    result = sp.optimize.differential_evolution(objective, bounds, disp=True)

    print(result)
    return result


# def Best_Ramp_gp(mina, mina_rellena, params_cono, min_area, ancho_rampa, x0=[],
#                  ref_rampa=500, angulo_apertura_up=20, angulo_apertura_down=20, config='9-points',
#                  BlockWidth=10, BlockHeight=10, BlockHeightZ=16, n_calls=50, random_state=42):

#     z_min = Z_min(mina, params_cono, Minimum_Area=min_area)

#     space = [
#         Real(-np.pi, np.pi, name='theta'),
#         Real(0.05, 0.15, name='descenso'),
#         Real(-1, 1, name='orientation')
#     ]

#     def objective(params):
#         theta, descenso, orientation, z_switchback = params
#         rampa = Rampa(params_cono, params, n=ref_rampa, ancho_rampa=ancho_rampa, z_min=z_min)

#         id_in_rampa = isin_rampa(mina, mina_rellena, cono=params_cono, BlockHeightZ=BlockHeightZ, rampa=rampa, ancho_rampa=ancho_rampa, ord=np.inf)

#         all_precedences, all_support = total_additional_blocks(mina_rellena, id_in_rampa,
#                                                   BlockWidth, BlockHeight, BlockHeightZ,
#                                                   config=config, angulo_apertura_up=angulo_apertura_up,
#                                                   angulo_apertura_down=angulo_apertura_down)

#         total_ids = (set(mina['id']) | all_precedences) - all_support
#         mina_con_rampa = mina_rellena[mina_rellena['id'].isin(total_ids)]

#         return -mina_con_rampa['value'].sum()

#     result = gp_minimize(
#         func=objective,
#         dimensions=space,
#         n_calls=n_calls,
#         x0=x0 if x0 else None,
#         random_state=random_state,
#         verbose=True
#     )

#     print("Mejor solución encontrada:")
#     print("θ (rad):", result.x[0])
#     print("Descenso:", result.x[1])
#     print("Orientación:", result.x[2])
#     print("Valor objetivo (negativo):", result.fun)

#     return result


# Optimizacion Rampa en espiral
def Best_Ramp_gp(mina, mina_rellena, params_cono, min_area, ancho_rampa, options=dict()):

    options.setdefault('x0', [])
    options.setdefault('ref_rampa', 1000)
    options.setdefault('angulo_apertura_up', 20)
    options.setdefault('angulo_apertura_down', 20)
    options.setdefault('config', '9-points')
    options.setdefault('BlockWidthX', 10)
    options.setdefault('BlockWidthY', 10)
    options.setdefault('BlockHeightZ', 16)
    options.setdefault('n_calls', 50)
    options.setdefault('random_state', 73)

    x0 = options['x0']
    ref_rampa = options['ref_rampa']
    angulo_apertura_up = options['angulo_apertura_up']
    angulo_apertura_down = options['angulo_apertura_down']
    config = options['config']
    BlockWidthX = options['BlockWidthX']
    BlockWidthY = options['BlockWidthY']
    BlockHeightZ = options['BlockHeightZ']
    n_calls = options['n_calls']
    random_state = options['random_state']



    z_min = Z_min(mina, params_cono, min_area=min_area)
    z_max = mina['z'].max()
    alpha_z = 0.1
    z_low = (1-alpha_z)*z_min + alpha_z*z_max
    z_top = (alpha_z)*z_min + (1-alpha_z)*z_max
    space = [
        Real(-np.pi, np.pi, name='theta'),
        Real(0.02, 0.10, name='descenso'),
        Real(-1, 1, name='orientacion'),
        # Categorical([z_low], name='z_switchback'),
        # Categorical([0], name='switchback_mode'),
        # Categorical([0], name='lambda_switchback')
    ]

    def objective(params):
        # z_switchback, switchback_mode = params[3], params[4]
        # z_switchback = None if switchback_mode == 'none' else z_switchback
        params_rampa = (params[0], params[1], params[2], 0, 0, 0)


        rampa = Rampa(params_cono, params_rampa, n=ref_rampa, ancho_rampa=ancho_rampa, z_min=z_min)

        id_in_rampa = isin_rampa(mina, mina_rellena, cono=params_cono, BlockHeightZ=BlockHeightZ, rampa=rampa, ancho_rampa=ancho_rampa, ord=np.inf)

        all_precedences, all_support = total_additional_blocks(mina_rellena, id_in_rampa,
                                                               params={
                                                                   'BlockWidthX': BlockWidthX,
                                                                   'BlockWidthY': BlockWidthY,
                                                                   'BlockHeightZ': BlockHeightZ,
                                                                   'config': config,
                                                                   'angulo_apertura_up': angulo_apertura_up,
                                                                   'angulo_apertura_down': angulo_apertura_down
                                                               })

        total_ids = (set(mina['id']) | all_precedences) - all_support
        mina_con_rampa = mina_rellena[mina_rellena['id'].isin(total_ids)]

        return -mina_con_rampa['value'].sum()

    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        x0=x0 if x0 else None,
        random_state=random_state,
        verbose=True
    )

    print("Mejor solución encontrada:")
    print('[θ, descenso, orientacion, z_switchback, switchback_mode] = ', result.x)
    print("Valor objetivo:", result.fun)

    return result



# Optimizacion rampa en espiral con descenso variable
def Best_Ramp_gp_2(mina, mina_rellena, params_cono, min_area, ancho_rampa, options=dict()):

    options.setdefault('x0', [])
    options.setdefault('ref_rampa', 300)
    options.setdefault('separacion_switchback', 2)
    options.setdefault('angulo_apertura_up', 20)
    options.setdefault('angulo_apertura_down', 20)
    options.setdefault('config', '9-points')
    options.setdefault('BlockWidthX', 10)
    options.setdefault('BlockWidthY', 10)
    options.setdefault('BlockHeightZ', 16)
    options.setdefault('n_calls', 50)
    options.setdefault('n_initial_points', 10)
    options.setdefault('acq_func', 'gp_hedge')
    options.setdefault('random_state', 73)

    x0 = options['x0']
    ref_rampa = options['ref_rampa']
    separacion_switchback = options['separacion_switchback']
    angulo_apertura_up = options['angulo_apertura_up']
    angulo_apertura_down = options['angulo_apertura_down']
    config = options['config']
    BlockWidthX = options['BlockWidthX']
    BlockWidthY = options['BlockWidthY']
    BlockHeightZ = options['BlockHeightZ']
    n_calls = options['n_calls']
    n_initial_points = options['n_initial_points']
    acq_func = options['acq_func']
    random_state = options['random_state']



    z_min = Z_min(mina, params_cono, min_area=min_area)
    z_max = mina['z'].max()
    alpha_z = 0.1
    z_low = (1-alpha_z)*z_min + alpha_z*z_max
    z_top = (alpha_z)*z_min + (1-alpha_z)*z_max


    N = len(mina['z'].unique())-1
    space = [Real(-np.pi, np.pi, name='theta')] + \
            [Real(0.02, 0.10, name=f'descenso_{i}') for i in range(N)] + \
            [Real(-1, 1, name='orientacion'),
            Categorical([z_low], name='z_switchback'),
            Categorical([0], name='switchback_mode'),
            Categorical([0], name='lambda_switchback')]

    def objective(params):
        theta = params[0]
        descenso_vec = params[1:N+1]
        orientacion = params[N+1]
        z_switchback = params[N+2]
        switchback_mode = params[N+3]
        lambda_switchback = params[N+4]

        rampa = Rampa_descenso_variable(mina, params_cono, [theta, descenso_vec, orientacion, z_switchback, switchback_mode, lambda_switchback], min_area, n=ref_rampa, ancho_rampa=ancho_rampa, separacion_switchback=separacion_switchback, output_mode=True)

        id_in_rampa = isin_rampa(mina, mina_rellena, cono=params_cono, BlockHeightZ=BlockHeightZ, rampa=rampa, ancho_rampa=ancho_rampa, ord=np.inf)

        all_precedences, all_support = total_additional_blocks(mina_rellena, id_in_rampa,
                                                               params={
                                                                   'BlockWidthX': BlockWidthX,
                                                                   'BlockWidthY': BlockWidthY,
                                                                   'BlockHeightZ': BlockHeightZ,
                                                                   'config': config,
                                                                   'angulo_apertura_up': angulo_apertura_up,
                                                                   'angulo_apertura_down': angulo_apertura_down
                                                               })

        total_ids = (set(mina['id']) | all_precedences) - all_support
        mina_con_rampa = mina_rellena[mina_rellena['id'].isin(total_ids)]

        return -mina_con_rampa['value'].sum()

    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        x0=x0 if x0 else None,
        random_state=random_state,
        verbose=True,
        n_initial_points=n_initial_points,
        acq_func=acq_func
    )

    print("Mejor solución encontrada:")
    print('[θ, descenso, orientacion, z_switchback, switchback_mode] = ', result.x)
    print("Valor objetivo:", result.fun)

    return result



from skopt.space import Real, Integer, Categorical
def Best_Ramp_gp_3(mina, mina_rellena, params_cono, min_area, ancho_rampa, options=dict()):

    options.setdefault('x0', [])
    options.setdefault('ref_rampa', 300)
    options.setdefault('separacion_switchback', 2)
    options.setdefault('angulo_apertura_up', 20)
    options.setdefault('angulo_apertura_down', 20)
    options.setdefault('config', '9-points')
    options.setdefault('BlockWidthX', 10)
    options.setdefault('BlockWidthY', 10)
    options.setdefault('BlockHeightZ', 16)
    options.setdefault('n_calls', 50)
    options.setdefault('n_initial_points', 10)
    options.setdefault('acq_func', 'gp_hedge')
    options.setdefault('random_state', 73)

    x0 = options['x0']
    ref_rampa = options['ref_rampa']
    separacion_switchback = options['separacion_switchback']
    angulo_apertura_up = options['angulo_apertura_up']
    angulo_apertura_down = options['angulo_apertura_down']
    config = options['config']
    BlockWidthX = options['BlockWidthX']
    BlockWidthY = options['BlockWidthY']
    BlockHeightZ = options['BlockHeightZ']
    n_calls = options['n_calls']
    n_initial_points = options['n_initial_points']
    acq_func = options['acq_func']
    random_state = options['random_state']



    z_min = Z_min(mina, params_cono, min_area=min_area)
    z_max = mina['z'].max()
    alpha_z = 0.05
    z_low = (1-alpha_z)*z_min + alpha_z*z_max
    z_top = (alpha_z)*z_min + (1-alpha_z)*z_max

    print(z_low, z_top)

    N = len(mina['z'].unique())-1
    M = 3
    space = [Real(-np.pi, np.pi, name='theta')] + \
            [Real(0.02, 0.10, name=f'descenso_{i}') for i in range(N)] + \
            [Real(-1, 1, name='orientacion')] + \
            [Real(z_low, z_top, name=f'z_switchback_{i}') for i in range(M)] + \
            [Integer(0, M, name='switchback_mode'),
            Real(0, 1, name='lambda_switchback')]

    def objective(params):
        theta = params[0]
        descenso_vec = params[1:N+1]
        orientacion = params[N+1]
        z_switchback = params[N+2:N+2+M]
        switchback_mode = params[N+2+M]
        lambda_switchback = params[N+2+M+1]

        rampa = Rampa_descenso_variable_2(mina, params_cono, [theta, descenso_vec, orientacion, z_switchback, switchback_mode, lambda_switchback], min_area, n=ref_rampa, ancho_rampa=ancho_rampa, separacion_switchback=separacion_switchback, output_mode=True, correction_factor=(1-1.231)*(lambda_switchback) + 1.231)

        id_in_rampa = isin_rampa(mina, mina_rellena, cono=params_cono, BlockHeightZ=BlockHeightZ, rampa=rampa, ancho_rampa=ancho_rampa, ord=np.inf)

        all_precedences, all_support = total_additional_blocks(mina_rellena, id_in_rampa,
                                                               params={
                                                                   'BlockWidthX': BlockWidthX,
                                                                   'BlockWidthY': BlockWidthY,
                                                                   'BlockHeightZ': BlockHeightZ,
                                                                   'config': config,
                                                                   'angulo_apertura_up': angulo_apertura_up,
                                                                   'angulo_apertura_down': angulo_apertura_down
                                                               })

        total_ids = (set(mina['id']) | all_precedences) - all_support
        mina_con_rampa = mina_rellena[mina_rellena['id'].isin(total_ids)]

        return -mina_con_rampa['value'].sum() + Longitud_Rampa(rampa)

    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        x0=x0 if x0 else None,
        random_state=random_state,
        verbose=True,
        n_initial_points=n_initial_points,
        acq_func=acq_func
    )

    print("Mejor solución encontrada:")
    print('[θ, descenso, orientacion, z_switchback, switchback_mode] = ', result.x)
    print("Valor objetivo:", result.fun)

    return result


'''
Calcula los puntos iniciales de minado respecto a la rampa.
Idealmente debe usarse por fases.
Si z_min no se especifica, se asume que la rampa alcanza el fondo de la mina.
'''
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


def Clustering(mina, cm=2, cr=0.25, cp=10, P=4, R=0.85, ley_corte=0.1423036578621615,
               options=dict()):
    if mina.empty:
        raise ValueError('Inserte un dataframe "mina" no vacio.')
    
    if not('destino' in mina.columns):
        mina['destino'] = [1 if mina.iloc[i]['cut']>= ley_corte else 0 for i in range(len(mina))]

    options.setdefault('BlockWidthX', 10)
    options.setdefault('BlockWidthY', 10)
    options.setdefault('BlockHeightZ', 16)
    options.setdefault('area_minima_operativa', np.pi*80*80)

    # options.setdefault('refinamiento_rampa', 300)
    # options.setdefault('ancho_rampa', 30)


    options.setdefault('peso_distancia', 2)
    options.setdefault('peso_ley', 0)
    options.setdefault('tolerancia_ley', 0.001)
    options.setdefault('peso_direccion_mineria', 0.25)
    options.setdefault('tolerancia_direccion_mineria', 0.001)
    options.setdefault('penalizacion_destino', 0.9)
    options.setdefault('penalizacion_roca', 0.9)

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

    BlockWidthX = options['BlockWidthX']
    BlockWidthY = options['BlockWidthY']
    BlockHeightZ = options['BlockHeightZ']
    area_minima_operativa = options['area_minima_operativa']

    ancho_rampa = options['ancho_rampa']
    ref_ramp = options['refinamiento_rampa']

    
    peso_distancia = options['peso_distancia']
    peso_ley = options['peso_ley']
    tol_ley = options['tolerancia_ley']
    peso_directional_mining = options['peso_direccion_mineria']
    tol_directional_mining = options['tolerancia_direccion_mineria']
    penalizacion_destino = options['penalizacion_destino']
    penalizacion_roca = options['penalizacion_roca']

    
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


    # options.setdefault('cm', 2)
    # cm = options['cm']
    # options.setdefault('cr', 0.25)
    # cr = options['cr']
    # options.setdefault('cp', 10)
    # cp = options['cp']
    # options.setdefault('P', 4)
    # P = options['P']
    # options.setdefault('R', 0.85)
    # R = options['R']
    # FTL = 2204.62
    # options.setdefault('ley_corte', 0.1423036578621615)   # Ley Corte Marginal con los valores default
    # ley_corte = options['ley_corte']
    


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

        if peso_directional_mining > 0:
            cono = list( np.load(Path(path_params + f'cono_{fase}.npy')) )
            # params_rampa = list( np.load(Path(path_params + f'rampa_{fase}.npy')) )

            # z_min = Z_min(mini_mina, cono, area_minima_operativa)
            # rampa = Rampa(cono, params_rampa, n=ref_ramp, z_min=z_min, ancho_rampa=ancho_rampa)

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
                    # p = puntos_iniciales[0]
                    # P1 = (p[0], p[1])
                    # alpha = (p[0] - x_min)/(x_max - x_min)
                    # beta = (p[1] - y_min)/(y_max - y_min)
                    # P2 = (alpha*x_min + (1-alpha)*x_max, beta*y_min + (1-beta)*y_max)
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

            # Max_Cluster_Length = int(0.1*tamaño_fase_banco)
            # Average_Length_Cluster = int((2/3)*Max_Cluster_Length)
            # Min_Cluster_Length = int(0.01*tamaño_fase_banco)

            if Max_Cluster_Length < options['tolerancia_tamaño_maximo_cluster']:
                Max_Cluster_Length = options['tolerancia_tamaño_maximo_cluster']
                Average_Cluster_Length = int(mult*Max_Cluster_Length)
            if Min_Cluster_Length < options['tolerancia_tamaño_minimo_cluster']:
                Min_Cluster_Length = options['tolerancia_tamaño_minimo_cluster']

            print(f'MaxCL: {Max_Cluster_Length}, AveCL: {Average_Cluster_Length}, MinCL: {Min_Cluster_Length}')

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
                    'path': path_image
                })
            
    print(f'Clusters creados: {contador_clusters}')
    print(f'Tiempo total de ejecucion: {sum(Tiempos_clusterizacion_fase_banco)}')

    if save:
        path_arch = Path(path_save + 'mina_clusterizada.csv')
        path_arch.parent.mkdir(parents=True, exist_ok=True)

        mina_clusterizada.to_csv(path_arch, index=False)
        metrics_df.to_csv(Path(path_save + 'metricas.csv'), index=False)
        precedences_df.to_csv(Path(path_save + 'precedencias.csv'), index=False)

        sizes = np.array(Tamaños_fase_banco)
        pd.DataFrame(sizes).to_csv(Path(path_save + 'sizes.csv'), index=False)

        times = np.array(Tiempos_clusterizacion_fase_banco)
        pd.DataFrame(times).to_csv(Path(path_save + 'execution_times.csv'), index=False)
    

    return mina_clusterizada