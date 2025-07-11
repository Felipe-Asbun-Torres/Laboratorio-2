import numpy as np
import numpy.matlib as matlib
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors 
import time
import os
import numba as nb
import numbers

from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

import plotly.graph_objects as go

from skopt.utils import use_named_args
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize
from itertools import product
from functools import partial

from pathlib import Path
from copy import deepcopy

from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count
from collections import defaultdict



###############################################################
########################### Conos #############################
###############################################################

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

    a_low = (0.1)*(mina['x'].max() - mina['x'].min())
    a_up = (2)*(mina['x'].max() - mina['x'].min())

    b_low = (0.1)*(mina['y'].max() - mina['y'].min())
    b_up = (2)*(mina['y'].max() - mina['y'].min())

    h_low = (0.1)*(mina['z'].max() - mina['z'].min())
    h_up = (2)*(mina['z'].max() - mina['z'].min())

    lambda_x = 0.1
    x_low = (1-lambda_x)*mina['x'].min() + (lambda_x)*mina['x'].max()
    x_up = (lambda_x)*mina['x'].min() + (1-lambda_x)*mina['x'].max()

    lambda_y = 0.1
    y_low = (1-lambda_y)*mina['y'].min() + (lambda_y)*mina['y'].max()
    y_up = (lambda_y)*mina['y'].min() + (1-lambda_y)*mina['y'].max()

    print(a_low, a_up)
    print(b_low, b_up)
    print(h_low, h_up)
    print(x_low, x_up)
    print(y_low, y_up)

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



###############################################################
########################### Rampas ############################
###############################################################

def Rampa(cono, params_rampa, z_min=None, n_por_vuelta=100, max_vueltas=3, 
          return_final_theta_z=False, shift=0, BlockWidthX=10, BlockWidthY=10, z_top=None):
    """
    Genera la trayectoria tridimensional (rampa helicoidal) dentro de un cono truncado.

    Parámetros
    ----------
    cono : tuple
        Parámetros del cono (a, b, h, alpha, x_c, y_c, z_c).
    params_rampa : tuple
        Parámetros de la rampa: (theta_0, descenso, orientación).
    z_min : float, opcional
        Cota inferior en Z para detener la generación de la rampa.
    n_por_vuelta : int, default=100
        Número de pasos por vuelta helicoidal.
    max_vueltas : float, default=3.5
        Número máximo de vueltas que puede dar la rampa.
    return_final_theta_z : bool, default=False
        Si es True, también retorna el ángulo y altura final de la rampa.
    shift : float or tuple, default=0
        Desplazamiento radial en función de theta (puede ser fijo o un interpolador lineal).
    BlockWidthX : float, default=10
        Ancho de bloque en X (para calcular orientación elíptica).
    BlockWidthY : float, default=10
        Ancho de bloque en Y (para calcular orientación elíptica).
    z_top : float, opcional
        Nueva cota superior del cono si se desea truncarlo desde arriba.

    Retorna
    -------
    X_curve, Y_curve, Z_curve : list of float
        Coordenadas 3D de la curva de la rampa.
    theta, z : float, opcional
        Solo si `return_final_theta_z=True`. Devuelve el ángulo y la altura final.
    """
    theta_0, descenso, orientacion = params_rampa

    if orientacion >= 0:
        orientacion = 1
    else:
        orientacion = -1

    p = -descenso
    a, b, h, alpha_cone, x_cone, y_cone, z_c = cono

    if z_top is not None:
        a = a*(1 - z_c/h + z_top/h)
        b = b*(1 - z_c/h + z_top/h)
        h = h - (z_c - z_top)
        z_c = z_top

    N = n_por_vuelta

    T = np.linspace(0, N, N)

    X_curve = []
    Y_curve = []
    Z_curve = []

    z = z_c
    theta = theta_0

    dtheta = orientacion*2*np.pi*max_vueltas/n_por_vuelta

    ell_theta_x = BlockWidthX*np.cos(theta+alpha_cone)
    ell_theta_y = BlockWidthY*np.sin(theta+alpha_cone)

    norm_ell = np.linalg.norm((ell_theta_x, ell_theta_y))
    if isinstance(shift, tuple):
        prev_sh, post_sh = shift
        shift = np.linspace(prev_sh, post_sh, N)
    else:
        shift = tuple([shift])*N

    x = (a*np.cos(theta_0)*np.cos(alpha_cone) - b*np.sin(theta_0)*np.sin(alpha_cone)) 
    y = (a*np.cos(theta_0)*np.sin(alpha_cone) + b*np.sin(theta_0)*np.cos(alpha_cone)) 

    x = x*(1+shift[0]*norm_ell*np.cos(theta+alpha_cone)/np.linalg.norm((x,y))) + x_cone
    y = y*(1+shift[0]*norm_ell*np.sin(theta+alpha_cone)/np.linalg.norm((x,y))) + y_cone

    X_curve.append(x); Y_curve.append(y); Z_curve.append(z)

    stop = False
    i = 0
    for t in T:
        if t == 0:
            t0 = 0
            i+=1
        else:
            R_1 = a*np.cos(theta)*np.cos(alpha_cone) - b*np.sin(theta)*np.sin(alpha_cone)
            dR_1 = (-a*np.sin(theta)*np.cos(alpha_cone) - b*np.cos(theta)*np.sin(alpha_cone))*dtheta
            R_2 = a*np.cos(theta)*np.sin(alpha_cone) + b*np.sin(theta)*np.cos(alpha_cone)
            dR_2 = (-a*np.sin(theta)*np.sin(alpha_cone) + b*np.cos(theta)*np.cos(alpha_cone))*dtheta

            A = p**2 - 1 + p**2 * R_1**2/h**2 + p**2 * R_2**2/h**2
            B = (2*p**2/h)*(R_1*dR_1 + R_2*dR_2)*(1-z_c/h + z/h)
            C = p**2*(dR_1**2 + dR_2**2)*(1 - z_c/h + z/h)**2

            dz = ((-B) + np.sqrt( B**2 - 4*A*C ))/(2*A)

            z_new = z + (t-t0)*dz
            theta_new = theta + (t-t0)*dtheta

            ell_theta_x = BlockWidthX*np.cos(theta_new+alpha_cone)
            ell_theta_y = BlockWidthY*np.sin(theta_new+alpha_cone)

            norm_ell = np.linalg.norm((ell_theta_x, ell_theta_y))

            x = (a*np.cos(theta_new)*np.cos(alpha_cone) - b*np.sin(theta_new)*np.sin(alpha_cone))*(1-z_c/h+z_new/h)
            y = (a*np.cos(theta_new)*np.sin(alpha_cone) + b*np.sin(theta_new)*np.cos(alpha_cone))*(1-z_c/h+z_new/h)

            x = x*(1+shift[i]*norm_ell*np.cos(theta_new+alpha_cone)/np.linalg.norm((x,y))) + x_cone
            y = y*(1+shift[i]*norm_ell*np.sin(theta_new+alpha_cone)/np.linalg.norm((x,y))) + y_cone

            X_curve.append(x); Y_curve.append(y); Z_curve.append(z_new)

            t0 = t
            z = z_new
            theta = theta_new

            if stop:
                break

            if z_min is not None:
                if z_new < z_min:
                    stop = True
            i+=1
    
    if return_final_theta_z:
        return X_curve, Y_curve, Z_curve, theta, z

    return X_curve, Y_curve, Z_curve


def Rampa_Carril(cono, params_rampa, shift, z_min=None, n_por_vuelta=100, max_vueltas=3, 
                 return_final_theta_z=False, 
                 BlockWidthX=10, BlockWidthY=10, arco_seccion=np.pi/2
                 ):
    """
    Genera una rampa compuesta por múltiples segmentos helicoidales que pueden variar su desplazamiento lateral.

    La rampa es construida por secciones angulares (arco_seccion), permitiendo cambiar el parámetro `shift` en cada trozo.
    Utiliza internamente la función `Rampa` para generar cada segmento helicoidal.

    Parámetros
    ----------
    cono : tuple
        Parámetros del cono de la forma (a, b, h, alpha, x_c, y_c, z_c).

    params_rampa : tuple
        Parámetros iniciales de la rampa: (theta_0, descenso, orientacion).

    shift : list or tuple
        Lista de valores de desplazamiento lateral. Puede variar entre segmentos.

    z_min : float, opcional
        Altura mínima deseada. El algoritmo se detiene al alcanzarla.

    n_por_vuelta : int, default=100
        Número de puntos por vuelta completa.

    max_vueltas : float, default=3
        Número máximo de vueltas a generar.

    return_final_theta_z : bool, default=False
        Si es True, también retorna el ángulo y altura final de la rampa.

    BlockWidthX : float, default=10
        Ancho del bloque en X, usado para escalar el desplazamiento.

    BlockWidthY : float, default=10
        Ancho del bloque en Y, usado para escalar el desplazamiento.

    arco_seccion : float, default=π/2
        Ángulo (en radianes) de cada sección de rampa.

    Retorna
    -------
    X_curve, Y_curve, Z_curve : list
        Coordenadas 3D de la rampa.

    theta_f, z_f : float
        (Opcional) Ángulo y altura final, si `return_final_theta_z` es True.
    """
    theta_0, descenso, orientacion = params_rampa

    X_curve = []; Y_curve = []; Z_curve = []

    N_shift = int(2*np.pi/(arco_seccion)*max_vueltas)
    len_shift = len(shift)
    if len_shift < N_shift:
        shift = ((N_shift//len_shift + 1)*tuple(shift))[:N_shift]

    if orientacion >= 0:
        orientacion = 1
    else:
        orientacion = -1

    n_trozo = int((arco_seccion/(2*np.pi))*n_por_vuelta)

    i = 0
    for sh in shift:
        if i==0:
            x_rampa, y_rampa, z_rampa, theta_f, z_f = Rampa(cono, params_rampa, 
                                                       z_min=z_min, n_por_vuelta=n_trozo, max_vueltas=arco_seccion/(2*np.pi),
                                                       return_final_theta_z=True, shift=sh,
                                                       BlockWidthX=BlockWidthX, BlockWidthY=BlockWidthY)
            
            X_curve += x_rampa; Y_curve += y_rampa; Z_curve += z_rampa
            previous_shift = sh

            
        else:
            if previous_shift != sh:
                s = (previous_shift, sh)
            else:
                s = sh
            
            x_rampa, y_rampa, z_rampa, theta_f, z_f = Rampa(cono, (theta_f, descenso, orientacion), 
                                                    z_min=z_min, n_por_vuelta=n_trozo, max_vueltas=arco_seccion/(2*np.pi),
                                                    return_final_theta_z=True, shift=s,
                                                    BlockWidthX=BlockWidthX, BlockWidthY=BlockWidthY,
                                                    z_top=z_f)
            
            X_curve += x_rampa; Y_curve += y_rampa; Z_curve += z_rampa
            previous_shift = sh

        i+=1

    if return_final_theta_z:
        return X_curve, Y_curve, Z_curve, theta_f, z_f
    
    return X_curve, Y_curve, Z_curve


def isin_rampa(mina, mina_rellena, rampa, BlockHeightZ,
            ancho_rampa, ord=np.inf):
    X_curve, Y_curve, Z_curve = rampa
    Z_curve = np.array(Z_curve) + BlockHeightZ / 2
    alturas = np.array(sorted(mina_rellena['z'].unique()))

    radio = ancho_rampa / 2
    filtro_aire = mina_rellena['tipomineral'].values != -1

    id_bloques_in_rampa = set()
    mina_rellena_filtrada = mina_rellena[filtro_aire]

    # Preconstruir árboles por cada nivel Z
    trees_por_z = {}
    for z in alturas:
        mask_z = (mina_rellena_filtrada['z'].values == z)
        coords = mina_rellena_filtrada.loc[mask_z, ['x', 'y']].values
        if len(coords) > 0:
            trees_por_z[z] = (cKDTree(coords), mina_rellena_filtrada.loc[mask_z, 'id'].values)

    # Buscar vecinos cercanos para cada punto de la rampa
    for x, y, z in zip(X_curve, Y_curve, Z_curve):
        z_cercano = alturas[np.abs(alturas - z).argmin()]
        if z_cercano not in trees_por_z:
            continue

        tree, ids = trees_por_z[z_cercano]
        idxs = tree.query_ball_point([x, y], r=radio, p=np.inf if ord==np.inf else ord)
        id_bloques_in_rampa.update(ids[idxs])

    # Calcular bloques "nuevos" que entran a la rampa
    in_rampa = id_bloques_in_rampa
    return in_rampa


@nb.njit(cache=True, fastmath=True)
def precedences_supp(
        x_all, y_all, z_all, ids, tipomineral,
        x_r, y_r, z_r,
        inv_bw_x, inv_bw_y, inv_bh_z,
        tan_angle_up, tan_angle_down, h_up, h_down):

    n_blocks   = x_all.size
    is_precise = np.zeros(n_blocks, dtype=np.bool_)
    is_supp = np.zeros(n_blocks, dtype=np.bool_)

    not_aire = tipomineral != -1

    for j in range(x_r.size):
        dx = np.abs(x_all - x_r[j]) * inv_bw_x
        dy = np.abs(y_all - y_r[j]) * inv_bw_y
        dist = np.maximum(dx, dy)

        arriba = (z_all >= z_r[j]) & not_aire
        z_rel  = ((z_all - (z_r[j] - h_up)) * inv_bh_z) * tan_angle_up

        abajo = (z_all < z_r[j]) & not_aire
        z_rel_down = (((z_r[j] + h_down) - z_all) * inv_bh_z) * tan_angle_down

        is_precise |= arriba & (dist <= z_rel)
        is_supp |= abajo & (dist <= z_rel_down)
    return ids[is_precise], ids[is_supp]

def total_additional_blocks_numba(mina_rellena, rampa, params=dict()):
    params.setdefault('BlockWidthX', 10)
    params.setdefault('BlockWidthY', 10)
    params.setdefault('BlockHeightZ', 16)
    params.setdefault('config', '9-points')
    params.setdefault('angulo_apertura_up', 20)
    params.setdefault('angulo_apertura_down', 20)
    params.setdefault('ancho_rampa', 30)

    BlockWidthX = params['BlockWidthX']
    BlockWidthY = params['BlockWidthY']
    BlockHeightZ = params['BlockHeightZ']
    config = params['config']
    angulo_apertura_up = np.radians(params['angulo_apertura_up'])
    angulo_apertura_down = np.radians(params['angulo_apertura_down'])
    ancho_rampa = params['ancho_rampa']

    x_all = mina_rellena['x'].values.astype(np.float32)
    y_all = mina_rellena['y'].values.astype(np.float32)
    z_all = mina_rellena['z'].values.astype(np.float32)
    ids_all = mina_rellena['id'].values.astype(np.int32)

    tm_all = mina_rellena['tipomineral'].values.astype(np.int8)

    tan_ang_up = np.tan(angulo_apertura_up)
    tan_ang_down = np.tan(angulo_apertura_down)
    inv_bw_x = 1.0/BlockWidthX
    inv_bw_y = 1.0/BlockWidthY
    inv_bh_z = 1.0/BlockHeightZ
    h_up = ancho_rampa/(2.0*tan_ang_up)
    h_down = ancho_rampa/(2.0*tan_ang_down)

    prec_ids, supp_ids = precedences_supp(
        x_all, y_all, z_all, ids_all, tm_all,
        np.array(rampa[0]).astype(np.float32),
        np.array(rampa[1]).astype(np.float32),
        np.array(rampa[2]).astype(np.float32),
        inv_bw_x, inv_bw_y, inv_bh_z,
        tan_ang_up, tan_ang_down, h_up, h_down
    )

    return prec_ids, supp_ids


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


def plot_perfil_mina_rampa(mina, rampa=None, cono=None, eje='x', cmap='tab10'):
    plt.figure(figsize=(12, 6))

    if rampa is not None:
        X_rampa, Y_rampa, Z_rampa = rampa

    if eje == 'x':
        eje_horizontal = mina['x']
        label = 'X'
        if rampa is not None:
            eje_rampa = X_rampa
    elif eje == 'y':
        eje_horizontal = mina['y']
        label = 'Y'
        if rampa is not None:
            eje_rampa = Y_rampa
    else:
        raise ValueError("El eje debe ser 'x' o 'y'")

    # Paredes de la mina
    fases = np.sort(mina['fase'].unique())
    n_fases = len(fases)

    cmap_obj = plt.get_cmap(cmap, n_fases)

    fase_color = {f: cmap_obj(i) for i, f in enumerate(fases)}
    z = mina['z'].to_numpy()

    for fase in fases:
        mask = mina['fase']==fase
        plt.scatter(eje_horizontal[mask],
                    z[mask],
                    c=[fase_color[fase]],
                    s=20, alpha=0.3,
                    label=f'Fase {fase}'
                    )
    
    # plt.scatter(eje_horizontal, mina['z'], c='gray', alpha=0.3, s=3, label='Mina')

    # Rampa
    if rampa is not None:
        plt.plot(eje_rampa, Z_rampa, c='red', lw=2, label='Rampa')

    

    if cono is not None:
        a, b, h, alpha, x_centro, y_centro, z_sup = cono
        Theta = np.linspace(0, 2 * np.pi, 200)
        Z = np.linspace(z_sup - h, z_sup, 100)
        Theta, Z = np.meshgrid(Theta, Z)

        X_cono = ((a / h) * np.cos(Theta) * np.cos(alpha) - (b / h) * np.sin(Theta) * np.sin(alpha)) * (h - z_sup + Z) + x_centro
        Y_cono = ((a / h) * np.cos(Theta) * np.sin(alpha) + (b / h) * np.sin(Theta) * np.cos(alpha)) * (h - z_sup + Z) + y_centro

        if eje == 'x':
            contorno_izq = X_cono.min(axis=1)
            contorno_der = X_cono.max(axis=1)
        else:  # eje == 'y'
            contorno_izq = Y_cono.min(axis=1)
            contorno_der = Y_cono.max(axis=1)
        
        plt.plot(contorno_izq, Z[:, 0], c='orange', lw=1.5, ls='--', label='Cono (contorno)')
        plt.plot(contorno_der, Z[:, 0], c='orange', lw=1.5, ls='--')


    plt.xlabel(f'Coordenada {label}')
    plt.ylabel('Altura Z')
    plt.title(f'Vista de perfil ({label}-Z)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_perfil_superior(mina, mina_rellena, rampa,
                         BlockWidthX=10, BlockWidthY=10, BlockHeightZ=16, ancho_rampa=30, block_size=5,
                         all_precedences=set(), width=900, height=800):
    id_in_rampa = isin_rampa(mina, mina_rellena, rampa, BlockHeightZ, ancho_rampa)


    x_min = mina['x'].min() - BlockWidthX
    x_max = mina['x'].max() + BlockWidthX
    y_min = mina['y'].min() - BlockWidthY
    y_max = mina['y'].max() + BlockWidthY

    ids_mina = set(mina['id'])

    mask = (~mina_rellena['id'].isin(ids_mina)) & (mina_rellena['x']<=x_max) & (mina_rellena['x']>=x_min) & (mina_rellena['y']>=y_min) & (mina_rellena['y']<=y_max) & (~mina_rellena['id'].isin(id_in_rampa)) & (mina_rellena['tipomineral']!=-1) & (~mina_rellena['id'].isin(all_precedences))

    notmina = mina_rellena[mask]
    blocks_rampa = mina_rellena[mina_rellena['id'].isin(id_in_rampa)]
    
    

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=notmina['x'], y=notmina['y'], z=notmina['z'],
        mode='markers',
        marker=dict(
            size=block_size,
            color=notmina['z'],
            colorscale='jet'
        )
    ))

    fig.add_trace(go.Scatter3d(
        x=blocks_rampa['x'], y=blocks_rampa['y'], z=blocks_rampa['z'],
        mode='markers',
        marker=dict(
            size=block_size,
            color='black'
        )
    ))

    fig.update_layout(
        width=width,
        height=height
    )


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
####################### Extra Functions #######################
###############################################################

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


def Puntos_Iniciales(mina, rampa, 
                     z_min=None, debug=False):
    """
    Obtiene los puntos iniciales de excavación desde una curva de rampa, a partir de un nivel mínimo.

    Parámetros
    ----------
    mina : pandas.DataFrame
        DataFrame que contiene al menos la columna 'z' (altitud).
    rampa : tuple of arrays
        Tupla con tres arrays (X_curve, Y_curve, Z_curve) que describen la curva de la rampa.
    z_min : float, opcional
        Altura mínima a considerar. Si no se especifica, se usa el valor mínimo de 'z' en `mina`.
    debug : bool, opcional
        Si es True, imprime información útil para depuración.

    Retorna
    -------
    puntos_iniciales : list of tuple
        Lista de tuplas (x, y, z) con las coordenadas de los puntos iniciales a distintas alturas.
    """
    X_curve, Y_curve, Z_curve = rampa

    if z_min is None:
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
        (P-cr)*FTL*R*(mina_rellena['cut']/100.0)*(mina_rellena['density'])*Block_Vol - (cp+cm)*mina_rellena['density']*Block_Vol)

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


def profit(mina_rellena, params_cono, min_area=0):
    in_cone = isin_cone(mina_rellena, params_cono, min_area=min_area)
    if in_cone.empty:
        return 0
    
    points_in_cone = mina_rellena[in_cone]

    return points_in_cone['value'].sum()


def Longitud_Rampa(rampa):
    X = np.array(rampa[0])
    Y = np.array(rampa[1])
    Z = np.array(rampa[2])


    dX = X[1:] - X[:-1]
    dY = Y[1:] - Y[:-1]
    dZ = Z[1:] - Z[:-1]


    length = np.sum(np.sqrt(dX**2 + dY**2 + dZ**2))

    return length

