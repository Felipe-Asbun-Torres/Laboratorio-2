import numpy as np
import pandas as pd
import scipy as sp
import time
import os
import json

from multiprocessing import get_context

import ramp_tools as rt


_global_params = {}

def init_objective(params_dict):
    """Inicializador del Pool: guarda parámetros pesados en variable global"""
    global _global_params
    _global_params = params_dict

def objective(params):
    """Función objetivo compatible con multiprocessing"""
    theta, descenso, orientacion = params[:3]
    shift_raw = params[3:]
    shift = rt.discrete_shift(shift_raw, _global_params['s_min'], _global_params['s_max'])
    # print(shift)

    # if not rt.is_valid_shift(shift):
    #     return 1e12

    shift = rt.valid_shift(shift)

    mina_rellena = pd.DataFrame(_global_params['mina_rellena'])

    alturas = mina_rellena['z'].unique()

    rampa = rt.Rampa_Carril(
        _global_params['cono'], (theta, descenso, orientacion), shift,
        _global_params['min_area'], alturas,
        _global_params['n_por_vuelta'], _global_params['max_vueltas'],
        BlockWidthX=_global_params['BlockWidthX'], BlockWidthY=_global_params['BlockWidthY'],
        arco_seccion=_global_params['arco_seccion'], len_seccion=_global_params['len_seccion']
    )

    min_z_alcanzado = np.array(rampa[2]).min()

    mina_con_rampa = rt.pit_con_rampa(mina_rellena, _global_params['cono'], rampa, _global_params['BlockWidthX'], _global_params['BlockWidthY'], _global_params['BlockHeightZ'], _global_params['ancho_rampa'], _global_params['angulo_apertura_up'], _global_params['angulo_apertura_down'], _global_params['tolerance_pared'])

    # x_rampa, y_rampa, z_rampa = rampa
    # mina_x = mina_rellena['x'].to_numpy()
    # mina_y = mina_rellena['y'].to_numpy()
    # mina_z = mina_rellena['z'].to_numpy()
    # mina_id = mina_rellena['id'].to_numpy()
    # a, b, h, alpha, x_c, y_c, z_c = _global_params['cono']

    # pared_ext, pared_int = rt.pared_rampa_numba(
    #     x_rampa, y_rampa, z_rampa,
    #     mina_x, mina_y, mina_z, mina_id,
    #     x_c, y_c, z_c, a, b, h,
    #     _global_params['ancho_rampa'], _global_params['tolerance_pared']
    # )
    # pared_ext = set(pared_ext)
    # pared_int = set(pared_int)

    # prec, supp = rt.total_additional_blocks_numba(mina_rellena, rampa, params={
    #     'BlockWidthX': _global_params['BlockWidthX'],
    #     'BlockWidthY': _global_params['BlockWidthY'],
    #     'BlockHeightZ': _global_params['BlockHeightZ'],
    #     'angulo_apertura_up': _global_params['angulo_apertura_up'],
    #     'angulo_apertura_down': _global_params['angulo_apertura_down'],
    #     'ancho_rampa': _global_params['ancho_rampa']
    # })

    # prec = set(prec) | pared_int
    # supp = (set(supp) - prec) | pared_ext

    # upit = mina_rellena[rt.isin_cone(mina_rellena, _global_params['cono'])]

    # total_ids = (set(upit['id'].values) | prec) - supp
    # pre_mina_con_rampa = mina_rellena[mina_rellena['id'].isin(total_ids)]

    # boundary = rt.boundary(pre_mina_con_rampa, _global_params['BlockWidthX'], _global_params['BlockWidthY'], _global_params['BlockHeightZ'])
    # prec_boundary = rt.boundary_precedences(mina_rellena, boundary, _global_params['angulo_apertura_up'], _global_params['BlockWidthX'], _global_params['BlockWidthY'], _global_params['BlockHeightZ'])
    # prec_boundary = set(prec_boundary)
    # total_ids = total_ids | prec_boundary
    # mina_con_rampa = mina_rellena[mina_rellena['id'].isin(total_ids)]


    val = - mina_con_rampa[mina_con_rampa['z'] >= min_z_alcanzado]['value'].sum()

    if _global_params['penalizacion_aire'] == 0:
        print(-val)
        return val

    in_rampa = rt.isin_rampa(mina_rellena, rampa, _global_params['BlockHeightZ'], _global_params['ancho_rampa'], filtrar_aire=False)

    blocks_aire = mina_rellena[mina_rellena['id'].isin(in_rampa)]['tipomineral'].value_counts()
    if -1 in blocks_aire.index:
        count_aire = blocks_aire[-1]
    else:
        count_aire = 0

    val += count_aire*_global_params['penalizacion_aire']

    print(-val)
    return val


def Best_Ramp_diff_evol_carril(mina, mina_rellena, cono, min_area, ancho_rampa, n_por_vuelta=100, angulo_apertura_up=20, angulo_apertura_down=20, config='5-points', BlockWidthX=10, BlockWidthY=10, BlockHeightZ=16, arco_seccion=np.pi/6, max_vueltas=3, workers=1, init='sobol', popsize=10, maxiter=10, mutation=(0.5, 1), recombination=0.7, s_min=-1, s_max=1, len_seccion=None, penalizacion_aire=0, tolerance_pared=3):
    # z_min = rt.Z_min(mina, cono, min_area=min_area)
    # N_sh = int((2*np.pi/arco_seccion)*max_vueltas)

    a, b, h, alpha_cone, x_cone, y_cone, z_c = cono
    arcsecc = arco_seccion
    if len_seccion is not None:
        m = ((a-b)**2)/((a+b)**2)
        perimeter = np.pi*(a+b)*(1+3*m/(10+np.sqrt(4-3*m)))
        arco_seccion_loc = (len_seccion/perimeter)*2*np.pi
        arcsecc = np.min((arco_seccion, arco_seccion_loc))
    
    N_shift = int(2*np.pi/(arcsecc)*max_vueltas)+1

    bounds = [ (-np.pi, np.pi), (0.06, 0.12), (-1, 1) ] + [(s_min, s_max)]*N_shift

    params_dict = dict(
        cono=cono,
        min_area=min_area,
        n_por_vuelta=n_por_vuelta,
        max_vueltas=max_vueltas,
        BlockWidthX=BlockWidthX,
        BlockWidthY=BlockWidthY,
        BlockHeightZ=BlockHeightZ,
        arco_seccion=arco_seccion,
        angulo_apertura_up=angulo_apertura_up,
        angulo_apertura_down=angulo_apertura_down,
        ancho_rampa=ancho_rampa,
        mina=mina.to_dict(orient='list'),
        mina_rellena=mina_rellena.to_dict(orient='list'),
        s_min=s_min,
        s_max=s_max,
        len_seccion=len_seccion,
        penalizacion_aire=penalizacion_aire,
        config=config,
        tolerance_pared=tolerance_pared
    )

    if workers == 1:
        result = sp.optimize.differential_evolution(
            objective,
            bounds,
            workers=1,
            init=init,
            popsize=popsize,
            maxiter=maxiter,
            polish=False,
            disp=True,
            mutation=mutation,
            recombination=recombination
        )
    else:
        with get_context("spawn").Pool(processes=workers, initializer=init_objective, initargs=(params_dict,)) as pool:

            result = sp.optimize.differential_evolution(
                objective,
                bounds,
                workers=pool.map,
                init=init,
                popsize=popsize,
                maxiter=maxiter,
                polish=False,
                disp=True,
                mutation=mutation,
                recombination=recombination
            )

    print('\nMejor solución:', result.fun)

    return result

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import scipy as sp
    import time
    import os
    import numba as nb
    import json

    from pathlib import Path

    from multiprocessing import get_context

    import ramp_tools as rt

    ##################################################################
    ##################################################################
    ##################################################################
    # CP Fases
    # R = 0.85
    # P = 4
    # cm = 2
    # cp = 10
    # cr = 0.25
    # FTL = 2204.62
    # alpha_ley_corte = 0 # 0 -> Ley Corte Marginal ... 1 -> Ley Corte Critica

    # BlockWidthX=10
    # BlockWidthY=10
    # BlockHeightZ=16

    # area_minima_operativa = np.pi*80*80
    # ancho_rampa = 30


    # ley_marginal = cp/((P-cr)*FTL*R)
    # ley_critica = (cp+cm)/((P-cr)*FTL*R)

    # ley_corte = ((1-alpha_ley_corte)*ley_marginal + alpha_ley_corte*(ley_critica))*100

    # path_mina = Path('Pruebas_Rampas_Carril/Params/mina.csv')
    # path_mina.parent.mkdir(parents=True, exist_ok=True)

    # if not path_mina.is_file():
    #     mina_cp = pd.read_csv('Pruebas_Rampas_Carril/Params/CP_fases.txt', sep=r"\s+")
    #     mina_cp.rename(columns={'0': 'id'}, inplace=True)

    #     num_fases = mina_cp['fase'].unique()
    #     fases_new = []
    #     for f in num_fases:
    #         fase = mina_cp[mina_cp['fase']==f].copy()
    #         z_sorted = np.sort(fase['z'].unique())[::-1]
    #         bancos = np.array(range(1,len(z_sorted)+1))
    #         z_to_banco = dict(zip(z_sorted, bancos))
    #         fase['banco'] = fase['z'].map(z_to_banco)
    #         fases_new.append(fase)

    #     mina = pd.concat(fases_new)
    #     mina = mina.sort_index()

    #     mina['destino'] = [1 if mina.iloc[i]['cut']>= ley_corte else 0 for i in range(len(mina))]

    #     mina = mina.copy()
    #     mina.to_csv(path_mina, index=False)
    
    # mina = pd.read_csv(path_mina)

    # path_mina_rellena = Path('Pruebas_Rampas_Carril/Params/mina_rellena.csv')
    # path_mina_rellena.parent.mkdir(parents=True, exist_ok=True)

    # # mina = mina[mina['fase']==1].copy()

    # if not path_mina_rellena.is_file():
    #     mina_rellena = rt.relleno_mina(mina, 2.5,
    #                     BlockWidthX, BlockWidthY, BlockHeightZ,
    #                     cm=cm, cr=cr, cp=cp, P=P, FTL=FTL, R=R, ley_corte=ley_corte,
    #                     relleno_lateral=100)

    #     mina_rellena.to_csv(Path(path_mina_rellena), index=False)

    # mina_rellena = pd.read_csv(path_mina_rellena)

    # cono = list(np.load('Pruebas_Rampas_Carril/Params/cono.npy'))

    # popsize = 10
    # maxiter = 20
    # workers = 6
    # init = 'sobol'
    # n_por_vuelta = 200
    # max_vueltas = 3
    # arco_seccion = 2*np.pi
    # len_seccion = 200
    # s_min = -1
    # s_max = 1
    

    # print('Optimización Diff Evol')
    # t1 = time.time()
    # result = Best_Ramp_diff_evol_carril(mina, mina_rellena, cono, area_minima_operativa, ancho_rampa, n_por_vuelta=n_por_vuelta, max_vueltas=max_vueltas, arco_seccion=arco_seccion, workers=workers, popsize=popsize, maxiter=maxiter, init=init, len_seccion=len_seccion, s_min=s_min, s_max=s_max)
    # t2 = time.time()
    # np.save('Pruebas_Rampas_Carril/tiempo_carril.npy', t2-t1)
    # np.save('Pruebas_Rampas_Carril/params_rampa_carril.npy', result.x)
    # np.save('Pruebas_Rampas_Carril/value_rampa_carril.npy', result.fun)
    # print('\nTiempo:', t2-t1)

    # z_min = rt.Z_min(mina, cono, min_area=area_minima_operativa)
    # params = result.x
    # theta, descenso, orientacion = params[:3]
    # shift_raw = params[3:]
    # shift = rt.discrete_shift(shift_raw, s_min, s_max)

    # rampa = rt.Rampa_Carril(cono, (theta, descenso, orientacion), shift, z_min, n_por_vuelta=n_por_vuelta, max_vueltas=max_vueltas, arco_seccion=arco_seccion, len_seccion=len_seccion, BlockWidthX=BlockWidthX, BlockWidthY=BlockWidthY, return_sep=False)
    # rampa_sep = rt.Rampa_Carril(cono, (theta, descenso, orientacion), shift, z_min, n_por_vuelta=n_por_vuelta, max_vueltas=max_vueltas, arco_seccion=arco_seccion, len_seccion=len_seccion, BlockWidthX=BlockWidthX, BlockWidthY=BlockWidthY, return_sep=True)
    # np.save(f'Pruebas_Rampas_Carril/rampa_carril.npy', rampa)

    # with open(f'Pruebas_Rampas_Carril/rampa_carril_div.json', 'w') as arch:
    #     json.dump(rampa_sep, arch)







    ##################################################################
    ##################################################################
    ##################################################################

    # Marvin

    BlockWidthX = 30
    BlockWidthY = 30
    BlockHeightZ = 30
    min_area = np.pi*80*80
    ancho_rampa = 30

    marvin = pd.read_table('Marvin/marvin.blocks', header=None, names=['id', 'x', 'y', 'z', 'tonnage', 'au_ppm', 'cut', 'value'], sep=' ')

    marvin['density'] = marvin['tonnage']/(BlockWidthX*BlockWidthY*BlockHeightZ)
    marvin['x'] = marvin['x']*BlockWidthX
    marvin['y'] = marvin['y']*BlockWidthY
    marvin['z'] = marvin['z']*BlockHeightZ
    marvin['tipomineral'] = 1

    marvin_upit_id = pd.read_csv('Marvin/marvin_upitsol.txt', header=None, names=['id'])
    marvin_upit = marvin[marvin['id'].isin(set(marvin_upit_id['id']))]

    marvin_cpit_id = pd.read_csv('Marvin/marvin_cpit_gmunoz120723sol.txt', header=None, names=['id', 'x_t', 'period', 'fraction'], sep=' ')
    marvin_cpit = marvin[marvin['id'].isin(set(marvin_cpit_id['id']))]

    marvin_pcpsp_id = pd.read_csv('Marvin/marvin_pcpsp_gmunoz120723sol.txt', header=None, names=['id', 'x_t', 'period', 'fraction'], sep=' ')
    marvin_pcpsp = marvin[marvin['id'].isin(set(marvin_pcpsp_id['id']))]

    alturas = marvin_upit['z'].unique()

    marvin_aire = rt.relleno_mina(marvin, 2.266667, BlockWidthX, BlockWidthY, BlockHeightZ, 1, 1, 1, 1, 1, 1, 1, 0)


    popsize = 30
    maxiter = 30
    workers = 6
    init = 'sobol'
    n_por_vuelta = 300
    max_vueltas = 3
    mutation = (1, 1.85)
    recombination = 0.25
    arco_seccion = 6*np.pi
    len_seccion = None
    s_min = 0
    s_max = 0
    angulo_apertura_up = 45
    angulo_apertura_down = 45
    config = '5-points'
    tolerance_pared = 5
    penalizacion_aire = 0

    cono = list(np.load('Marvin/cone.npy'))

    print('Optimización Diff Evol')
    t1 = time.time()
    result = Best_Ramp_diff_evol_carril(marvin_upit, marvin_aire, cono, min_area, ancho_rampa, n_por_vuelta, angulo_apertura_up, angulo_apertura_down, BlockWidthX=BlockWidthX, BlockWidthY=BlockWidthY, BlockHeightZ=BlockHeightZ, arco_seccion=arco_seccion, max_vueltas=max_vueltas, workers=workers, init=init, popsize=popsize, maxiter=maxiter, mutation=mutation, recombination=recombination, s_min=s_min, s_max=s_max, len_seccion=len_seccion, penalizacion_aire=penalizacion_aire, config=config, tolerance_pared=tolerance_pared)

    t2 = time.time()
    np.save('Marvin/tiempo_carril_inf_ang45.npy', t2-t1)
    np.save('Marvin/params_rampa_carril_inf_ang45.npy', result.x)
    np.save('Marvin/value_rampa_carril_inf_ang45.npy', result.fun)
    print('\nTiempo:', t2-t1)

    params = result.x
    theta, descenso, orientacion = params[:3]
    shift_raw = params[3:]
    shift = rt.discrete_shift(shift_raw, s_min, s_max)

    rampa = rt.Rampa_Carril(cono, (theta, descenso, orientacion), shift, min_area, alturas, n_por_vuelta, max_vueltas, return_final_theta_z=False, return_sep=False, BlockWidthX=BlockWidthX, BlockWidthY=BlockWidthY, arco_seccion=arco_seccion, len_seccion=len_seccion)
    rampa_sep = rt.Rampa_Carril(cono, (theta, descenso, orientacion), shift, min_area, alturas, n_por_vuelta, max_vueltas, return_final_theta_z=False, return_sep=True, BlockWidthX=BlockWidthX, BlockWidthY=BlockWidthY, arco_seccion=arco_seccion, len_seccion=len_seccion)

    np.save(f'Marvin/rampa_carril_inf_ang45.npy', rampa)

    with open(f'Marvin/rampa_carril_div_inf_ang45.json', 'w') as arch:
        json.dump(rampa_sep, arch)

    # z_min = rt.Z_min(mina, cono, min_area=area_minima_operativa)
    # params = result.x
    # theta, descenso, orientacion = params[:3]
    # shift_raw = params[3:]
    # shift = rt.discrete_shift(shift_raw, s_min, s_max)

    # rampa = rt.Rampa_Carril(cono, (theta, descenso, orientacion), shift, z_min, n_por_vuelta=n_por_vuelta, max_vueltas=max_vueltas, arco_seccion=arco_seccion, len_seccion=len_seccion, BlockWidthX=BlockWidthX, BlockWidthY=BlockWidthY, return_sep=False)
    # rampa_sep = rt.Rampa_Carril(cono, (theta, descenso, orientacion), shift, z_min, n_por_vuelta=n_por_vuelta, max_vueltas=max_vueltas, arco_seccion=arco_seccion, len_seccion=len_seccion, BlockWidthX=BlockWidthX, BlockWidthY=BlockWidthY, return_sep=True)
    # np.save(f'Pruebas_Rampas_Carril/rampa_carril.npy', rampa)

    # with open(f'Pruebas_Rampas_Carril/rampa_carril_div.json', 'w') as arch:
    #     json.dump(rampa_sep, arch)