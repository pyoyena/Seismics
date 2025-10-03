
import numpy as np
import matplotlib.pyplot as plt

from e_utils import segypy
from e_utils import seismicToolBox as seismic


def generate_vmodel(segy_path, cmp_positions, velocity_range=(1500, 7000), velocity_step=25, sortkey=3, smute=0.1):
    """
    segy_path : str
        Path to the SEG-Y file
    cmp_positions : list
        List of CMP positions to analyse
    velocity_range : tuple, optional
        Default: (1500, 7000)
    velocity_step : int, optional
        Default: 25
    sortkey : int, optional
        Default: 3
        Valid values for sortkey are:
        - 1 = Common Shot
        - 2 = Common Receiver
        - 3 = Common Midpoint (CMP)
        - 4 = Common Offset
    smute : float, optional
        Mute parameter for NMO correction. Default: 0.1

    """
    data, sh, sth = segypy.readSegy(segy_path)
    data_cmp, sth_cmp = seismic.sortdata(data, sth, sortkey)

    cmppicks = cmp_positions
    tvpicks = [[], []]

    for cmp in cmppicks:
        data_cmp_i, sth_cmp_i = seismic.selectCMP(data_cmp, sth_cmp, cmp)
        v_semblance, t_semblance = seismic.semblanceWiggle(data_cmp_i, sth_cmp_i, sh, velocity_range[0] , velocity_range[1] , velocity_step)
        tvpicks[0].append(t_semblance)
        tvpicks[1].append(v_semblance)

    positions, folds  = seismic.analysefold(sth_cmp, sortkey)
    vmodel = seismic.generatevmodel2(cmppicks, tvpicks, positions, sh)

    return vmodel

def process_seimics(segy_path, cmp_positions, velocity_range=(1500, 7000), velocity_step=25, sortkey=3, smute=0.1):
    """
    Parametres :
    -----------------------
       segy_path : str
        Path to the SEG-Y file
    cmp_positions : list
        List of CMP positions to analyse
    velocity_range : tuple, optional
        Default: (1500, 7000)
    velocity_step : int, optional
        Default: 25
    sortkey : int, optional
        Default: 3
        Valid values for sortkey are:
        - 1 = Common Shot
        - 2 = Common Receiver
        - 3 = Common Midpoint (CMP)
        - 4 = Common Offset
    smute : float, optional
        Mute parameter for NMO correction. Default: 0.1

    Returns :
    ---------------------------
    data_mig : array
    t_mig : array
    x_mig : array
    zosection : array
    vmodel : array

    Example :
    ---------------------------
    data_mig, t_mig, x_mig, zosection, vmodel = process_seismics()
    seismic.imageseis(data_mig, x=x_mig, t=t_mig)
    plt.title("Final Migrated Seismic Section")
    plt.xlabel("Position (m)")
    plt.ylabel("Time (s)")
    plt.show()    
    """

    # Loading data 
    data, sh, sth = segypy.readSegy(segy_path)
    data_cmp, sth_cmp = seismic.sortdata(data, sth, sortkey)

    # Velocity analysis
    cmppicks = cmp_positions
    tvpicks = [[], []]

    for cmp in cmppicks:
        data_cmp_i, sth_cmp_i = seismic.selectCMP(data_cmp, sth_cmp, cmp)
        v_semblance, t_semblance = seismic.semblanceWiggle(data_cmp_i, sth_cmp_i, sh, velocity_range[0] , velocity_range[1] , velocity_step)
        tvpicks[0].append(t_semblance)
        tvpicks[1].append(v_semblance)

    positions, folds  = seismic.analysefold(sth_cmp, sortkey)
    vmodel = seismic.generatevmodel2(cmppicks, tvpicks, positions, sh)

    # Stacking
    dt = sh["time"][-1] - sh["time"][-2]
    dCMP = sth_cmp[3][-1] -sth_cmp[3][-2]
    t = np.arange(0, sh["time"][-1], dt)
    x = np.arange(sth_cmp[3][0], sth_cmp[3][-1], dCMP)

    zosection = seismic.nmo_stack(data_cmp, sth_cmp, positions, folds, sh, vmodel, smute=0.1)


    # Migration
    data_mig, t_mig, x_mig = seismic.kirk_mig(zosection, vmodel, t, positions)

    return data_mig, t_mig, x_mig, zosection, vmodel