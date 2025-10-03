Oveview 
-------------------
This package provides a complete workflow for seismic data processing in four steps:
1) data loading
2) velocity analysis
3) stacking
4) migration


Installation
-------------------
pip install numpy matplotlib

numpy - Numerical computations
matplotlib - Visualization
e_utils.segypy - SEG-Y file reading utilities
e_utils.seismicToolBox - Seismic processing functions


Core Functions
-------------------
generate_vmodel(segy_path, cmp_positions, velocity_range=(1500, 7000), velocity_step=25, sortkey=3, smute=0.1)

Parameters:
- segy_path (str): Path to the SEG-Y file
- cmp_positions (list): List of CMP positions to analyse
- velocity_range (tuple): Velocity search range in m/s (default: (1500, 7000))
- velocity_step (int): Velocity increment for analysis (default: 25)
- sortkey (int): Data sorting method (1=Shot, 2=Receiver, 3=CMP, 4=Offset)
- smute (float): Mute parameter for NMO correction (default: 0.1)

Returnss:
vmodel (array): Generated velocity model


process_seismics(segy_path, cmp_positions, velocity_range=(1500, 7000), velocity_step=25, sortkey=3, smute=0.1)

Parameters:
Same as generate_vmodel

Returns:
data_mig (array): Migrated seismic section
t_mig (array): Time axis for migrated section
x_mig (array): Position axis for migrated section
zosection (array): Stacked section (zero-offset) before migration
vmodel (array): Velocity model used for processing


Usage Example
-------------------
import numpy as np
import matplotlib.pyplot as plt
from functions import process_seismics

data_mig, t_mig, x_mig, zosection, vmodel = process_seismics(
    "/path/to/your/data.segy",
    cmp_positions=[2002, 3002, 4002, 5002, 6002]
    velocity_range=(1500, 7000),
    velocity_step=25
)

plt.figure()
seismic.imageseis(data_mig, x=x_mig, t=t_mig)
plt.title("Final Migrated Seismic Section")
plt.xlabel("Position (m)")
plt.ylabel("Time (s)")
plt.show()


Notes
-------------------
- Ensure SEG-Y files are properly formatted (data, sh, sth)
- CMP positions should exist in your dataset (verify with header analysis)


