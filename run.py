import sys
import os
print(os.environ.get('QT_API'))

parent_dir = "/Users/yena/Desktop/Integrated Exercise"
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
import matplotlib
matplotlib.use("QtAgg") 
import matplotlib.pyplot as plt

from e_utils import segypy
from e_utils import seismicToolBox as seismic
from functions import *

data_mig, t_mig, x_mig, zosection, vmodel = process_seimics("/Users/yena/Desktop/Integrated Exercise/datasets/seismic24.segy", [2002, 4002, 6002])

seismic.imageseis(data_mig, x=x_mig, t=t_mig)
plt.title("Final Migrated Seismic Section")
plt.xlabel("Position (m)")
plt.ylabel("Time (s)")
plt.show()