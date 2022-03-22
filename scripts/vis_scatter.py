"""
vis_scatter.py

I'm tired. Documentation later.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv(r"D:\data\naturalstories\rt-surp.tsv", sep='\t', header=0)

plt.plot(np.log(data['reading-time']), np.log(data['brn-surp']), '.')

plt.show()
