# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 18:30:47 2022

@author: Christer
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

datax = [1, 0.1, 0.3, 0.35, 1]
datay = [0, 0.2, 0.5, 0.6, 1]
d = pd.DataFrame(datax, datay)

sns.set_theme(style="ticks")

graph = sns.lineplot(data=d, markers=True)
graph.set(xlabel=r"$X_{1-\alpha}Y_{\alpha}$", ylabel=r"$\Delta H_f \, (eV/Atom)$")
graph.set(yticklabels=[])
graph.get_legend().remove()

plt.text(0.15, 0.04, r"$X_{0.8} Y_{0.2}$", fontsize=10)
plt.text(0.55, 0.28, r"$X_{0.4 \,} Y_{0.6}$", fontsize=10)
plt.xlim(0,1)
plt.ylim(0,1)
