import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import numpy as np
from os.path import abspath, dirname, join

l10 = pd.read_csv("Monte_Carlo_Ising_Importance__FS_Scaling_L10.csv")
l20 = pd.read_csv("Monte_Carlo_Ising_Importance__FS_Scaling_L20.csv")
l30 = pd.read_csv("Monte_Carlo_Ising_Importance__FS_Scaling_L30.csv")

m10 = l10["m"].to_numpy()
m20 = l20["m"].to_numpy()
m30 = l30["m"].to_numpy()
x = l10["x"].to_numpy()

L = np.arange(10, 40, 10)

fig, ax = plt.subplots(sharey=True, tight_layout=True)
fig.suptitle("Finite size scaling of $m$ in heat-bath Ising model")

# Choose the data such that the curve is the closest to a linear line
idx = 2

m = np.array([m10[idx], m20[idx], m30[idx]])
coeff = np.polyfit(np.log(L), np.log(m), 1)
p = np.poly1d(coeff)


ax.scatter(np.log(L), np.log(m), marker="x", label=f"$L(T-T_c)=${x[idx]}")
ax.plot(np.log(L), p(np.log(L)), label="Fit:${}ln(L)+{}$".format(coeff[0], coeff[1]))

ax.set_xlabel("ln(L)")
ax.set_ylabel("ln(E(m))")

ax.legend()
path = join(dirname(abspath(__file__)), "hw07_plot.pdf")

fig.savefig(path)
