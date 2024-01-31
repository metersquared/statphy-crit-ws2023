from hw06_ising import Ising
import argparse
import pandas as pd
import numpy as np


# Use code from hw 6 and make a parser based on it, just need to calculate T such that aforementioned bounds fit.
parser = argparse.ArgumentParser(
    description="Process Monte Carlo simulation of Ising model with importance sampling. For Finite-Size scaling analysis"
)

parser.add_argument("length", metavar="L", type=int, nargs=1, help="Length of system")

args = parser.parse_args()

# Take the argument length of parser to system length.
L = args.length[0]

# Take the argument temperatures of parser to system temperature.
tc = 2 / np.log(np.sqrt(2) + 1)
Ts = np.linspace(-5 / L + tc, 5 / L + tc, num=10)
betas = 1 / Ts
x = (Ts - tc) * L

# Initialize variable to store magnetization squared.
m = np.zeros(len(betas))
msquared = np.zeros(len(betas))
msquared_err = np.zeros(len(betas))

# Initialize an Ising class with system length L.
ising = Ising(L)

for idx, beta in enumerate(betas):
    # To keep track of progress, print the current temperature at which value is calculated.
    print(f"kT:{Ts[idx]}")

    # Initialize variable to store 20 runs of the magnetization and matnetization squared
    ms = np.zeros(20)
    msquareds = np.zeros(20)

    for i in np.arange(20):
        # This is a single run consisting of:
        # Equilibration - Initialize Ising model in cold start, and run 5L sweeps.
        ising.equilibration(beta)
        # Measurement - Measure mean of the next 30L sweeps.
        ms[i], msquareds[i] = ising.measurement(beta)

    # Mean of all the 20 runs along with the error.
    m[idx] = np.mean(ms)
    msquared[idx] = np.mean(msquareds)
    msquared_err[idx] = (
        1 / (19) * np.sqrt(np.abs(np.mean(msquareds) - np.square(np.mean(ms))))
    )

d = {"T": Ts, "x": x, "m": m, "msquared": msquared, "err": msquared_err}
df = pd.DataFrame(data=d)
filename = "Monte_Carlo_Ising_Importance__FS_Scaling_L" + str(L) + ".csv"
df.to_csv(filename, index=False)
