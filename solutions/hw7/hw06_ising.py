import numpy as np


# Created the Ising class to store Ising models with different setup.
class Ising:
    """Class for storing ising model"""

    def __init__(self, L: int, J: float = 1.0, seed: int = 12345, bc="periodic"):
        super(Ising, self).__init__()
        self.L = L
        self.J = J
        self.rng = np.random.default_rng(seed)
        self.bc = "periodic"

    def initialize(self, start="cold"):
        """Initialize LxL ising model.

        Parameters
        ----------
        L : int
            Length of system

        start : str, optional
            Initializations option, with hot start (random spins) or cold start (uniform spins), by default "cold"
        """

        if start == "cold":
            self.spins = np.ones((self.L, self.L))
        elif start == "hot":
            self.spins = np.power(-1, self.rng.integers(2, size=(self.L, self.L)))

    def flip(self, i, j):
        """Flip a spin at site (i,j).

        Parameters
        ----------
        i : int
            row position
        j : int
            column position
        """
        self.spins[i, j] = -self.spins[i, j]

    def get_energy(self):
        """Calculate energy of system (explicitly).

        Notes
        -----
        This is a costly process for large L.

        Returns
        -------
        float
            Energy of system.
        """
        spins = self.spins
        energy = 0
        for i, j in np.ndindex(spins.shape):
            energy = energy + spins[i, j] * spins[i - 1, j]
            energy = energy + spins[i, j] * spins[i, j - 1]
            if i + 1 == self.L:
                energy = energy + spins[i, j] * spins[0, j]
            else:
                energy = energy + spins[i, j] * spins[i + 1, j]
            if j + 1 == self.L:
                energy = energy + spins[i, j] * spins[i, 0]
            else:
                energy = energy + spins[i, j] * spins[i, j + 1]
        return -self.J * energy / 2

    def get_magnetization(self):
        """Calculate magnetization of system (explicitly).

        Notes
        -----
        This is a costly process for large L.

        Returns
        -------
        float
            Magnetization of system.
        """
        spins = self.spins
        m = 0
        for i, j in np.ndindex(spins.shape):
            m = m + spins[i, j]
        return m

    def delta_energy(self, i, j):
        """Calculate change of energy when flipping site (i,j).

        Parameters
        ----------
        i : int
            row position
        j : int
            column position

        Returns
        -------
        int
            Change in energy of system.
        """
        spins = self.spins
        energy = 0
        energy = energy + spins[i, j] * spins[i - 1, j]
        energy = energy + spins[i, j] * spins[i, j - 1]
        if i + 1 == self.L:
            energy = energy + spins[i, j] * spins[0, j]
        else:
            energy = energy + spins[i, j] * spins[i + 1, j]
        if j + 1 == self.L:
            energy = energy + spins[i, j] * spins[i, 0]
        else:
            energy = energy + spins[i, j] * spins[i, j + 1]
        return 2 * self.J * energy

    def delta_magnetization(self, i, j):
        """Calculate change of magnetization when flipping site (i,j).

        Parameters
        ----------
        i : int
            row position
        j : int
            column position

        Returns
        -------
        int
            Change in magnetization of system.
        """
        return -2 * self.spins[i, j]

    def update(self, beta, i, j):
        """Update the spin of the system under heat bath probability.

        Parameters
        ----------
        beta : float
            Temperature
        i : int
            row position
        j : int
            column position
        """
        deltaE = self.delta_energy(i, j)
        deltaM = self.delta_magnetization(i, j)
        r = self.rng.random()

        if r < 1 / (np.exp(beta * deltaE) + 1):
            self.flip(i, j)
            self.energy = self.energy + deltaE
            self.m = self.m + deltaM

    def sweep(self, beta):
        """Perform a Monte Carlo sweep in the system.

        Parameters
        ----------
        beta : float
            Temperature
        """
        for i, j in np.ndindex(self.spins.shape):
            self.update(beta, i, j)

    def equilibration(self, beta):
        """Perform an equilibration process in the system.

        Parameters
        ----------
        beta : float
            Temperature
        """
        self.initialize()
        self.m = self.get_magnetization()
        self.energy = self.get_energy()
        for i in np.arange(5 * self.L):
            self.sweep(beta)

    def measurement(self, beta):
        """Perform a measurement process in the system.

        Parameters
        ----------
        beta : float
            Temperature

        Returns
        -------
        float, float
            magnetization and magnetization squared of the system.
        """
        ms = np.zeros(30 * self.L)
        for i in np.arange(30 * self.L):
            self.sweep(beta)
            ms[i] = self.m / (self.L * self.L)
        msquared = np.square(ms)

        return np.mean(ms), np.mean(msquared)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    # I created a parser so that we can input variable arguments in the python script, i.e. varying L or temperature bounds.
    parser = argparse.ArgumentParser(
        description="Process Monte Carlo simulation of Ising model with importance sampling."
    )

    parser.add_argument(
        "length", metavar="L", type=int, nargs=1, help="Length of system"
    )

    parser.add_argument(
        "t",
        metavar="T",
        type=float,
        nargs=2,
        help="Temperature bounds in [min max] format",
    )

    args = parser.parse_args()

    # Take the argument length of parser to system length.
    L = args.length[0]

    # Take the argument temperatures of parser to system temperature.
    Ts = np.linspace(args.t[0], args.t[1], num=10)
    betas = 1 / Ts

    # Initialize variable to store magnetization squared.
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
        msquared[idx] = np.mean(msquareds)
        msquared_err[idx] = (
            1 / (19) * np.sqrt(np.abs(np.mean(msquareds) - np.square(np.mean(ms))))
        )

    d = {"T": Ts, "msquared": msquared, "err": msquared_err}
    df = pd.DataFrame(data=d)
    filename = "Monte_Carlo_Ising_Importance_L" + str(L) + ".csv"
    df.to_csv(filename, index=False)
