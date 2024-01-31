import numpy as np


# Created the Ising class to store Ising models with different setup.
class XY:
    """Class for storing XY model"""

    def __init__(self, L: int, J: float = 1.0, h: float = 1.0, seed: int = 12345):
        super(XY, self).__init__()
        self.L = L
        self.J = J
        self.h = h
        self.rng = np.random.default_rng(seed)

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
            self.spins = np.pi * np.ones((self.L, self.L))
        elif start == "hot":
            self.spins = 2 * np.pi * self.rng.random((self.L, self.L)) - np.pi

    def shift(self, i, j, phi):
        """Shift a spin at site (i,j).

        Parameters
        ----------
        i : int
            row position
        j : int
            column position
        phi : float
            new spin angle
        """
        self.spins[i, j] = phi

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
        Jenergy = 0
        henergy = 0
        for i, j in np.ndindex(spins.shape):
            henergy = henergy + np.cos(spins[i, j])
            Jenergy = Jenergy + np.cos(spins[i, j] - spins[i - 1, j])
            Jenergy = Jenergy + np.cos(spins[i, j] - spins[i, j - 1])
            if i + 1 == self.L:
                Jenergy = Jenergy + np.cos(spins[i, j] - spins[0, j])
            else:
                Jenergy = Jenergy + np.cos(spins[i, j] - spins[i + 1, j])
            if j + 1 == self.L:
                Jenergy = Jenergy + np.cos(spins[i, j] - spins[i, 0])
            else:
                Jenergy = Jenergy + np.cos(spins[i, j] - spins[i, j + 1])
        return -self.J * Jenergy / 2 - self.h * henergy

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
            m = m + np.cos(spins[i, j])
        return m

    def delta_energy(self, i, j, phi):
        """Calculate change of energy when shifting site (i,j), with another angle.

        Parameters
        ----------
        i : int
            row position
        j : int
            column position
        phi : float
            new angle

        Returns
        -------
        float
            Change in energy of system.
        """
        spins = self.spins
        Jenergy = 0
        Jenergy = (
            Jenergy
            + np.cos(phi - spins[i - 1, j])
            - np.cos(spins[i, j] - spins[i - 1, j])
        )
        Jenergy = (
            Jenergy
            + np.cos(phi - spins[i, j - 1])
            - np.cos(spins[i, j] - spins[i, j - 1])
        )
        if i + 1 == self.L:
            Jenergy = (
                Jenergy + np.cos(phi - spins[0, j]) - np.cos(spins[i, j] - spins[0, j])
            )
        else:
            Jenergy = (
                Jenergy
                + np.cos(phi - spins[i + 1, j])
                - np.cos(spins[i, j] - spins[i + 1, j])
            )
        if j + 1 == self.L:
            Jenergy = (
                Jenergy + np.cos(phi - spins[i, 0]) - np.cos(spins[i, j] - spins[i, 0])
            )
        else:
            Jenergy = (
                Jenergy
                + np.cos(phi - spins[i, j + 1])
                - np.cos(spins[i, j] - spins[i, j + 1])
            )
        return -self.J * Jenergy - self.h * (np.cos(phi) - np.cos(spins[i, j]))

    def delta_magnetization(self, i, j, phi):
        """Calculate change of magnetization when shifting site (i,j), with another angle.

        Parameters
        ----------
        i : int
            row position
        j : int
            column position
        phi : float
            new angle

        Returns
        -------
        int
            Change in magnetization of system.
        """
        return np.cos(phi) - np.cos(self.spins[i, j])

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
        phi = 2 * np.pi * self.rng.random() - np.pi
        deltaE = self.delta_energy(i, j, phi)
        deltaM = self.delta_magnetization(i, j, phi)
        r = self.rng.random()

        if r < 1 / (np.exp(beta * deltaE) + 1):
            self.shift(i, j, phi)
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
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # I created a parser so that we can input variable arguments in the python script, i.e. varying L or temperature bounds.
    parser = argparse.ArgumentParser(
        description="Plot Monte Carlo simulation result of XY model with importance sampling."
    )

    parser.add_argument(
        "length", metavar="L", type=int, nargs=1, help="Length of system"
    )

    parser.add_argument(
        "beta",
        metavar="B",
        type=float,
        nargs=1,
        help="Temperature of simulation",
    )

    parser.add_argument(
        "int_energy",
        metavar="J",
        type=float,
        nargs=1,
        help="Interaction energy of system",
    )

    parser.add_argument(
        "h_energy",
        metavar="H",
        type=float,
        nargs=1,
        help="External field of system",
    )

    args = parser.parse_args()

    L = args.length[0]
    beta = args.beta[0]
    J = args.int_energy[0]
    h = args.h_energy[0]

    xy = XY(L, J, h)

    # Intialization
    xy.initialize("cold")
    xy.m = xy.get_magnetization()
    xy.energy = xy.get_energy()

    # 100 sweeps
    for i in range(100):
        xy.sweep(beta)

    fig = plt.figure()

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "my_colormap", ["black", "white"], 256
    )
    bounds = np.linspace(-np.pi, np.pi, 20)
    norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)

    img = plt.imshow(
        xy.spins, interpolation="nearest", cmap=cmap, origin="lower", norm=norm
    )

    plt.colorbar(img, cmap=cmap, boundaries=bounds, ticks=[-np.pi, 0, np.pi])

    fig.suptitle(f"XY Plot, $L=${L} $T=${1/beta} $J=${J} $h=${h}")
    fig.savefig(f"XY_Plot_L{L}_beta{beta}_J{J}_h{h}.pdf")
