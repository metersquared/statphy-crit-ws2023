import numpy as np


# Created the Ising class to store Ising models with different setup.
class Ising:
    """Class for storing ising model"""

    def __init__(self, L: int, J: float = 1.0, seed: int = 12345):
        super(Ising, self).__init__()
        self.L = L
        self.J = J
        self.rng = np.random.default_rng(seed)

    def initialize(self, start="cold"):
        """Initialize LxL ising model with interface boundary.

        Parameters
        ----------
        L : int
            Length of system

        start : str, optional
            Initializations option, with hot start (random spins) or cold start (uniform spins), by default "cold"
        """
        # L+2 to add boundary to sites as well.
        if start == "cold":
            self.spins = np.ones((self.L + 2, self.L + 2))
        elif start == "hot":
            self.spins = np.power(
                -1, self.rng.integers(2, size=(self.L + 2, self.L + 2))
            )

        # Define the interface boundary condition
        self.spins[0, :] = -1
        self.spins[0 : int((self.L + 2) / 2), [0, -1]] = -1
        self.spins[int((self.L + 2) / 2) : -1, [0, -1]] = 1
        self.spins[-1, :] = 1

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
        # Consider edge cases where spins pairs with less than 4 spins
        for i, j in np.ndindex(spins.shape):
            if i > 0:
                energy = energy + spins[i, j] * spins[i - 1, j]
            if j > 0:
                energy = energy + spins[i, j] * spins[i, j - 1]
            if i < self.L + 1:
                energy = energy + spins[i, j] * spins[i + 1, j]
            if j < self.L + 1:
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
        # Consider edge cases where spins pairs with less than 4 spins
        if i > 0:
            energy = energy + spins[i, j] * spins[i - 1, j]
        if j > 0:
            energy = energy + spins[i, j] * spins[i, j - 1]
        if i < self.L + 1:
            energy = energy + spins[i, j] * spins[i + 1, j]
        if j < self.L + 1:
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
            if (
                0 < i < self.L + 1 and 0 < j < self.L + 1
            ):  # Ensures only inner sites are updated.
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
    import matplotlib.animation as animation

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
        nargs=1,
        help="Temperature of simulation",
    )

    parser.add_argument(
        "sweeps",
        metavar="s",
        type=int,
        nargs=1,
        help="Number of Monte Carlo sweeps",
    )

    parser.add_argument(
        "frames",
        metavar="f",
        type=int,
        nargs=1,
        help="Number of animation frames",
    )

    args = parser.parse_args()

    L = args.length[0]
    T = args.t[0]
    beta = 1 / T
    frames = args.frames[0]
    sweeps = args.sweeps[0]

    spf = sweeps / frames  # sweeps per frame

    ising = Ising(L)

    # Intialization and equilibration
    ising.initialize("hot")
    ising.equilibration(beta)

    snapshot = []

    for i in range(frames):
        for j in range(int(spf)):
            ising.sweep(beta)
        # For every 10 sweeps, perform a snapshot of the spin setup
        snapshot.append(np.copy(ising.spins))

    # Create plot with black and white color for +1 and -1
    fig, ax = plt.subplots()

    cmap = mpl.colors.ListedColormap(["black", "white"])
    bounds = [-1, 0, 1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    img = ax.imshow(snapshot[0], interpolation="nearest", cmap=cmap, norm=norm)
    plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-1, 0, 1])

    def update(frame):
        # Update the grid plot:
        data = snapshot[frame]
        img.set_data(data)
        # Update current number of sweeps
        ax.set_title(f"$T=${T}, {int((frame+1)*spf)} sweeps")
        return img

    ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=100)
    ani.save(
        filename=f"Ising_Evolution_L{L}_T{T}_sweeps{sweeps}_spf{spf}.gif",
        writer="pillow",
    )
