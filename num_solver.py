# EQs in SF form: dω/dt + dψ/dy * d/dx (ω) - dψ/dx * d/dy(ψ) = nu * ω^2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Solver:
    def __init__(self, N: int | float, Lxy: float, dt: float, nu: float, T: float, psi0: np.ndarray, omega0: np.ndarray):
        self.N = N
        self.Lxy = Lxy
        self.dt = dt
        self.nu = nu #kinematic viscosity
        self.T = T
        self.num_steps = int(T / dt)  # Number of time steps

        # Initialize grid and wavenumbers
        self.xx = np.linspace(0, Lxy, N, False)
        self.yy = np.linspace(0, Lxy, N, False)
        self.XX, self.YY = np.meshgrid(self.xx, self.yy)

        # Wavenumbers
        kk = np.fft.fftfreq(N, Lxy/N) * 2 * np.pi
        self.kx = kk.reshape((N, 1))
        self.ky = kk.reshape((1, N))
        self.k2 = self.kx**2 + self.ky**2
        self.k2[0, 0] = 1  # avoid division by zero

        #initial conditions
        self.psi = psi0
        self.psi_hat = np.fft.fft2(self.psi)
        self.omega = omega0
        self.omega_hat = np.fft.fft2(self.omega)


# psi = np.sin(XX)*np.sin(YY) #TG vortex IC
# psi_hat = np.fft.fft2(psi)
# omega = 2 * np.sin(XX) * np.sin(YY)
# omega_hat = np.fft.fft2(omega)
    def dealias(self, f_hat):
        N = f_hat.shape[0]
        k_cutoff = N // 3

        bool_mask = np.ones_like(f_hat, dtype=bool)
        bool_mask[k_cutoff:-k_cutoff, :] = False
        bool_mask[:, k_cutoff:-k_cutoff] = False

        dealiased = np.copy(f_hat)
        dealiased[~bool_mask] = 0
        return dealiased
#J = dpsi/dy * domega/dx - dpsi/dx * domega/dy
    def det_jacobian(self, psi_hat):
        scale = 1 / (self.Lxy * self.Lxy)  # Rescale to match the original domain size
        dpsi_dy = np.fft.ifft2(1j * self.ky * psi_hat).real * scale
        domega_dx = np.fft.ifft2(1j * self.kx * self.k2 * psi_hat).real * scale
        dpsi_dx = np.fft.ifft2(1j * self.kx * psi_hat).real * scale
        domega_dy = np.fft.ifft2(1j * self.ky * self.k2 * psi_hat).real * scale

        # print(f"dpsi/dy: {dpsi_dy}\ndomega/dx: {domega_dx}\ndpsi/dx: {dpsi_dx}\ndomega/dy: {domega_dy}\n")
        return dpsi_dy * domega_dx - dpsi_dx * domega_dy



#ETD-RK4 method
    def nonlinear(self, psi_hat):
        J = self.det_jacobian(psi_hat)
        J_hat = self.dealias(np.fft.fft2(J))
        return -J_hat
        
    def phi1(self, z):
        phi = np.empty_like(z, dtype=np.complex128)
        small = np.abs(z) < 1e-6
        phi[small] = 1 + z[small] / 2 + z[small]**2 / 6 + z[small]**3 / 24 + z[small]**4 / 120
        phi[~small] = (np.exp(z[~small]) - 1)/z[~small]
        return phi
    def run(self) -> np.ndarray:
        L = -self.nu * self.k2 #generalize
        E = np.exp(self.dt * L)
        E2 = np.exp(self.dt * L/2)
        phi_E = self.phi1(L * self.dt)
        phi_E2 = self.phi1(L * self.dt / 2)
        psis = np.zeros((self.num_steps, self.N, self.N), dtype=np.float64)
        for step in tqdm(range(self.num_steps)):
            a = E2 * self.omega_hat + self.dt * phi_E2 * self.nonlinear(self.psi_hat)
            Na = self.nonlinear(a)
            b = E2 * self.omega_hat + self.dt * phi_E2 * Na
            Nb = self.nonlinear(b)
            c = E * self.omega_hat + self.dt * phi_E * (2* Nb - self.nonlinear(self.psi_hat))
            Nc = self.nonlinear(c)
            self.omega_hat = E * self.omega_hat + self.dt * (phi_E * self.nonlinear(self.psi_hat) + 2*phi_E*(Na + Nb) + phi_E * Nc)/6
            self.omega = np.fft.ifft2(self.omega_hat).real #/ (self.Lxy * self.Lxy)  # Rescale to match the original domain size
            self.psi_hat = self.omega_hat / self.k2 #i^2 = -1; -1 / - 1 = 1.
            self.psi_hat[0, 0] = 0  # Enforce zero mean for
            self.psi = np.fft.ifft2(self.psi_hat).real #/ (self.Lxy * self.Lxy)
            if step < self.num_steps:
                psis[step ] = self.psi
        return psis
   

# Animation using matplotlib
    def animate_snapshots(self, psis, snapshot_steps):
        if len(psis) == 0:
            print("No psi snapshots were saved. Animation will not run.")
            return
        # Select evenly spaced indices
        indices = np.linspace(0, len(psis), snapshot_steps, False, dtype=int)
        psi_snapshots = psis[indices]
        # For title, get the actual time step for each snapshot
        step_numbers = indices
        # Check shapes
        shapes = [np.shape(s) for s in psi_snapshots]
        if not all(s == shapes[0] and len(s) == 2 for s in shapes):
            raise ValueError(f"Not all psi_snapshots have the same 2D shape: {shapes}")
        # Animate
        fig, ax = plt.subplots()
        im = ax.imshow(psi_snapshots[0], origin='lower', cmap='viridis', extent=(0, self.Lxy, 0, self.Lxy))
        ax.set_title(f'Streamfunction ψ, step {step_numbers[0]}')
        plt.colorbar(im, ax=ax)
        def update(frame):
            im.set_data(psi_snapshots[frame])
            ax.set_title(f'Streamfunction ψ, step {step_numbers[frame]}')
            return (im,)
        ani = animation.FuncAnimation(fig, update, frames=len(psi_snapshots), interval=100, blit=False)
        plt.show()

X, Y = np.meshgrid(np.linspace(0, 2 * np.pi, 128, False), np.linspace(0, 2 * np.pi, 128, False))
psi = np.sin(X) * np.sin(Y)  # TG vortex initial condition
omega = 2 * psi
TG_vortex = Solver(128, 2 * np.pi, 0.001, 10, 1, psi, omega)
# psis = TG_vortex.run()
# TG_vortex.animate_snapshots(psis, 10)