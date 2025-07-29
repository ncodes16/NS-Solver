# EQs in SF form: dω/dt + dψ/dy * d/dx (ω) - dψ/dx * d/dy(ψ) = nu * ω^2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
#defining parameters
N = 128
Lxy = 2*np.pi #periodic BCs

xx = np.linspace(0, Lxy, N, False)
yy = np.linspace(0, Lxy, N, False)

XX, YY = np.meshgrid(xx, yy)



#wavenumbers
kk = np.fft.fftfreq(N, Lxy/N)
kx = kk.reshape((N, 1))
ky = kk.reshape((1, N))
k2 = kx**2 + ky**2
k2[0, 0] = 1  # avoid division by zero
#time stuff
T = 1
dt = 0.0001
nu = 10 #kinematic viscosity

psi = np.sin(XX)*np.sin(YY) #TG vortex IC
psi_hat = np.fft.fft2(psi)
omega = 2 * np.sin(XX) * np.sin(YY)
omega_hat = np.fft.fft2(omega)
def dealias(f_hat):
    N = f_hat.shape[0]
    k_cutoff = N // 3

    bool_mask = np.ones_like(f_hat, dtype=bool)
    bool_mask[k_cutoff:-k_cutoff, :] = False
    bool_mask[:, k_cutoff:-k_cutoff] = False

    dealiased = np.copy(f_hat)
    dealiased[~bool_mask] = 0
    return dealiased
#J = dpsi/dy * domega/dx - dpsi/dx * domega/dy
def det_jacobian(psi_hat):
    dpsi_dy = np.fft.ifft2(1j * ky * psi_hat).real
    domega_dx = np.fft.ifft2(1j * kx * k2 * psi_hat).real
    dpsi_dx = np.fft.ifft2(1j * kx * psi_hat).real
    domega_dy = np.fft.ifft2(1j * ky * k2 * psi_hat).real

    # print(f"dpsi/dy: {dpsi_dy}\ndomega/dx: {domega_dx}\ndpsi/dx: {dpsi_dx}\ndomega/dy: {domega_dy}\n")
    return dpsi_dy * domega_dx - dpsi_dx * domega_dy




# Store psi snapshots for animation
#TODO double check range kutta
psi_snapshots = []
snapshot_steps = []
num_steps = int(T / dt)

#ETD-RK4 method
def nonlinear(psi_hat):
    J = det_jacobian(psi_hat)
    J_hat = dealias(np.fft.fft2(J))
    return -J_hat

L = np.diag(-nu * k2)
E = np.exp(dt * L)
E2 = np.exp(dt * L/2)

def phi1(z):
    phi = np.empty_like(z, dtype=np.complex128)
    small = np.abs(z) < 1e-6
    phi[small] = 1 + z[small] / 2 + z[small]**2 / 6
    phi[~small] = (np.exp(z[~small]) - 1)/z[~small]
    return phi
phi_E = phi1(L * dt)
phi_E2 = phi1(L * dt / 2)
for step in tqdm(range(num_steps)):
    a = E2 * omega_hat + dt * phi_E2 * nonlinear(psi_hat)
    Na = nonlinear(a)
    b = E2 * omega_hat + dt * phi_E2 * Na
    Nb = nonlinear(b)
    c = E * omega_hat + dt * phi_E * (2* Nb - nonlinear(psi_hat))
    Nc = nonlinear(c)
    omega_hat = E * omega_hat + dt * (phi_E * nonlinear(psi_hat) + 2*phi_E*(Na + Nb) + phi_E * Nc)/6
    omega = np.fft.ifft2(omega_hat).real / (Lxy * Lxy)  # Rescale to match the original domain size
    psi_hat = -omega_hat / k2
    psi_hat[0, 0] = 0  # Enforce zero mean for
    psi = np.fft.ifft2(psi_hat).real / (Lxy * Lxy)  # Rescale to match the original domain size
    if step % 100 == 0:
        psi_snapshots.append(psi.copy())
        snapshot_steps.append(step)
   

# Animation using matplotlib
import matplotlib.animation as animation
if len(psi_snapshots) == 0:
    print("No psi snapshots were saved. Animation will not run.")
else:
    psi_snapshots = np.array(psi_snapshots, dtype=np.float64)
    print(psi_snapshots)
    shapes = [np.shape(s) for s in psi_snapshots]
    print(f"First 5 snapshot shapes: {shapes[:10]}")
    if not all(s == shapes[0] and len(s) == 2 for s in shapes):
        raise ValueError(f"Not all psi_snapshots have the same 2D shape: {shapes}")
    try:
        psi_snapshots_arr = np.stack(psi_snapshots, axis=0)
        print(f"Shape of psi_snapshots_arr: {psi_snapshots_arr.shape}")
    except Exception as e:
        print(f"Error stacking psi_snapshots: {e}")
        print(f"Shapes: {shapes}")
        raise
    if psi_snapshots_arr.shape[0] == 0:
        print("No valid psi snapshots to animate.")
    else:
        fig, ax = plt.subplots()
        im = ax.imshow(psi_snapshots_arr[0], origin='lower', cmap='viridis', extent=(0, Lxy, 0, Lxy))
        ax.set_title('Streamfunction ψ, step 0')
        plt.colorbar(im, ax=ax)
        def update(frame):
            im.set_data(psi_snapshots_arr[frame])
            ax.set_title(f'Streamfunction ψ, step {snapshot_steps[frame]}')
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(psi_snapshots_arr), interval=10, blit=False)
        plt.show()
