Plan:

1. Convert 2D eqs to streamfunction form so we can remove pressure
2. Convert to vorticity via ω = ∇^2 * ψ
3. Find the Jacobian(not the one in the code; equation) in physical space then transform to Fourier (pseudo-spectral space)
4. calculate dw_hat/dt = -J_hat (from step 3) - nu * k * k * w_hat
5. use RK4 to integrate over time and find the vorticity field for each step in time
6. use poisson equation (step 2) to find streamfunc evolution over time -> velocity evolution over time
7. plot velocity over time; run this with initial conditions of a TG vortex and see if the results match expected ones
8. if they do the solver works; then implement into dapper and then run DA!