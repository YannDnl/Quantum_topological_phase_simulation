import sympy as sp

theta = sp.symbols('theta', real=True)
phi = sp.symbols('phi', real=True)

i = sp.I

m = 0
d = 1

hamiltonian = sp.Matrix([[m - d * sp.cos(theta), - d * sp.sin(theta) * sp.exp(- i * phi)], [- d * sp.sin(theta) * sp.exp(i * phi), m + d * sp.cos(theta)]])

psis = hamiltonian.eigenvects()

energy = psis[0][0]
psi = psis[0][2][0]
for val, mult, state in psis:
    if val > energy:
        energy = val
        psi = state[0]

A_phi = - i * (psi.H).dot(psi.diff(phi))
A_theta = - i * (psi.H).dot(psi.diff(theta))

print('phi', A_phi.subs({theta: 1, phi: 3}))
print('theta', A_theta.subs({theta: 1, phi: 3}))

F_theta_phi = A_phi.diff(theta) - A_theta.diff(phi)

c = sp.integrate(F_theta_phi, (phi, 0, 2 * sp.pi), (theta, 0, sp.pi))/2 * sp.pi

print(c)