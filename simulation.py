import importlib

import numpy as np
import matplotlib.pyplot as plt
import tqdm

import compute.mesh as mesh

importlib.reload(mesh)

def plotPsi(n: int, m: list, h: list, e: list, r: float, q: int = 100, p: int = 200) -> None:
    angle_mesh = mesh.MESH(n, m, h, e, r, q = q, p = p, theta_min=0.001, theta_max=np.pi, phi_min=0.001, phi_max=2*np.pi)
    psi_mesh_list = angle_mesh.psiMesh()
    for psi_mesh in psi_mesh_list:
        psi_mesh.plot()

def plotF(n: int, m: list, h: list, e: list, r: float, q: int = 100, p: int = 200) -> None:
    '''Plot the function f(theta, phi) = d_theta a_phi - d_phi a_theta'''
    angle_mesh = mesh.MESH(n, m, h, e, r, q = q, p = p, theta_min=0.001, theta_max=np.pi, phi_min=0.001, phi_max=2*np.pi)
    psi_mesh_list = angle_mesh.psiMesh()
    for psi_mesh in psi_mesh_list:
        d_theta_psi_mesh = psi_mesh.differentiate_mesh('theta')
        d_phi_psi_mesh = psi_mesh.differentiate_mesh('phi')
        a_theta_mesh = psi_mesh.a_mesh(d_theta_psi_mesh)
        a_phi_mesh = psi_mesh.a_mesh(d_phi_psi_mesh)
        d_theta_a_phi_mesh = a_phi_mesh.differentiate_mesh('theta')
        d_theta_a_phi_mesh.plot()
        d_phi_a_theta_mesh = a_theta_mesh.differentiate_mesh('phi')
        d_phi_a_theta_mesh.plot()
        f_mesh = d_theta_a_phi_mesh.f_mesh(d_phi_a_theta_mesh)
        f_mesh.plot()

def plotA(axis: str, m: float, d: float, q: int = 100, p: int = 200) -> None:
    angle_mesh = mesh.MESH(m, d, q = q, p = p, theta_min=0.001, theta_max=np.pi, phi_min=0.001, phi_max=2*np.pi)
    psi_mesh_list = angle_mesh.psiMesh()
    for psi_mesh in psi_mesh_list:
        d_theta_psi_mesh = psi_mesh.differentiate_mesh('theta')
        d_phi_psi_mesh = psi_mesh.differentiate_mesh('phi')
        if axis == 'theta':
            a_theta_mesh = psi_mesh.a_mesh(d_theta_psi_mesh)
            a_theta_mesh.plot()
        elif axis == 'phi':
            a_phi_mesh = psi_mesh.a_mesh(d_phi_psi_mesh)
            a_phi_mesh.plot()
        else:
            raise ValueError('Axis must be either theta or phi')

def computeC(n: int, m: list, h: list, e: list, r: float, q: int = 100, p: int = 200):
    '''Compute the Chern number for every sphere with a given dipole, field magnitude and coupling strength'''
    Cs = []
    angle_mesh = mesh.MESH(n, m, h, e, r, q = q, p = p, theta_min=0.001, theta_max=np.pi, phi_min=0.001, phi_max=2*np.pi)
    psi_mesh_list = angle_mesh.psiMesh()
    for psi_mesh in psi_mesh_list:
        d_theta_psi_mesh = psi_mesh.differentiate_mesh('theta')
        d_phi_psi_mesh = psi_mesh.differentiate_mesh('phi')
        a_theta_mesh = psi_mesh.a_mesh(d_theta_psi_mesh)
        a_phi_mesh = psi_mesh.a_mesh(d_phi_psi_mesh)
        d_theta_a_phi_mesh = a_phi_mesh.differentiate_mesh('theta')
        d_phi_a_theta_mesh = a_theta_mesh.differentiate_mesh('phi')
        f_mesh = d_theta_a_phi_mesh.f_mesh(d_phi_a_theta_mesh)
        Cs.append(f_mesh.getC())
    return Cs

def plotCvsM(m_sur_d_min: float, m_sur_d_max: float, q: int, p: int, n_points: int) -> None:
    '''Compute the Chern number for a range of ratio field magnitude, dipole and plots it'''
    n = 1
    h = [1.]
    ms = np.linspace(m_sur_d_min, m_sur_d_max, n_points)
    e = [1]
    r = 1.
    c = []
    for m in tqdm.tqdm(ms, desc='Computing Chern numbers'):
        c.append(computeC(n, [m], h, e, r, q, p)[0])
    plt.plot(ms, c)
    plt.xlabel('m/d')
    plt.ylabel('C')
    plt.title('Chern number as a function of m/d')
    plt.show()

def plotPhase(k: int, l: int, q: int = 100, p: int = 200):
    H = 1.

    n = 2
    m1 = 1./3.
    ms = np.linspace(0, H, k)
    h = [H for _ in range(n)]
    e = [1 for _ in range(n)]
    rs = np.linspace(0, 1.5 * H, l)
    c = []
    for m in ms:
        c.append([])
        for r in tqdm.tqdm(rs, desc=f'Computing Chern numbers for M2 = {m}'):
            v = computeC(n, [m1, m], h, e, r, q, p)
            sum = 0
            while len(v) != 0:
                sum *= 2
                sum += v.pop()
            c[-1].append(sum)
    R, M = np.meshgrid(rs, ms)
    c = np.array(c)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(R, M, c, cmap='viridis')

    ax.set_xlabel('r/H')
    ax.set_ylabel('M2/H')
    ax.set_zlabel('C')

    plt.show()