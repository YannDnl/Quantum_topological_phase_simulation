import importlib
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import tqdm

from parameters import *
import compute.mesh as mesh

importlib.reload(mesh)



def plotPsi(n: int, m: list, h: list, e: list, r: np.ndarray, q: int, p: int) -> None:
    angle_mesh = mesh.MESH(n, m, h, e, r, q = q, p = p, theta_min=THETA_MIN, theta_max=THETA_MAX, phi_min=PHI_MIN, phi_max=PHI_MAX)
    psi_mesh = angle_mesh.psiMesh()
    psi_mesh.plot()

def plotA(axis: str, n:int, m: list, h: list, e: list, r: np.ndarray, q: int, p: int) -> None:
    angle_mesh = mesh.MESH(n, m, h, e, r, q = q, p = p, theta_min=THETA_MIN, theta_max=THETA_MAX, phi_min=PHI_MIN, phi_max=PHI_MAX)
    psi_mesh = angle_mesh.psiMesh()
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

def plotF(n: int, m: list, h: list, e: list, r: np.ndarray, q: int, p: int) -> None:
    '''Plot the function f(theta, phi) = d_theta a_phi - d_phi a_theta'''
    angle_mesh = mesh.MESH(n, m, h, e, r, q = q, p = p, theta_min=THETA_MIN, theta_max=THETA_MAX, phi_min=PHI_MIN, phi_max=PHI_MAX)
    psi_mesh = angle_mesh.psiMesh()
    d_theta_psi_mesh = psi_mesh.differentiate_mesh('theta')
    d_phi_psi_mesh = psi_mesh.differentiate_mesh('phi')
    a_theta_mesh = psi_mesh.a_mesh(d_theta_psi_mesh)
    a_phi_mesh = psi_mesh.a_mesh(d_phi_psi_mesh)
    d_theta_a_phi_mesh = a_phi_mesh.differentiate_mesh('theta')
    d_phi_a_theta_mesh = a_theta_mesh.differentiate_mesh('phi')
    f_mesh = d_theta_a_phi_mesh.f_mesh(d_phi_a_theta_mesh)
    f_mesh.plot()

def plotSteps(n: int, m: list, h: list, e: list, r: np.ndarray, q: int, p: int) -> None:
    '''Plot the function f(theta, phi) = d_theta a_phi - d_phi a_theta'''
    angle_mesh = mesh.MESH(n, m, h, e, r, q = q, p = p, theta_min=THETA_MIN, theta_max=THETA_MAX, phi_min=PHI_MIN, phi_max=PHI_MAX)
    psi_mesh = angle_mesh.psiMesh()
    print('psi')
    psi_mesh.plot()
    d_theta_psi_mesh = psi_mesh.differentiate_mesh('theta')
    d_phi_psi_mesh = psi_mesh.differentiate_mesh('phi')
    a_theta_mesh = psi_mesh.a_mesh(d_theta_psi_mesh)
    a_phi_mesh = psi_mesh.a_mesh(d_phi_psi_mesh)
    print('A theta')
    a_theta_mesh.plot()
    print('A phi')
    a_phi_mesh.plot()
    d_theta_a_phi_mesh = a_phi_mesh.differentiate_mesh('theta')
    d_phi_a_theta_mesh = a_theta_mesh.differentiate_mesh('phi')
    f_mesh = d_theta_a_phi_mesh.f_mesh(d_phi_a_theta_mesh)
    print('f')
    f_mesh.plot()

def computeC(n: int, m: list, h: list, e: list, r: np.ndarray, q: int, p: int):
    '''Compute the Chern number for every sphere with a given dipole, field magnitude and coupling strength'''
    angle_mesh = mesh.MESH(n, m, h, e, r, q = q, p = p, theta_min=THETA_MIN, theta_max=THETA_MAX, phi_min=PHI_MIN, phi_max=PHI_MAX)
    psi_mesh = angle_mesh.psiMesh()
    d_theta_psi_mesh = psi_mesh.differentiate_mesh('theta')
    d_phi_psi_mesh = psi_mesh.differentiate_mesh('phi')
    a_theta_mesh = psi_mesh.a_mesh(d_theta_psi_mesh)
    a_phi_mesh = psi_mesh.a_mesh(d_phi_psi_mesh)
    d_theta_a_phi_mesh = a_phi_mesh.differentiate_mesh('theta')
    d_phi_a_theta_mesh = a_theta_mesh.differentiate_mesh('phi')
    f_mesh = d_theta_a_phi_mesh.f_mesh(d_phi_a_theta_mesh)
    return f_mesh.getC()

def computeCLineForParallel(args):
    n, m, h, e, r, q, p, k = args
    return k, computeC(n, m, h, e, r, q, p)

def computeCSquareForParallel(args):
    n, m, h, e, r, q, p, w, v = args
    return w, v, computeC(n, m, h, e, r, q, p)

def plotCvsMParallel(m_sur_d_min: float, m_sur_d_max: float, q: int, p: int, n_points: int) -> None:
    '''Compute the Chern number for a range of ratio field magnitude, dipole and plots it, parallelized
    faster than serial, 3 times'''
    n = 1
    h = [H]
    ms = np.linspace(m_sur_d_min, m_sur_d_max, n_points)
    e = [E]
    r = np.array([[RXX, RXY, RXZ],
                  [RYX, RYY, RYZ],
                  [RZX, RZY, RZZ]])
    c = [None for _ in range(n_points)]

    input = [(n, [m], h, e, r, q, p, k) for k, m in enumerate(ms)]
    with multiprocessing.Pool(N_PROCESSES) as pool:
        for result in pool.imap_unordered(computeCLineForParallel, input):
            k, c_ = result
            c[k] = c_

    plt.scatter(ms, c, marker='+')
    plt.xlabel('m/d')
    plt.ylabel('C')
    plt.title('Chern number as a function of m/d')
    plt.show()

def plotPhaseParallel(k: int, l: int, q: int, p: int):
    '''Faster than serial, 5 times'''
    n = N
    m1 = M1
    ms = np.linspace(0, H, k)
    h = [H for _ in range(n)]
    e = [E for _ in range(n)]
    rzs = np.linspace(0, 1.5 * H, l)
    rs = [np.array([[RXX, RXY, RXZ],
                    [RYX, RYY, RYZ],
                    [RZX, RZY, rz]]) for rz in rzs]
    c = [[None for _ in range(l)] for _ in range(k)]

    input = []
    for w, m in enumerate(ms):
        i = [(n, [m1, m], h, e, r, q, p, w, v) for v, r in enumerate(rs)]
        input.extend(i)

    with multiprocessing.Pool(N_PROCESSES) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(computeCSquareForParallel, input), total = k * l):
            w, v, c_ = result
            c[w][v] = c_

    R, M = np.meshgrid(rzs, ms)
    c = np.array(c)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(R, M, c, cmap='viridis')

    ax.set_xlabel('r/H')
    ax.set_ylabel('M2/H')
    ax.set_zlabel('C')

    plt.show()

def plotSingleLineParallel(l: int, m2, q: int, p: int):
    n = N
    m1 = M1
    h = [H for _ in range(n)]
    e = [E for _ in range(n)]
    rzs = np.linspace(0, 2 * H, l)
    rs = [np.array([[RXX, RXY, RXZ],
                    [RYX, RYY, RYZ],
                    [RZX, RZY, rz]]) for rz in rzs]
    c = [None for _ in range(l)]

    input = [(n, [m1, m2], h, e, r, q, p, k) for k, r in enumerate(rs)]

    with multiprocessing.Pool(N_PROCESSES) as pool:
        for result in pool.imap_unordered(computeCLineForParallel, input):
            k, c_ = result
            c[k] = c_
    
    plt.plot(rzs, c)
    plt.xlabel('r/H')
    plt.ylabel('C')
    plt.title('Chern number as a function of r/h')
    plt.show()