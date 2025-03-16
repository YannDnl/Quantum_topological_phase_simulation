from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt

def hamiltonian(theta, phi, m, d):
    return np.array([[m - d * np.cos(theta), - d * np.sin(theta) * np.exp(- 1j * phi)], [- d * np.sin(theta) * np.exp(1j * phi), m + d * np.cos(theta)]])

def psi(hamiltonian):
    energies, states = np.linalg.eig(hamiltonian)
    i = np.argmax(energies)
    psi = states[i]
    return psi

def compute_c(m, d, h_theta, h_phi):
    n_theta = int(1/h_theta)
    n_phi = int(1/h_phi)
    mesh = [[(np.pi * h_theta * k + h_theta/10, 2 * np.pi * h_phi * i) for k in range(n_theta)] for i in range(n_phi)]### h/10 is a hack to avoid pi/2, which is handled badly by np.linalg.eig
    psis = []
    #dists = []
    for line in mesh:
        psis.append([])
        #dists.append([])
        for theta, phi in line:
            p = psi(hamiltonian(theta, phi, m, d))
            if len(psis[-1]) != 0:
                old_p = psis[-1][-1]
                d_plus = np.linalg.norm(p - old_p)
                d_minus = np.linalg.norm(-p - old_p)
                #dists[-1].append((d_plus, d_minus))
                if d_minus < d_plus:
                    p = -1 * p
            psis[-1].append(p)
    dpsi_theta = []
    dpsi_phi = []
    for i in range(n_phi):
        dpsi_theta.append([])
        dpsi_phi.append([])
        for k in range(n_theta):
            if k == 0:
                dpsi_theta[-1].append((psis[i][k + 1] - psis[i][k])/h_theta)
            elif k == n_theta - 1:
                dpsi_theta[-1].append((psis[i][k] - psis[i][k - 1])/h_theta)
            else:
                dpsi_theta[-1].append((psis[i][k + 1] - psis[i][k - 1])/(2 * h_theta))
            if i == 0:
                dpsi_phi[-1].append((psis[i + 1][k] - psis[i][k])/h_phi)
            elif i == n_phi - 1:
                dpsi_phi[-1].append((psis[i][k] - psis[i - 1][k])/h_phi)
            else:
                dpsi_phi[-1].append((psis[i + 1][k] - psis[i - 1][k])/(2 * h_phi))
    A_theta = []
    A_phi = []
    for i in range(n_phi):
        A_theta.append([])
        A_phi.append([])
        for k in range(n_theta):
            a_theta = np.dot(psis[i][k].conj().T, dpsi_theta[i][k])
            a_phi = np.dot(psis[i][k].conj().T, dpsi_phi[i][k])
            A_theta[-1].append(-1j * a_theta)
            A_phi[-1].append(-1j * a_phi)
    f = []
    for i in range(n_phi):
        f.append([])
        for k in range(n_theta):
            if k == 0:
                d_theta_a_phi = (A_phi[i][k + 1] - A_phi[i][k])/h_theta
            elif k == n_theta - 1:
                d_theta_a_phi = (A_phi[i][k] - A_phi[i][k - 1])/h_theta
            else:
                d_theta_a_phi = (A_phi[i][k + 1] - A_phi[i][k - 1])/(2 * h_theta)
            if i == 0:
                d_phi_a_theta = (A_theta[i + 1][k] - A_theta[i][k])/h_phi
            elif i == n_phi - 1:
                d_phi_a_theta = (A_theta[i][k] - A_theta[i - 1][k])/h_phi
            else:
                d_phi_a_theta = (A_theta[i + 1][k] - A_theta[i - 1][k])/(2 * h_phi)
            f[-1].append(d_theta_a_phi - d_phi_a_theta)
    c = 0
    for l in f:
        c += sum(l)
    c = c * h_phi * h_theta/(2 * np.pi)
    fig, (psi_plot, dpsi_plot) = plt.subplots(2, 1)
    psi_plot.plot(mesh[0], [k[0] for k in psis[0]])
    psi_plot.plot(mesh[0], [k[1] for k in psis[0]])
    psi_plot.set_title('Psi')
    dpsi_plot.plot(mesh[0], [k[0] for k in dpsi_theta[0]])
    dpsi_plot.plot(mesh[0], [k[1] for k in dpsi_theta[0]])
    dpsi_plot.set_title('dPsi/dTheta')
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    psix_plot = fig.add_subplot(1, 2, 1, projection='3d')
    c = np.array([[np.angle(e[0]) + np.pi for e in l] for l in psis])
    surf = psix_plot.plot_surface(np.array([[e[1] for e in l] for l in mesh]), np.array([[e[0] for e in l] for l in mesh]), np.array([[np.abs(e[0]) for e in l] for l in psis]).T, facecolors=plt.cm.viridis(c /(2 * np.pi)), rstride=1, cstride=1, linewidth=0, antialiased=False)
    m = plt.cm.ScalarMappable(cmap='viridis')
    m.set_array(c)
    fig.colorbar(m, ax=psix_plot, shrink=0.5, aspect=10)
    psix_plot.set_ylabel('Theta')
    psix_plot.set_xlabel('Phi')
    psix_plot.set_title('Psi x')
    plt.show()
    fig = plt.figure(figsize=(10, 7))
    psiy_plot = fig.add_subplot(1, 2, 1, projection='3d')
    c = np.array([[np.angle(e[1]) + np.pi for e in l] for l in psis])
    surf = psiy_plot.plot_surface(np.array([[e[0] for e in l] for l in mesh]), np.array([[e[1] for e in l] for l in mesh]), np.array([[np.abs(e[1]) for e in l] for l in psis]), facecolors=plt.cm.viridis(c /(2 * np.pi)), rstride=1, cstride=1, linewidth=0, antialiased=False)
    m = plt.cm.ScalarMappable(cmap='viridis')
    m.set_array(c)
    fig.colorbar(m, ax=psiy_plot, shrink=0.5, aspect=10)
    psiy_plot.set_xlabel('Theta')
    psiy_plot.set_ylabel('Phi')
    psiy_plot.set_title('Psi y')
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    dpsitx_plot = fig.add_subplot(1, 2, 1, projection='3d')
    c = np.array([[np.angle(e[0]) + np.pi for e in l] for l in dpsi_theta])
    surf = dpsitx_plot.plot_surface(np.array([[e[0] for e in l] for l in mesh]), np.array([[e[1] for e in l] for l in mesh]), np.array([[np.abs(e[0]) for e in l] for l in dpsi_theta]), facecolors=plt.cm.viridis(c /(2 * np.pi)), rstride=1, cstride=1, linewidth=0, antialiased=False)
    m = plt.cm.ScalarMappable(cmap='viridis')
    m.set_array(c)
    fig.colorbar(m, ax=dpsitx_plot, shrink=0.5, aspect=10)
    dpsitx_plot.set_xlabel('Theta')
    dpsitx_plot.set_ylabel('Phi')
    dpsitx_plot.set_title('dPsi/dTheta x')
    plt.show()
    fig = plt.figure(figsize=(10, 7))
    dpsity_plot = fig.add_subplot(1, 2, 1, projection='3d')
    c = np.array([[np.angle(e[1]) + np.pi for e in l] for l in dpsi_theta])
    surf = dpsity_plot.plot_surface(np.array([[e[0] for e in l] for l in mesh]), np.array([[e[1] for e in l] for l in mesh]), np.array([[np.abs(e[1]) for e in l] for l in dpsi_theta]), facecolors=plt.cm.viridis(c /(2 * np.pi)), rstride=1, cstride=1, linewidth=0, antialiased=False)
    m = plt.cm.ScalarMappable(cmap='viridis')
    m.set_array(c)
    fig.colorbar(m, ax=dpsity_plot, shrink=0.5, aspect=10)
    dpsity_plot.set_xlabel('Theta')
    dpsity_plot.set_ylabel('Phi')
    dpsity_plot.set_title('dPsi/dTheta y')
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    dpsipx_plot = fig.add_subplot(1, 2, 1, projection='3d')
    c = np.array([[np.angle(e[0]) + np.pi for e in l] for l in dpsi_phi])
    surf = dpsipx_plot.plot_surface(np.array([[e[0] for e in l] for l in mesh]), np.array([[e[1] for e in l] for l in mesh]), np.array([[np.abs(e[0]) for e in l] for l in dpsi_phi]), facecolors=plt.cm.viridis(c /(2 * np.pi)), rstride=1, cstride=1, linewidth=0, antialiased=False)
    m = plt.cm.ScalarMappable(cmap='viridis')
    m.set_array(c)
    fig.colorbar(m, ax=dpsipx_plot, shrink=0.5, aspect=10)
    dpsipx_plot.set_xlabel('Theta')
    dpsipx_plot.set_ylabel('Phi')
    dpsipx_plot.set_title('dPsi/dPhi x')
    plt.show()
    fig = plt.figure(figsize=(10, 7))
    dpsipy_plot = fig.add_subplot(1, 2, 1, projection='3d')
    c = np.array([[np.angle(e[1]) + np.pi for e in l] for l in dpsi_phi])
    surf = dpsipy_plot.plot_surface(np.array([[e[0] for e in l] for l in mesh]), np.array([[e[1] for e in l] for l in mesh]), np.array([[np.abs(e[1]) for e in l] for l in dpsi_phi]), facecolors=plt.cm.viridis(c /(2 * np.pi)), rstride=1, cstride=1, linewidth=0, antialiased=False)
    m = plt.cm.ScalarMappable(cmap='viridis')
    m.set_array(c)
    fig.colorbar(m, ax=dpsipy_plot, shrink=0.5, aspect=10)
    dpsipy_plot.set_xlabel('Theta')
    dpsipy_plot.set_ylabel('Phi')
    dpsipy_plot.set_title('dPsi/dTheta y')
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    a_theta_plot = fig.add_subplot(1, 2, 1, projection='3d')
    surf = a_theta_plot.plot_surface(np.array([[e[0] for e in l] for l in mesh]), np.array([[e[1] for e in l] for l in mesh]), np.array(A_theta), cmap='viridis')
    a_theta_plot.set_xlabel('Theta')
    a_theta_plot.set_ylabel('Phi')
    a_theta_plot.set_title('A_theta')
    plt.show()
    fig = plt.figure(figsize=(10, 7))
    a_phi_plot = fig.add_subplot(1, 2, 1, projection='3d')
    surf = a_phi_plot.plot_surface(np.array([[e[0] for e in l] for l in mesh]), np.array([[e[1] for e in l] for l in mesh]), np.array(A_phi), cmap='viridis')
    a_phi_plot.set_xlabel('Theta')
    a_phi_plot.set_ylabel('Phi')
    a_phi_plot.set_title('A_phi')
    plt.show()
    fig = plt.figure(figsize=(10, 7))
    f_plot = fig.add_subplot(1, 2, 1, projection='3d')
    surf = f_plot.plot_surface(np.array([[e[0] for e in l] for l in mesh]), np.array([[e[1] for e in l] for l in mesh]), np.array(f), cmap='viridis')
    f_plot.set_xlabel('Theta')
    f_plot.set_ylabel('Phi')
    f_plot.set_title('f')
    plt.show()
    plt.show()
    return c
    #fig, (ax1, ax2) = plt.subplots(2, 1)
    #ax1.scatter([e[0] for e in mesh[0]], [k[0] for k in psis[0]], label='x', marker='+')
    #ax1.scatter([e[0] for e in mesh[0]], [k[1] for k in psis[0]], label='y', marker='+')
    #ax1.legend()
    #ax2.plot([e[0] for e in dists[0]], label='d_plus')
    #ax2.plot([e[1] for e in dists[0]], label='d_minus')
    #ax2.legend()
    #plt.scatter([e[0] for e in mesh[0]], [-k[0] for k in psis[0]])
    #plt.scatter([e[0] for e in mesh[0]], [-k[1] for k in psis[0]])
    #plt.show()

def d_psi(arg, theta, phi, m, d, h):
    if arg == 'theta':
        psi_plus = psi(hamiltonian(theta + h, phi, m, d))
        psi_minus = psi(hamiltonian(theta - h, phi, m, d))
        d_psi = (psi_plus - psi_minus) / (2 * h)
    else:
        psi_plus = psi(hamiltonian(theta, phi + h, m, d))
        psi_minus = psi(hamiltonian(theta, phi - h, m, d))
        d_psi = (psi_plus - psi_minus) / (2 * h)
    return d_psi

def A(arg, theta, phi, m, d, h):
    a = np.dot(psi(hamiltonian(theta, phi, m, d)).conj().T, d_psi(arg, theta, phi, m, d, h))
    return - 1j * a

def F(theta, phi, m, d, h, h_prime):
    d_theta_a_phi_plus = A('phi', theta + h_prime, phi, m, d, h)
    d_theta_a_phi_minus = A('phi', theta - h_prime, phi, m, d, h)
    d_phi_a_theta_plus = A('theta', theta, phi + h_prime, m, d, h)
    d_phi_a_theta_minus = A('theta', theta, phi - h_prime, m, d, h)
    return (d_theta_a_phi_plus - d_theta_a_phi_minus - (d_phi_a_theta_plus - d_phi_a_theta_minus)) / (2 * h_prime)

def F_wrapper(args):
    return F(*args)

def c(m, d, h, h_prime, h_second):
    integral = 0

    args = [(i, k, m, d, h, h_prime) for i in range(int(np.pi//h_second)) for k in range(int(2 * np.pi//h_second))] 
    with Pool(processes=4) as pool:
        for result in pool.imap_unordered(F_wrapper, args):
            integral += result

    #ie,
    #for i in range(int(np.pi//h_second)):
    #    for k in range(int(2 * np.pi//h_second)):
    #        integral += F(i, k, m, d, h, h_prime)
    return integral * h_second**2/(2 * np.pi)

#print(c(0, 1, .1, .1, .1))

def plot_A(arg, phi, m, d, h):
    A_values = []
    for i in range(100):
        A_values.append(A(arg, np.pi * i/100, phi, m, d, h))
    plt.plot(A_values)
    plt.show()

def plot_psi(phi, m, d):
    px = []
    py = []
    n = 100
    h = .00000000000001
    x = [h * (k- n/2)/n for k in range(n)]
    for i in x:
        p = psi(hamiltonian(np.pi/2 + i, phi, m, d))
        px.append(p[0])
        py.append(p[1])
    plt.scatter(x, px, marker='+')
    plt.scatter(x, py, marker='+')
    plt.show()

#plot_A('phi', 0, 0, 1, .1)

def plot_eigenstates(phi, m, d):
    e0 = []
    e1 = []
    p1 = []
    p2 = []
    n = 1000
    x = [4 * np.pi * i/n for i in range(n)]
    for i in x:
        energies, states = np.linalg.eig(hamiltonian(i, phi, m, d))
        e0.append(energies[0])
        e1.append(energies[1])
        p1.append(states[0])
        p2.append(states[1])
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(x, e0)
    ax1.plot(x, [k[0] for k in p1])
    ax1.plot(x, [k[1] for k in p1])
    ax2.plot(x, e1)
    ax2.plot(x, [k[0] for k in p2])
    ax2.plot(x, [k[1] for k in p2])
    plt.tight_layout()
    plt.show()