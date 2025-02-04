import numpy as np
import matplotlib.pyplot as plt

def hamiltonian(theta, M, d):
    return np.array([[M - d * np.cos(theta), - d * np.sin(theta)], [- d * np.sin(theta), M + d * np.cos(theta)]])

def sigma_z(hamil):
    _, eigenvectors = np.linalg.eig(hamil)
    if eigenvectors[0][0] * eigenvectors[0][1] >= 0:
        eigenvector = eigenvectors[0]
    else:
        eigenvector = eigenvectors[1]
    print(eigenvector)
    sigma_z = np.array([[1, 0], [0, -1]])
    expectation_values = np.dot(eigenvector.conj().T, np.dot(sigma_z, eigenvector))
    return expectation_values

def churn_number(M, d):
    h_zero = hamiltonian(0 + .1, M, d)
    sigma_zero = sigma_z(h_zero)
    h_pi = hamiltonian(np.pi - .1, M, d)
    sigma_pi = sigma_z(h_pi)
    return - (sigma_zero - sigma_pi) / 2

def plot_churn_number():
    d = 1
    M = np.linspace(0, 2 * d, 100)
    c = []
    for m in M:
        c.append(churn_number(m, d))
    #print(c[-1])
    plt.plot(M, c)
    plt.show()
    