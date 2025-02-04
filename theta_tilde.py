import numpy as np
import matplotlib.pyplot as plt

def theta_tilde(m_sur_d: float, theta: float) -> float:
    '''Computes the tilde theta for a given theta'''
    return np.arccos((-m_sur_d + np.cos(theta))/np.sqrt(1 + m_sur_d**2 - 2 * m_sur_d * np.cos(theta)))

def plot():
    n = 1000
    thetas = np.linspace(0, np.pi, n)
    m = [0, .5, .9, 1.1, 2, 64]
    mm = [.1, .2, .3, .4, .6, .7, .8, .95, .99, .999, 1.001, 1.01, 1.05, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.5, 3, 4, 6, 8, 16]
    for m_sur_d in mm:
        plt.plot(thetas, [theta_tilde(m_sur_d, t) for t in thetas], color='lightblue')
    for m_sur_d in m:
        plt.plot(thetas, [theta_tilde(m_sur_d, t) for t in thetas], label=f'm/d = {m_sur_d}')
    thetas = np.delete(thetas, 0)
    plt.plot(thetas, [theta_tilde(1, t) for t in thetas], label='m/d = 1', color='black')
    #plt.plot([0, np.pi], [np.pi/2, np.pi], label='limit', color='black')
    plt.legend()
    plt.xlabel('theta')
    plt.ylabel('theta tilde')
    plt.title('theta tilde as a function of theta')
    plt.show()

