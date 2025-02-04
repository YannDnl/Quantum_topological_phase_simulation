import numpy as np

def spinHamiltonian(theta: float, phi: float, d: float, m: float) -> np.ndarray:
    '''Return the Hamiltonian for a single sphere given field and dipole magnitude and orientation
    
    Args:
        theta (float): angle (latitude) of the dipole
        phi (float): angle (longitude) of the dipole
        d (float): magnitude of the dipole
        m (float): magnitude of the field
        
    Returns:
        np.ndarray: the spin Hamiltonian matrix
    '''
    return -d * np.array([[-m/d + np.cos(theta), np.sin(theta) * np.exp(- 1j * phi)], [np.sin(theta) * np.exp(1j * phi), m/d - np.cos(theta)]])

def hamiltonian(n: int, theta: float, phi: float, h: list, m: list, e: list, r: float) -> np.ndarray:
    '''Return the Hamiltonian for n spheres given field and dipole magnitude and orientation

    Args:
        n (int): number of spheres
        theta (float): angle (latitude) of the dipole
        phi (float): angle (longitude) of the dipole
        h (list): magnitudes of the dipoles
        m (list): list of the magnitudes of the fields
        e (list): list of the relative phase of the spheres (each element is 1 or -1, standard being 1)
        r (float): interaction strength
    
    Returns:
        np.ndarray: the spin Hamiltonian matrix'''
    ham = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    i = np.eye(2)
    sigma_z = np.array([[1, 0], [0, -1]])
    sig = 1
    for k in range(n):
        h1 = e[k] * spinHamiltonian(theta, phi, -h[k], -m[k])
        operator = 1
        for j in range(k):
            operator = np.kron(operator, i)
        operator = np.kron(operator, h1)
        for j in range(n - k - 1):
            operator = np.kron(operator, i)
        ham += operator
        sig = np.kron(sig, sigma_z)
    
    if n != 1:
        ham += r * sig
    return -1 * ham

def psi(n: int, hamiltonian: np.ndarray) -> list:
    '''Returns the list of the eigen vectors of the ground state of a given hamiltonian
    
    Args:
        n (int): the number of spheres and thus of eigen vectors
        hamiltonian (np.ndarray): the Hamiltonian matrix
    
    Returns:
        list: the list of the eigenstates of lowest energy of the Hamiltonian
    '''
    energies, states = np.linalg.eig(hamiltonian)
    i = np.argmax(energies)
    Psi = states[i]
    psis = []
    for k in range(n):
        psis.append(processPhase(Psi[2 * k: 2 * (k + 1)]))
    return psis

def processPhase(psi: np.ndarray) -> np.ndarray:
    '''Changes the complex phase of the eigenstate's components so that they are opposite
    
    Args:
        psi (np.ndarray): the eigenstate
    
    Returns:
        np.ndarray: the eigenstate with opposite phases
    '''
    return psi * np.exp(-1j * np.sum(np.angle(psi))/2)