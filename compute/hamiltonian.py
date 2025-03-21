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

def hamiltonian(n: int, theta: float, phi: float, h: list, m: list, e: list, r: np.ndarray) -> np.ndarray:
    '''Return the Hamiltonian for n spheres given field and dipole magnitude and orientation

    Args:
        n (int): number of spheres
        theta (float): angle (latitude) of the dipole
        phi (float): angle (longitude) of the dipole
        h (list): magnitudes of the dipoles
        m (list): list of the magnitudes of the fields
        e (list): list of the relative phase of the spheres (each element is 1 or -1, standard being 1)
        r (np.array, 3x3): 0:x, 1:y, 2:z, interaction strength
    
    Returns:
        np.ndarray: the spin Hamiltonian matrix'''
    ham = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    i = np.eye(2)
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sig_xx = np.kron(sigma_x, sigma_x)
    sig_yy = np.kron(sigma_y, sigma_y)
    sig_zz = np.kron(sigma_z, sigma_z)
    sig_xy = np.kron(sigma_x, sigma_y)
    sig_yz = np.kron(sigma_y, sigma_z)
    sig_zx = np.kron(sigma_z, sigma_x)
    sig_yx = np.kron(sigma_y, sigma_x)
    sig_xz = np.kron(sigma_x, sigma_z)
    sig_zy = np.kron(sigma_z, sigma_y)
    sig = [[sig_xx, sig_xy, sig_xz],
           [sig_yx, sig_yy, sig_yz],
           [sig_zx, sig_zy, sig_zz]]
    for k in range(n):
        h1 = e[k] * spinHamiltonian(theta, phi, -h[k], -m[k])
        operator = 1
        for j in range(k):
            operator = np.kron(operator, i)
        operator = np.kron(operator, h1)
        for j in range(n - k - 1):
            operator = np.kron(operator, i)
        ham += operator

    if n != 1:
        for k in range(3):
            for l in range(3):
                ham += r[k][l] * sig[k][l]
    return -1 * ham

def psi(hamiltonian: np.ndarray) -> list:
    '''Returns the list of the eigen vectors of the ground state of a given hamiltonian
    
    Args:
        n (int): the number of spheres and thus of eigen vectors
        hamiltonian (np.ndarray): the Hamiltonian matrix
    
    Returns:
        list: the list of the eigenstates of lowest energy of the Hamiltonian
    '''
    energies, states = np.linalg.eig(hamiltonian)
    i = np.argmax(energies)
    Psi = processPhase(states[i])
    return Psi

def processPhase(psi: np.ndarray) -> np.ndarray:
    '''Changes the complex phase of the eigenstate's components so that they are opposite
    
    Args:
        psi (np.ndarray): the eigenstate
    
    Returns:
        np.ndarray: the eigenstate with first component real
    '''
    return psi * np.exp(-1j * np.angle(psi[0]))

def getPsi(input) -> np.ndarray:
    k, j, n, theta, phi, h, m, e, r = input
    '''Returns the ground state of a system of n spheres given field and dipole magnitude and orientation
    
    Args:
        n (int): number of spheres
        theta (float): angle (latitude) of the dipole
        phi (float): angle (longitude) of the dipole
        h (list): magnitudes of the dipoles
        m (list): list of the magnitudes of the fields
        e (list): list of the relative phase of the spheres (each element is 1 or -1, standard being 1)
        r (float): interaction strength
    
    Returns:
        np.ndarray: the ground state of the system
    '''
    ham = hamiltonian(n, theta, phi, h, m, e, r)
    return k, j, psi(ham)