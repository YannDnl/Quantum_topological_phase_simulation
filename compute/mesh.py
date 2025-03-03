import copy
import importlib

import numpy as np
import matplotlib.pyplot as plt

import compute.hamiltonian as hamiltonian

importlib.reload(hamiltonian)

class MESH:
    def __init__(self, n: int, m: list, h: list, e: list, r: float, theta_min: float, theta_max: float, phi_min: float, phi_max: float, mesh: np.ndarray = None, delta_theta: float = None, delta_phi: float = None, q: int = None, p: int = None, axis: list = []) -> None:
        '''Creates an angle MESH object either from another mesh or from a map of polar space

        Args:
            n (int): number of spheres
            m (list): magnitude of the fields
            h (list): magnitude of the dipoles (previous d)
            e (list): relative phase of the spheres (list of 1 and -1, 1 being standard)
            r (float): strength of the coupling
            q (int): number of points on latitude (theta)
            p (int): number of points on longitude (phi)
            theta_min (float): minimum value of latitude
            theta_max (float): maximum value of latitude
            phi_min (float): minimum value of longitude
            phi_max (float): maximum value of longitude

        Returns:
            MESH: ndarray of tuples (latitude, longitude), variation of latitude along the first axis and of longitude along the second ([1][2])'''
        self.n: int = n
        self.m: list = m
        self.d: list = h
        self.e: list = e
        self.r: float = r
        self.axis: list = axis
        self.theta_max: float = theta_max
        self.theta_min: float = theta_min
        self.phi_max: float = phi_max
        self.phi_min: float = phi_min
        if mesh is None:
            self.q: int = q
            self.p: int = p
            self.delta_theta: float = (theta_max - theta_min)/q
            self.delta_phi: float = (phi_max - phi_min)/p
            self.mesh: np.ndarray = np.array([[(theta_min + (theta_max - theta_min) * k/q, phi_min + (phi_max - phi_min) * l/p) for l in range(p)] for k in range(q)])
        else:
            self.q: int = len(mesh)
            self.p: int = len(mesh[0])
            self.delta_theta: float = delta_theta
            self.delta_phi: float = delta_phi
            self.mesh: np.ndarray = mesh
    
    def getN(self) -> int:
        return self.n
    
    def getM(self) -> list:
        return self.m
    
    def getD(self) -> list:
        return self.d
    
    def getE(self) -> list:
        return self.e
    
    def getR(self) -> float:
        return self.r
    
    def getQ(self) -> int:
        return self.q
    
    def getP(self) -> int:
        return self.p
    
    def getDeltaTheta(self) -> float:
        return self.delta_theta
    
    def getDeltaPhi(self) -> float:
        return self.delta_phi
    
    def getMesh(self) -> np.ndarray:
        return self.mesh
    
    def getAxis(self) -> list:
        return self.axis

    def getThetaMax(self) -> float:
        return self.theta_max
    
    def getThetaMin(self) -> float:
        return self.theta_min
    
    def getPhiMax(self) -> float:
        return self.phi_max
    
    def getPhiMin(self) -> float:
        return self.phi_min
    
    def plot(self) -> None:
        '''Plot a mesh
    
        Args:
            mesh (np.ndarray): mesh to plot

        Returns:
            None'''
        dimensions = len(self.getMesh().shape) - 1
        x = np.linspace(self.getThetaMin(), self.getThetaMax(), self.getQ())
        y = np.linspace(self.getPhiMin(), self.getPhiMax(), self.getP())
        X, Y = np.meshgrid(y, x)

        fig, (ax1, ax2) = plt.subplots(dimensions, 2, subplot_kw={'projection': '3d'}, figsize=(12, 10))

        if dimensions == 1:
            ax1.plot_surface(X, Y, np.abs(self.getMesh()), cmap='viridis')
            ax1.set_title('norm as function of theta and phi')
            ax1.set_xlabel('phi')
            ax1.set_ylabel('theta')
            ax1.set_zlabel('norm')

            ax2.plot_surface(X, Y, np.angle(self.getMesh()), cmap='viridis')
            ax2.set_title('phase as function of theta and phi')
            ax2.set_xlabel('phi')
            ax2.set_ylabel('theta')
        
        elif dimensions == 2:
            up = self.getMesh()[:, :, 0]
            ax1[0].plot_surface(X, Y, np.abs(up), cmap='viridis')
            ax1[0].set_title('norm of up component as function of theta and phi')
            ax1[0].set_xlabel('phi')
            ax1[0].set_ylabel('theta')
            ax1[0].set_zlabel('norm')
            ax1[1].plot_surface(X, Y, np.angle(up), cmap='viridis')
            ax1[1].set_title('phase of up component as function of theta and phi')
            ax1[1].set_xlabel('phi')
            ax1[1].set_ylabel('theta')

            down = self.getMesh()[:, :, 1]
            ax2[0].plot_surface(X, Y, np.abs(down), cmap='viridis')
            ax2[0].set_title('norm of down component as function of theta and phi')
            ax2[0].set_xlabel('phi')
            ax2[0].set_ylabel('theta')
            ax2[0].set_zlabel('norm')
            ax2[1].plot_surface(X, Y, np.angle(down), cmap='viridis')
            ax2[1].set_title('phase of down component as function of theta and phi')
            ax2[1].set_xlabel('phi')
            ax2[1].set_ylabel('theta')

        plt.tight_layout()
        plt.show()
        return None

    def vdotMesh(self, other):
        '''returns the mesh of the dot products of two meshes'''
        dot_mesh = []
        for k, l in enumerate(self.getMesh()):
            dot_mesh.append([])
            for j, t in enumerate(l):
                dot_mesh[-1].append(np.vdot(t, other.getMesh()[k, j]))
        return MESH(self.getN(), self.getM(), self.getD(), self.getE(), self.getR(), self.getThetaMin(), self.getThetaMax(), self.getPhiMin(), self.getPhiMax(), np.array(dot_mesh), self.getDeltaTheta(), self.getDeltaPhi())

    def psiMesh(self):
        '''Builds the mesh of psi plus from the angle mesh'''
        psi_mesh = []
        for l in self.getMesh():
            psi_mesh.append([])
            for t in l:
                ham = hamiltonian.hamiltonian(self.getN(), t[0], t[1], self.getD(), self.getM(), self.getE(), self.getR())
                psi = hamiltonian.psi(ham)
                psi_mesh[-1].append(psi)
        return MESH(self.getN(), self.getM(), self.getD(), self.getE(), self.getR(), self.getThetaMin(), self.getThetaMax(), self.getPhiMin(), self.getPhiMax(), np.array(psi_mesh), self.getDeltaTheta(), self.getDeltaPhi())
    
    def psiMeshBis(self):
        '''Builds the mesh of psi plus from the angle mesh
        Used for debbuging'''
        psi_mesh = []
        for l in self.getMesh():
            psi_mesh.append([])
            for t in l:
                psi_mesh[-1].append(hamiltonian.psi_bis(t[0], t[1], self.getD(), self.getM()))
        return MESH(self.getN(), self.getM(), self.getD(), self.getE(), self.getR(), self.getThetaMin(), self.getThetaMax(), self.getPhiMin(), self.getPhiMax(), np.array(psi_mesh), self.getDeltaTheta(), self.getDeltaPhi())

    def differentiate_mesh(self, axis: str):
        '''Builds the mesh of the differentiated values along axis 'axis' of the original mesh. Made for psi and A'''
        dpsi_mesh = []
        if axis == 'theta':
            q = self.getQ()
            for k in range(q):
                if k == 0:
                    da: np.ndarray = self.mesh[k + 1] - self.mesh[k]
                elif k == q - 1:
                    da: np.ndarray = self.mesh[k] - self.mesh[k - 1]
                else:
                    da: np.ndarray = (self.mesh[k + 1] - self.mesh[k - 1])/2
                dpsi_mesh.append(da)
            dpsi_mesh = np.array(dpsi_mesh)
            dpsi_mesh = dpsi_mesh/self.getDeltaTheta()
        elif axis == 'phi':
            for l in self.mesh:
                dpsi_mesh.append([])
                p = self.getP()
                for k in range(p):
                    if k == 0:
                        d = l[k + 1] - l[k]
                    elif k == p - 1:
                        d = l[k] - l[k - 1]
                    else:
                        d = (l[k + 1] - l[k - 1])/2
                    dpsi_mesh[-1].append(d)
            dpsi_mesh = np.array(dpsi_mesh)
            dpsi_mesh = dpsi_mesh/self.getDeltaPhi()
        else:
            raise ValueError('wrong axis')
        l = copy.deepcopy(self.getAxis())
        l.append(axis)
        return MESH(self.getN(), self.getM(), self.getD(), self.getE(), self.getR(), self.getThetaMin(), self.getThetaMax(), self.getPhiMin(), self.getPhiMax(), dpsi_mesh, self.getDeltaTheta(), self.getDeltaPhi(), axis = l)

    def a_mesh(self, dpsi_mesh):
        '''returns a mesh of A with the same axis as dpsi_mesh'''
        q = self.getQ()
        p = self.getP()
        a_mesh = np.zeros((q, p), dtype=np.complex128)
        for k in range(q):
            for l in range(p):
                a_mesh[k][l] = -1j * np.vdot(self.getMesh()[k][l], dpsi_mesh.getMesh()[k][l])
        return MESH(self.getN(), self.getM(), self.getD(), self.getE(), self.getR(), self.getThetaMin(), self.getThetaMax(), self.getPhiMin(), self.getPhiMax(), a_mesh, self.getDeltaTheta(), self.getDeltaPhi(), axis = dpsi_mesh.getAxis())

    def f_mesh(self, d_a_mesh):
        '''returns a mesh of F theta phi'''
        if self.getAxis() == ['theta', 'phi'] and d_a_mesh.getAxis() == ['phi', 'theta']:
            ans = MESH(self.getM(), self.getD(), d_a_mesh.getMesh() - self.getMesh(), self.getDeltaTheta(), self.getDeltaPhi(), axis = self.getAxis())
        elif self.getAxis() == ['phi', 'theta'] and d_a_mesh.getAxis() == ['theta', 'phi']:
            ans = MESH(self.getN(), self.getM(), self.getD(), self.getE(), self.getR(), self.getThetaMin(), self.getThetaMax(), self.getPhiMin(), self.getPhiMax(), self.getMesh() - d_a_mesh.getMesh(), self.getDeltaTheta(), self.getDeltaPhi(), axis = d_a_mesh.getAxis())
        else:
            raise ValueError('wrong axis')
        return ans
    
    def getC(self):
        '''returns the Chern number'''
        return (np.sum(self.getMesh()) * self.getDeltaTheta() * self.getDeltaPhi())/(2 * np.pi)