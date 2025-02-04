import numpy as np
import plot.frame as frame

def plotVector(genre: str, e) -> None:
    '''Plot a vector in a given form
    
    Args:
        genre (str): form of the vector, ie psi or field
        e: vector to plot (d, m, theta, phi) for field, psi for eigenstate
    
    Returns:
        None
    '''
    if genre == 'psi':
        up = e[0]
        down = e[1]

        form = 'circle'
        axs = frame.open(form)

        axs[0].quiver(0, 0, up.real, up.imag, angles='xy', scale_units='xy', scale=1, color='red', label=f'up component')  # Complex number
        axs[1].quiver(0, 0, down.real, down.imag, angles='xy', scale_units='xy', scale=1, color='red', label=f'down component')  # Complex number

        frame.close(form, axs)
    elif genre == 'field':
        d, m, theta, phi = e
        x = d * np.sin(theta) * np.cos(phi)
        y = d * np.sin(theta) * np.sin(phi)
        z = d * np.cos(theta) - m

        form = 'field'
        ax = frame.open(form)

        ax.quiver(0, 0, 0, x, y, z, color='red', linewidth=2, label="Vector")

        frame.close(form, ax)
    else:
        raise ValueError(f'Unknown form {form}')

def plotPath(genre: str, e) -> None:
    '''Plot a path in a given form
    
    Args:
        genre (str): form of the path, ie array, bounds (for field), circle or norm_phase (for eigenstates)
        e: path to plot (d, m, thetas, phis) for array, (d, m, theta_min, theta_max, phi_min, phi_max) for bounds, psis for eigenstates
    
    Returns:
        None'''
    if genre == 'bounds':
        d, m, theta_min, theta_max, phi_min, phi_max = e
        n: int = 100
        thetas = np.linspace(theta_min, theta_max, n)
        phis = np.linspace(phi_min, phi_max, n)
        genre = 'field'
    elif genre == 'array':
        d, m, thetas, phis = e
        genre = 'field'
    if genre == 'field':
        x = d * np.sin(thetas) * np.cos(phis)
        y = d * np.sin(thetas) * np.sin(phis)
        z = d * np.cos(thetas) - m

        form = 'field'
        ax = frame.open(form)

        ax.plot(x, y, z, color='pink', linewidth=2, label="path")
        ax.quiver(0, 0, 0, x[-1], y[-1], z[-1], color='red', linewidth=2, label="last state")

        frame.close(form, ax)
    else:
        up = e[:, 0]
        down = e[:, 1]
        if genre == 'circle':
            form = 'circle'
            axs = frame.open(form)

            axs[0].plot(up.real, up.imag, color='pink', linewidth=4, label="up path")
            axs[1].plot(down.real, down.imag, color='pink', linewidth=4, label="down path")

            axs[0].quiver(0, 0, up[-1].real, up[-1].imag, angles='xy', scale_units='xy', scale=1, color='red', label=f'last up state')  # Complex number
            axs[1].quiver(0, 0, down[-1].real, down[-1].imag, angles='xy', scale_units='xy', scale=1, color='red', label=f'last down state')  # Complex number

            frame.close(form, axs)
        elif genre == 'norm_phase':
            form = 'norm_phase'
            ax1, ax2 = frame.open(form)

            up_norms = np.abs(up)
            down_norms = np.abs(down)
            ax1.plot(up_norms, color='blue', label='up norm')
            ax1.plot(down_norms, color='green', label='down norm')

            up_phi = np.angle(up)
            down_phi = np.angle(down)
            ax2.plot(up_phi, color='pink', linewidth=4, label="up phase")
            ax2.plot(down_phi, color='pink', linewidth=4, label="down phase")

            frame.close(form, (ax1, ax2))
        
        else:
            raise ValueError(f'Unknown form {form}')