import matplotlib.patches as patches
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def addUnitDisk(ax) -> None:
    '''Adds a unit disk to the plot
    
    Args:
        ax (matplotlib.axes.Axes): the axis of the plot
    
    Returns:
        None'''
    unit_disk = patches.Circle((0, 0), radius=1, color='lightblue', alpha=0.5)
    ax.add_patch(unit_disk)

def open(type: str) -> matplotlib.axes.Axes:
    '''Opens a frame for a plot
    
    Args:
        type (str): the type of plot ie field, circle or norm_phase
        
    Returns:
        ax (matplotlib.axes.Axes): the axis of the plot'''
    if type == 'field':
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.3)

        max_radius = 1.5
        ax.quiver(0, 0, 0, max_radius, 0, 0, color='black', linewidth=0.5)
        ax.quiver(0, 0, 0, 0, max_radius, 0, color='black', linewidth=0.5)
        ax.quiver(0, 0, 0, 0, 0, max_radius, color='black', linewidth=0.5)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ans = ax
    
    elif type == 'circle':
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        for ax in axs:
            addUnitDisk(ax)
            ax.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Horizontal axis
            ax.axvline(0, color='black', linewidth=0.5, linestyle='--')  # Vertical axis
            ax.set_xlabel('Real')
            ax.set_ylabel('Imaginary')
        
        ans = axs
    
    elif type == 'norm_phase':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        ax1.set_title('Eigenstate components norms')
        ax1.set_xlabel('time')
        ax1.set_ylabel('norm')

        ax2.set_title('Eigenstate components phases')
        ax.set_xlabel('time')
        ax.set_ylabel('phase')
        
        ans = ax1, ax2
    
    else:
        raise ValueError('Invalid type')
    
    return ans

def close(type: str, axs) -> None:
    '''Closes the frame and plots a plot
    
    Args:
        type (str): the type of plot ie field, circle or norm_phase
        axs (matplotlib.axes.Axes): the axis of the plot
        
    Returns:
        None'''
    if type == 'field':
        axs.set_box_aspect([1, 1, 1])
        axs.set_xlim([-1.5, 1.5])
        axs.set_ylim([1.5, -1.5])
        axs.set_zlim([-1.5, 1.5])
        axs.legend()
    
    else:
        if type == 'circle':
            for ax in axs:
                ax.set_aspect('equal')
        for ax in axs:
            ax.legend()
        
        plt.tight_layout()
    
    plt.show()