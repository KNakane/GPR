import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

def main():

    # Get input data
    x = np.sin(np.linspace(0, 2 * np.pi, 1000)) * 2
    y = np.cos(np.linspace(0, 2 * np.pi, 1000))
    xy = np.stack([x,y], axis=1)
    z = x * y

    # mesh grid data
    X, Y = np.meshgrid(x, y)
    Z = interpolate.griddata(xy, z, (X, Y),method='cubic')


    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X,Y,Z,cmap='jet',antialiased=False)
    fig.colorbar(surf)
    plt.show()
    return

if __name__ == "__main__":
    main()