import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def _visualize_point_cloud(points):
    """
    Visualize the 3D point cloud using Open3D.

    Args:
        points (np.ndarray): Array of 3D points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    ax.scatter(x, y, z, c='b', marker='o', s=1)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

# Load the point cloud
points = np.load("/Users/Varun/Home/Penn/Other/Take Homes/dimensionalOS_take_home/point_cloud.npy")
# _visualize_point_cloud(points)