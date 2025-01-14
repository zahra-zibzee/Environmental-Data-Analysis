import os
import numpy as np
import pandas as pd
import laspy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rasterio
from rasterio.plot import show


# Function to visualize LiDAR point cloud
def visualize_lidar(lidar_file):
    """Visualize a LiDAR point cloud file (.las or .laz)."""
    # Load LiDAR data
    las = laspy.read(lidar_file)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    print(points.shape)
    # Plot the point cloud
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="viridis", s=0.5
    )
    ax.set_title("LiDAR Point Cloud Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


# Function to visualize RGB raster image
def visualize_rgb(raster_file):
    """Visualize an RGB raster image."""
    with rasterio.open(raster_file) as src:
        plt.figure(figsize=(10, 10))
        show(src.read([1, 2, 3]), transform=src.transform)
        plt.title("RGB Orthophoto Visualization")
        plt.show()


# Main function
def main():
    lidar_file = "data/als/plot_01.las"
    raster_file = "data/ortho/plot_01.tif"

    if os.path.exists(lidar_file):
        print("Visualizing LiDAR Point Cloud...")
        visualize_lidar(lidar_file)
    else:
        print(f"LiDAR file not found: {lidar_file}")

    if os.path.exists(raster_file):
        print("Visualizing RGB Raster...")
        visualize_rgb(raster_file)
    else:
        print(f"Raster file not found: {raster_file}")


if __name__ == "__main__":
    main()
