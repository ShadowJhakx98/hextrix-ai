# simple_point_cloud.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PointCloud:
    def __init__(self, points=None, colors=None):
        """Initialize a point cloud with optional points and colors."""
        self.points = np.asarray(points) if points is not None else np.zeros((0, 3))
        self.colors = np.asarray(colors) if colors is not None else np.zeros((0, 3))
        
    def add_points(self, points, colors=None):
        """Add points to the point cloud."""
        points = np.asarray(points)
        if points.shape[1] != 3:
            raise ValueError("Points must be Nx3 array")
            
        if colors is not None:
            colors = np.asarray(colors)
            if colors.shape != points.shape:
                raise ValueError("Colors must have same shape as points")
            
            if self.points.shape[0] == 0:
                self.points = points
                self.colors = colors
            else:
                self.points = np.vstack([self.points, points])
                self.colors = np.vstack([self.colors, colors])
        else:
            if self.points.shape[0] == 0:
                self.points = points
                self.colors = np.zeros_like(points)
            else:
                self.points = np.vstack([self.points, points])
                self.colors = np.vstack([self.colors, np.zeros_like(points)])
    
    def voxel_downsample(self, voxel_size):
        """Downsample point cloud using voxel grid."""
        if self.points.shape[0] == 0:
            return PointCloud()
            
        # Compute voxel indices for each point
        voxel_indices = np.floor(self.points / voxel_size).astype(int)
        
        # Create a dictionary to store points in each voxel
        voxel_dict = {}
        for i in range(self.points.shape[0]):
            idx = tuple(voxel_indices[i])
            if idx in voxel_dict:
                voxel_dict[idx].append(i)
            else:
                voxel_dict[idx] = [i]
        
        # Compute downsampled points and colors
        downsampled_points = []
        downsampled_colors = []
        
        for idx, point_indices in voxel_dict.items():
            # Average points in voxel
            voxel_points = self.points[point_indices]
            center = np.mean(voxel_points, axis=0)
            downsampled_points.append(center)
            
            # Average colors in voxel
            if self.colors.shape[0] > 0:
                voxel_colors = self.colors[point_indices]
                color = np.mean(voxel_colors, axis=0)
                downsampled_colors.append(color)
        
        result = PointCloud()
        result.points = np.array(downsampled_points)
        if downsampled_colors:
            result.colors = np.array(downsampled_colors)
        
        return result
    
    def remove_statistical_outliers(self, nb_neighbors, std_ratio):
        """Remove outliers using statistical analysis."""
        if self.points.shape[0] <= nb_neighbors:
            return self, np.ones(self.points.shape[0], dtype=bool)
            
        # Compute the mean distance to k nearest neighbors for each point
        from scipy.spatial import KDTree
        tree = KDTree(self.points)
        distances, _ = tree.query(self.points, k=nb_neighbors + 1)
        
        # Exclude the point itself (distance=0)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        
        # Compute the mean and std of average distances
        mu = np.mean(avg_distances)
        sigma = np.std(avg_distances)
        
        # Identify outliers
        valid_indices = avg_distances <= (mu + std_ratio * sigma)
        
        # Create a new point cloud without outliers
        result = PointCloud()
        result.points = self.points[valid_indices]
        if self.colors.shape[0] > 0:
            result.colors = self.colors[valid_indices]
        
        return result, valid_indices
    
    def visualize(self, title="Point Cloud"):
        """Visualize the point cloud using matplotlib."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        if self.colors.shape[0] == self.points.shape[0]:
            # Normalize colors to 0-1 range if needed
            colors = self.colors
            if np.max(colors) > 1:
                colors = colors / 255.0
            ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], 
                      c=colors, s=1)
        else:
            ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], s=1)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Auto-scale axes
        max_range = np.max([
            np.max(self.points[:, 0]) - np.min(self.points[:, 0]),
            np.max(self.points[:, 1]) - np.min(self.points[:, 1]),
            np.max(self.points[:, 2]) - np.min(self.points[:, 2])
        ])
        
        mid_x = (np.max(self.points[:, 0]) + np.min(self.points[:, 0])) / 2
        mid_y = (np.max(self.points[:, 1]) + np.min(self.points[:, 1])) / 2
        mid_z = (np.max(self.points[:, 2]) + np.min(self.points[:, 2])) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        plt.show()
        
    def save_to_ply(self, filename):
        """Save point cloud to PLY file."""
        with open(filename, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {self.points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            
            if self.colors.shape[0] == self.points.shape[0]:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                
            f.write("end_header\n")
            
            # Write vertex data
            for i in range(self.points.shape[0]):
                line = f"{self.points[i, 0]} {self.points[i, 1]} {self.points[i, 2]}"
                
                if self.colors.shape[0] == self.points.shape[0]:
                    # Convert colors to 0-255 range if needed
                    r, g, b = self.colors[i]
                    if np.max(self.colors) <= 1:
                        r, g, b = int(r * 255), int(g * 255), int(b * 255)
                    else:
                        r, g, b = int(r), int(g), int(b)
                    line += f" {r} {g} {b}"
                    
                f.write(line + "\n")