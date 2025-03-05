# kinect_processor.py
import cv2
import numpy as np
from simple_point_cloud import PointCloud

class KinectProcessor:
    def __init__(self):
        """Initialize the Kinect processor."""
        self.color_capture = None
        self.depth_capture = None
        
        # Camera parameters (these will need calibration for accurate results)
        self.fx = 525.0  # focal length x
        self.fy = 525.0  # focal length y
        self.cx = 319.5  # optical center x
        self.cy = 239.5  # optical center y
        
    def initialize_cameras(self, color_index=0, depth_index=1):
        """Initialize color and depth cameras."""
        try:
            self.color_capture = cv2.VideoCapture(color_index)
            self.depth_capture = cv2.VideoCapture(depth_index)
            
            # Try to set high resolution
            self.color_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.color_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            if not self.color_capture.isOpened() or not self.depth_capture.isOpened():
                raise Exception("Failed to open camera streams")
                
            return True
        except Exception as e:
            print(f"Error initializing cameras: {e}")
            return False
    
    def capture_frames(self):
        """Capture color and depth frames."""
        if not self.color_capture or not self.depth_capture:
            return None, None
        
        # Capture color frame
        ret_color, color_frame = self.color_capture.read()
        if not ret_color:
            return None, None
            
        # Capture depth frame
        ret_depth, depth_frame = self.depth_capture.read()
        if not ret_depth:
            return color_frame, None
            
        return color_frame, depth_frame
    
    def convert_depth_to_point_cloud(self, depth_image, color_image=None, depth_scale=1000.0, max_depth=3.0):
        """Convert depth image to point cloud."""
        if depth_image is None:
            return None
            
        # Get image dimensions
        height, width = depth_image.shape[:2]
        
        # Create arrays for 3D points and colors
        points = []
        colors = []

        # Create meshgrid for pixel coordinates
        pixel_x, pixel_y = np.meshgrid(np.arange(width), np.arange(height))
        pixel_x = pixel_x.flatten()
        pixel_y = pixel_y.flatten()
        
        # Get depth values
        z = depth_image.flatten() / depth_scale  # Convert to meters
        
        # Filter out invalid depth values
        valid_indices = np.where((z > 0) & (z < max_depth))[0]
        
        # Calculate 3D coordinates
        x = (pixel_x[valid_indices] - self.cx) * z[valid_indices] / self.fx
        y = (pixel_y[valid_indices] - self.cy) * z[valid_indices] / self.fy
        z = z[valid_indices]
        
        # Combine XYZ coordinates
        points = np.vstack([x, y, z]).T
        
        # Get colors if color image is available
        if color_image is not None:
            colors = color_image.reshape(-1, 3)[valid_indices]
        
        # Create point cloud
        point_cloud = PointCloud(points=points, colors=colors if len(colors) > 0 else None)
        
        return point_cloud
    
    def release(self):
        """Release camera resources."""
        if self.color_capture:
            self.color_capture.release()
        if self.depth_capture:
            self.depth_capture.release()