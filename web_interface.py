# web_interface.py
import json
import base64
import cv2
import numpy as np
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

class PointCloudVisualizer:
    def __init__(self, app, socketio):
        """Initialize the visualizer with Flask and SocketIO instances."""
        self.app = app
        self.socketio = socketio
        
        # Register routes and events
        self._register_routes()
        self._register_socketio_events()
        
    def _register_routes(self):
        """Register Flask routes."""
        @self.app.route('/visualize')
        def visualize():
            return render_template('visualize.html')
        
        @self.app.route('/point_cloud_data')
        def point_cloud_data():
            # This would be a simple API endpoint to get the latest point cloud data
            return jsonify({'status': 'error', 'message': 'No point cloud data available'})
    
    def _register_socketio_events(self):
        """Register SocketIO event handlers."""
        @self.socketio.on('request_point_cloud')
        def handle_request_point_cloud():
            # Client is requesting the latest point cloud data
            emit('point_cloud_update', {'status': 'error', 'message': 'No data available'})
    
    def update_point_cloud(self, point_cloud, max_points=10000):
        """Send updated point cloud data to connected clients."""
        if point_cloud is None or point_cloud.points.shape[0] == 0:
            return False
            
        # Downsample if needed
        if point_cloud.points.shape[0] > max_points:
            downsampling_ratio = max_points / point_cloud.points.shape[0]
            indices = np.random.choice(
                point_cloud.points.shape[0], 
                size=max_points, 
                replace=False
            )
            points = point_cloud.points[indices].tolist()
            colors = point_cloud.colors[indices].tolist() if point_cloud.colors.shape[0] > 0 else []
        else:
            points = point_cloud.points.tolist()
            colors = point_cloud.colors.tolist() if point_cloud.colors.shape[0] > 0 else []
        
        # Send data via SocketIO
        self.socketio.emit('point_cloud_update', {
            'status': 'success',
            'points': points,
            'colors': colors
        })
        
        return True
    
    def update_camera_view(self, color_frame, depth_frame=None):
        """Send camera frames to connected clients."""
        if color_frame is None:
            return False
            
        # Convert color frame to JPEG
        _, buffer = cv2.imencode('.jpg', color_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        color_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # If depth frame exists, convert it to a colormap for visualization
        depth_b64 = None
        if depth_frame is not None:
            # Normalize depth for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            _, buffer = cv2.imencode('.jpg', depth_colormap, [cv2.IMWRITE_JPEG_QUALITY, 80])
            depth_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send frames to clients
        self.socketio.emit('camera_update', {
            'color_frame': f'data:image/jpeg;base64,{color_b64}',
            'depth_frame': f'data:image/jpeg;base64,{depth_b64}' if depth_b64 else None
        })
        
        return True