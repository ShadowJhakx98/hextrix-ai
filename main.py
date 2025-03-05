# main.py
import time
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO

from simple_point_cloud import PointCloud
from kinect_processor import KinectProcessor
from web_interface import PointCloudVisualizer

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create instances
kinect = KinectProcessor()
visualizer = PointCloudVisualizer(app, socketio)

# Processing thread
def processing_thread():
    if not kinect.initialize_cameras():
        print("Failed to initialize cameras")
        return
    
    print("Cameras initialized successfully")
    
    try:
        while True:
            # Capture frames
            color_frame, depth_frame = kinect.capture_frames()
            
            if color_frame is not None:
                # Update web UI with camera views
                visualizer.update_camera_view(color_frame, depth_frame)
                
                # If depth frame is available, create point cloud
                if depth_frame is not None:
                    # Convert depth to point cloud
                    point_cloud = kinect.convert_depth_to_point_cloud(depth_frame, color_frame)
                    
                    if point_cloud:
                        # Downsample for performance
                        downsampled = point_cloud.voxel_downsample(0.05)
                        
                        # Remove outliers
                        filtered, _ = downsampled.remove_statistical_outliers(20, 2.0)
                        
                        # Update web UI with point cloud
                        visualizer.update_point_cloud(filtered)
            
            # Sleep to control frame rate
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("Processing stopped by user")
    finally:
        kinect.release()

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    # Start processing in a separate thread
    thread = threading.Thread(target=processing_thread)
    thread.daemon = True
    thread.start()
    
    # Start Flask server
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)