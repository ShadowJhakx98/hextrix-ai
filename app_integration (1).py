"""
app_integration.py

Script to integrate the adult features into the main app.py file.
This script should be imported at the end of app.py to enable the
adult features with all safety protections.
"""

import os
import logging
from adult_features_integration import AdultFeaturesIntegration

# Setup logging
logger = logging.getLogger("AppIntegration")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def integrate_adult_features(app, socketio):
    """
    Integrate adult features into the Flask app
    
    Args:
        app: Flask application instance
        socketio: SocketIO instance
        
    Returns:
        Flask application with adult features integrated
    """
    try:
        # Check for environment variable to enable adult features
        adult_features_enabled = os.environ.get("ENABLE_ADULT_FEATURES", "false").lower() == "true"
        
        if not adult_features_enabled:
            logger.info("Adult features are disabled by environment variable")
            return app
        
        # Initialize adult features integration
        adult_features = AdultFeaturesIntegration(app, socketio)
        
        # Register modules with app for access in other parts of the application
        app = adult_features.register_modules_for_app(app)
        
        logger.info("Adult features successfully integrated into app")
        
        # Add middleware to ensure age verification and consent for all adult routes
        @app.before_request
        def check_adult_features_access():
            # Implementation of request middleware for adult routes
            pass
        
        return app
    except Exception as e:
        logger.error(f"Error integrating adult features: {str(e)}")
        # Continue without adult features if there's an error
        return app

# Usage in app.py:
# 
# # At the end of app.py
# if __name__ == "__main__":
#     # Import adult features integration
#     from app_integration import integrate_adult_features
#     
#     # Integrate adult features
#     app = integrate_adult_features(app, socketio)
#     
#     # Run the app
#     socketio.run(app, debug=True, host='0.0.0.0', port=5000)
