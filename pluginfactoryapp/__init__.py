# pluginfactoryapp/__init__.py

import os
from flask import Flask
from flask_cors import CORS

def create_app():
    """Application factory function."""
    app = Flask(__name__)
    CORS(app)

    app.config.from_pyfile(os.path.join('..', 'config.py'))
    
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    with app.app_context():
        # Import services here
        from .services import vertex_service, file_service

        # --- NEW: Initialize services directly ---
        vertex_service.initialize_vertex_ai()
        file_service.create_initial_directories_and_files()
        
        # Import and register routes
        from . import routes

    return app