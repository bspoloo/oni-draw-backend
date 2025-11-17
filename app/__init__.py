from flask import Flask
from .config import Config
from app.controllers.file_controller import file_bp
from app.controllers.generator_controller import generator_bp
from app.controllers.ai_model_controller import ai_model_bp
import os
from flask_cors import CORS
from flask_migrate import Migrate
from app.database import db
from flask_jwt_extended import JWTManager


def create_app():
    # Configurar las rutas de templates y static correctamente
    # template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
    # static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
    # app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)
    db.init_app(app)
    Migrate(app, db)
    
    from app.models.ai_model import AIModel
    # jwt = JWTManager(app)
    
    app.config.from_object(Config)
    app.register_blueprint(file_bp, url_prefix="/api/files")
    app.register_blueprint(generator_bp, url_prefix="/api/generator")
    app.register_blueprint(ai_model_bp, url_prefix="/api/ai_models")
    
    return app