

from fileinput import filename
from flask import Blueprint, current_app
from app.services import file_service
from app.services.ai_model_service import AIModelService
from flask import Blueprint, jsonify, request
from app.models.ai_model import AIModel

ai_model_bp = Blueprint('ai_model', __name__)
ai_model_service = AIModelService()

@ai_model_bp.route('/', methods=['POST'])
def create():
    data = request.get_json()
    name = data.get('name')
    model_id = data.get('model_id')
    description = data.get('description')
    prompt = data.get('prompt')
    image = data.get('image')

    new_model : AIModel = ai_model_service.create_ai_model(name, model_id, description, prompt, image)

    return jsonify(new_model.to_dict()), 201

@ai_model_bp.route('/', methods=['GET'])
def get_all():
    try:
        return jsonify({
            "status": "success",
            "message": "AI models retrieved successfully",
            "data": ai_model_service.get_all_ai_models()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error al obtener los modelos: {str(e)}'
        }), 400
@ai_model_bp.route('/<name>', methods=['GET'])
def get_by_name_id(name: str):
    try:
        model_data = ai_model_service.get_ai_model_by_name_id(name)
        if not model_data:
            return jsonify({
                'status': 'error',
                'message': f'Model with name_id "{name}" not found'
            }), 404

        return jsonify({
            "status": "success",
            "message": "AI model retrieved successfully",
            "data": model_data
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error al obtener el modelo: {str(e)}'
        }), 400