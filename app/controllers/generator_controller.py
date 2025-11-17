from flask import Blueprint, current_app, jsonify, request, send_from_directory
from app.services.file_service import FileService
from werkzeug.datastructures import FileStorage
from app.config import Config

from app.services.generator_service import GeneratorService

generator_bp = Blueprint('generator', __name__)
file_service = FileService()
generator_service = GeneratorService()

@generator_bp.route('/change_model', methods=['POST'])
def change_model():
    data = request.get_json()
    if not data or 'model_name' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No model_name provided'
        }), 400
    
    model_name = data['model_name']
    try:
        generator_service._load_image_model(model_name=model_name)
        return jsonify({
            'status': 'success',
            'message': f'Model changed to {model_name}'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error changing model: {str(e)}'
        }), 500

@generator_bp.route('/image-to-image', methods=['POST'])
def generate_image2image():
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file part'
        }), 400
    
    file : FileStorage = request.files['file']

    # En multipart/form-data, los datos JSON llegan como texto en request.form
    data_str = request.form.get('data')  # "data" es la key que envías en Postman
    data = {}
    if data_str:
        import json
        data = json.loads(data_str)

    filename: str = data.get('filename', file.filename)
    prompt: str = data.get('prompt', "")
    num_inference_steps: int = int(data.get('num_inference_steps', 50))
    strength: float = float(data.get('strength', 0.8))
    guidance_scale: float = float(data.get('guidance_scale', 7.5))
    num_images_per_prompt: int = int(data.get('num_images_per_prompt', 1))
    model_name: str = data.get('model_name', "sketch_to_anime_lora_final5")
    
    print(data)
    # Guardar el archivo primero
    saved_file_result = file_service.save_file(file)
    # Procesar con the generator_service
    result = generator_service.image_to_image(
        f"{Config.UPLOAD_FOLDER}/{saved_file_result['filename']}",
        prompt,
        num_inference_steps,
        strength,
        guidance_scale,
        num_images_per_prompt,
        model_name
    )

    if result['status'] == 'error':
        return jsonify(result), 400
    else:
        # return send_from_directory(current_app.config['RESULT_FOLDER'], result['filename'])
        return jsonify(result), 200

@generator_bp.route('/image-to-sketch', methods=['POST'])
def generate_image2sketch():
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file part'
        }), 400
    
    file : FileStorage = request.files['file']

    saved_file_result = file_service.save_file(file)
    # Procesar con the generator_service
    result = generator_service.image_to_sketch(f"{Config.UPLOAD_FOLDER}/{saved_file_result['filename']}")

    if result['status'] == 'error':
        return jsonify(result), 400
    else:
        # return send_from_directory(current_app.config['RESULT_FOLDER'], result['filename'])
        return jsonify(result), 200

@generator_bp.route('/text-to-image', methods=['POST'])
def generate_text2image():
    
    data = request.get_json()
    
    if data is {}:
    
        return jsonify({
            'status': 'error',
            'message': 'No data part'
        }), 400
    print(data)
    prompt: str = data.get('promp') if 'promp' in data else data.get('prompt', "")
    num_inference_steps: int = int(data.get('num_inference_steps', 50))
    strength: float = float(data.get('strength', 0.8))
    guidance_scale: float = float(data.get('guidance_scale', 7.5))
    num_images_per_prompt: int = int(data.get('num_images_per_prompt', 1))
    model_name: str = data.get('model_name', "sketch_to_anime_lora_final5")

    result = generator_service.text_to_image(prompt, num_inference_steps, strength, guidance_scale, num_images_per_prompt, model_name=model_name)
    if result['status'] == 'error':
        return jsonify(result), 400
    else:
        # return send_from_directory(current_app.config['RESULT_FOLDER'], result['filename'])
        return jsonify(result), 200
    