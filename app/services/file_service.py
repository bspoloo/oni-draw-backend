import uuid
from datetime import datetime
from werkzeug.datastructures import FileStorage
from flask import current_app, send_from_directory
import os
from functions.validate_image import allowed_file


class FileService:
    
    def save_file(self, file: FileStorage):
        if file.filename == '':
            return {
                'status': 'error',
                'message': 'No selected file'
            }
    
        if file and allowed_file(file.filename):
            extension = file.filename.rsplit(".", 1)[1].lower()
            # filename = secure_filename(file.filename)
            
            # filename = f"{uuid.uuid4().hex}.{extension}" 
            filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"

            upload_folder = current_app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)

            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            return {
                'status': 'success',
                'message': 'File uploaded successfully', 
                'filename': filename,
                # 'path': file_path
                }

        return {
            'status': 'error',
            'message': 'Invalid file type'
        }
    
    def get_file_results(self, filename: str):
        return send_from_directory(current_app.config['RESULT_FOLDER'], filename)
    
    def get_file_uploads(self, filename: str):
        return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
    
    def get_file_models(self, filename: str):
        return send_from_directory(current_app.config['MODELS_IMAGES_FOLDER'], filename)
    
    def get_file_sketches(self, filename: str):
        return send_from_directory(current_app.config['SKETCH_FOLDER'], filename)