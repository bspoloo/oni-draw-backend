from sys import exception
from app.models.ai_model import AIModel
from app.dtos.ai_model_input_dto import AIModelInputDTO
from app.dtos.ai_model_output_dto import AIModelOutputDTO
from app.database import db

class AIModelService:
    _model : AIModel = None
    def __init__(self):
        self.model_input_dto: AIModelInputDTO = AIModelInputDTO()
        self.model_output_dto_list: AIModelOutputDTO = AIModelOutputDTO(many=True)
        self.model_output_dto: AIModelOutputDTO = AIModelOutputDTO()
    
    def create_ai_model(self, name, model_id, description,prompt, image):
        new_model = AIModel(
            name=name,
            model_id=model_id,
            description=description,
            prompt=prompt,
            image=image
        )

        db.session.add(new_model)
        db.session.commit()
        return new_model

    def get_all_ai_models(self):
        models = AIModel.query.all()
        return self.model_output_dto_list.dump(models)
    
    def get_ai_model_by_name_id(self, name):
        model = AIModel.query.filter_by(model_id=name).first()
        if not model:
            return exception("Model not found")
        
        return self.model_output_dto.dump(model)
    
    def set_model(self, model: AIModel)-> None:
        self._model = model

    def get_model(self) -> AIModel:
        return self._model