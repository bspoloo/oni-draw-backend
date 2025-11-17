from app.configuration.auditable_entity import AuditableEntity
from app.database import db
from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

class AIModel(AuditableEntity):
    __tablename__ = 'ai_models'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text(), nullable=True)
    prompt: Mapped[str] = mapped_column(Text(), nullable=True)
    image: Mapped[str] = mapped_column(String(255), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'model_id': self.model_id,
            'description': self.description,
            'prompt': self.prompt,
            'image': self.image,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }