from marshmallow import Schema, fields, validate

class AIModelOutputDTO(Schema):
    id = fields.Int(required=True)
    name = fields.Str(required=True, validate=validate.Length(min=1))
    model_id = fields.Str(required=True, validate=validate.Length(min=1))
    description = fields.Str(required=False, allow_none=True)
    prompt = fields.Str(required=False, allow_none=True)
    image = fields.Str(required=True, validate=validate.Length(min=1))
