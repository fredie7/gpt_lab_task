from pydantic import BaseModel

class SymptomInput(BaseModel):
    message: str
    history: list[dict] = []
    session_id: str
