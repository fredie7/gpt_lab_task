from pydantic import BaseModel

# Define the server's schema
class SymptomInput(BaseModel):
    message: str
    history: list[dict] = []
    session_id: str
