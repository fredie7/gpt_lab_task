#Import the BaseModel from pydantic library for user data validation
from pydantic import BaseModel

# Define the server's schema
class SymptomInput(BaseModel):
    #message: to track messages from each user
    message: str
    #history: to store message history in the session
    history: list[dict] = []
    #session_id: to keep track of each user's interaction
    session_id: str
