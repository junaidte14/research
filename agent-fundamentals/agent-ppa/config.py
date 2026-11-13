import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv() # loads the environment variables from .env file

@dataclass(frozen=True)
class Settings:
    model: str

def load_settings() -> Settings:
    model = os.getenv("LLM_Model")

    if not model:
        raise ValueError(
            "LLM Model is missing"
            "Please define it inside the .env file"
        )

    return Settings(model=model)

settings = load_settings()