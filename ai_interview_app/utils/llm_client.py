from openai import OpenAI
from config import settings

def get_llm_client():
    return OpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

def get_model_name():
    return settings.GROQ_MODEL

def supports_json_mode():
    # mixtral does NOT support json_object mode on Groq
    return "mixtral" not in settings.GROQ_MODEL.lower()