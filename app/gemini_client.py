
import os

from dotenv import load_dotenv
from google import genai


def create_gemini_client():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
 model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Please configure it in the environment."
        )

    client = genai.Client(api_key=api_key)

    return client, model_name
