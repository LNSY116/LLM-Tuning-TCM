from google import genai
import os
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
for m in client.models.list():
    print(f"Name: {m.name}, Actions: {m.supported_actions}")
