from google import genai
API_KEY = "AIzaSyCnxcnu8VnxlcQrVj8ehfWHnaWd4baleK8"
client = genai.Client(api_key=API_KEY)
for m in client.models.list():
    print(f"Name: {m.name}, Actions: {m.supported_actions}")
