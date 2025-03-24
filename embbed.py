from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_embedding(text):
    embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    ).data[0].embedding
    print("Generated embedding:", embedding)
    return embedding