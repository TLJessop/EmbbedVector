from openai import OpenAI
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()  # Load environment variables from .env file

# Initialize clients
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_embedding(text):
    embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    ).data[0].embedding
    return embedding
        
hello_embedding = generate_embedding("Hello, world!")

def create_named_index(name):
    index = pinecone.create_index(
        name=name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    return index
    
        