from openai import OpenAI
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, Index

load_dotenv()  # Load environment variables from .env file

# Initialize clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_named_index(name):
    index = pc.create_index(
        name=name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(pc.describe_index(name))
    return index

def get_index_list():
    return pc.list_indexes()
