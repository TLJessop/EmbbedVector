from openai import OpenAI
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, Index
from datasets import load_dataset
from tqdm import tqdm
import time

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

def index_data(index_name):
    dataset = load_data()
    for item in tqdm(dataset):
        embedding = generate_embedding(item["text"])
        pc.index(index_name).upsert(
            vectors=[(str(item["index"]), embedding, {"text": item["text"]})]  # Using index as id and storing text as metadata
        )
    return

def get_index_list():
    return pc.list_indexes()
