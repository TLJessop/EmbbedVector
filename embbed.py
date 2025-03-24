from openai import OpenAI
import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
from tqdm import tqdm
import time

load_dotenv()  # Load environment variables from .env file

# Initialize clients
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_embedding(text):
    embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    ).data[0].embedding
    print("Generated embedding:", embedding)
    return embedding

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
    print(pinecone.describe_index(name))
    return index

def index_data(index_name):
    dataset = load_data()
    for item in tqdm(dataset):
        embedding = generate_embedding(item["text"])
        pinecone.index(index_name).upsert(
            vectors=[(str(item["index"]), embedding, {"text": item["text"]})]  # Using index as id and storing text as metadata
        )
    return

def get_index_list():
    return pinecone.list_indexes()

def show_help():
    print("Usage: python3 embbed.py <mode>")
    print("Available modes:")
    print("  create  - Create a new index")
    print("  list    - List all existing indexes")
    print("  delete  - Delete an existing index")

def create_mode():
    index_list = get_index_list()
    
    index_name = input("Enter the name for your new index: ").strip()
    if not index_name:
        print("Error: Index name cannot be empty")
        return
    
    if index_name in index_list:
        print(f"Error: Index '{index_name}' already exists")
        return
    
    create_named_index(index_name)
    print(f"Index '{index_name}' created successfully!")
    

def list_mode():
    indexes = get_index_list()
    if not indexes:
        print("No indexes found")
    else:
        print("Available indexes:")
        for idx in indexes:
            print(f"  - {idx}")

def delete_mode():
    indexes = get_index_list()
    index_names = [idx.name for idx in indexes]
    if not indexes:
        print("No indexes to delete")
        return
    
    print("Available indexes:")
    for idx in indexes:
        print(f"  - {idx.name}")
    
    index_name = input("Enter the name of the index to delete: ").strip()
    if not index_name:
        print("Error: Index name cannot be empty")
        return
    
    if index_name not in index_names:
        print(f"Error: Index '{index_name}' does not exist")
        return
    
    confirm = input(f"Are you sure you want to delete index '{index_name}'? (y/N): ").strip().lower()
    if confirm == 'y':
        pinecone.delete_index(index_name)
        print(f"Index '{index_name}' deleted successfully!")
    else:
        print("Deletion cancelled")

def main():
    if len(sys.argv) != 2:
        show_help()
        return
    
    mode = sys.argv[1].lower()
    
    if mode == 'create':
        create_mode()
    elif mode == 'list':
        list_mode()
    elif mode == 'delete':
        delete_mode()
    else:
        print(f"Error: Unknown mode '{mode}'")
        show_help()

if __name__ == "__main__":
    main()