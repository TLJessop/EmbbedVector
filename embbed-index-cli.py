import sys
import uuid
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import pinecone
from index import (
    get_index_list,
    create_named_index,
)
from embbed import generate_embedding

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  # Load environment variables from .env file

def show_help():
    print("Usage: python3 embbed-index-cli.py <mode>")
    print("Available modes:")
    print("  create-index  - Create a new index")
    print("  list-indexes    - List all existing indexes")
    print("  delete-index  - Delete an existing index")
    print("  create-embedding  - Create embeddings for an existing index")

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
        pc.delete_index(index_name)
        print(f"Index '{index_name}' deleted successfully!")
    else:
        print("Deletion cancelled")

def create_embedding_mode():
    index_name = input("Enter the name of the index to create embeddings for: ").strip()
    if not index_name:
        print("Error: Index name cannot be empty")
        return
    
    try:
        index_description = pc.describe_index(index_name)
        index = pinecone.Index(
            name=index_name,
            host=index_description.host,
            api_key=os.getenv("PINECONE_API_KEY")
        )
        upsert_data(index)
        print(f"Embeddings created successfully for index '{index_name}'")
    except Exception as e:
        print(f"Error: Could not access index '{index_name}': {e}")
        return

def query_mode():
    index_name = input("Enter the name of the index to query: ").strip()
    if not index_name:
        print("Error: Index name cannot be empty")
        return
    
    try:
        # Get index description first to verify it exists and get the host
        index_description = pc.describe_index(index_name)
        index = pinecone.Index(
            name=index_name,
            host=index_description.host,
            api_key=os.getenv("PINECONE_API_KEY")
        )
    except Exception as e:
        print(f"Error: Could not access index '{index_name}': {e}")
        return
    
    query = input("Enter the query: ").strip()
    if not query:
        print("Error: Query cannot be empty")
        return
    print("Generating embedding for query...")
    embedding = generate_embedding(query)

    
    results = index.query(
        vector=embedding,
        top_k=3,
        include_metadata=True
    )
    
    print("Query results:")
    if not results.matches:
        print("  No matches found")
        return
        
    for match in results.matches:
        print(f"  - Score: {match.score:.4f}")
        if hasattr(match, 'metadata') and match.metadata:
            print(f"    Text: {match.metadata.get('text', 'No text available')}")

def upsert_data(index):
    user_input = input("Enter the text to create embeddings for: ").strip()
    if not user_input:
        print("Error: Text cannot be empty")
        return
    
    embedding = generate_embedding(user_input)
    # Generate a unique ID for each vector
    vector_id = str(uuid.uuid4())

    try:
        index.upsert([
            (vector_id, embedding, {"text": user_input})
        ])
        print(f"Embeddings created successfully for text '{user_input}' with ID '{vector_id}'")
    except Exception as e:
        print(f"Error upserting data: {e}")




def main():
    if len(sys.argv) != 2:
        show_help()
        return
    
    mode = sys.argv[1].lower()
    
    if mode == 'create-index':
        create_mode()
    elif mode == 'list-indexes':
        list_mode()
    elif mode == 'delete-index':
        delete_mode()
    elif mode == 'create-embedding':
        create_embedding_mode()
    elif mode == 'query-index':
        query_mode()
    else:
        print(f"Error: Unknown mode '{mode}'")
        show_help()

if __name__ == "__main__":
    main()
