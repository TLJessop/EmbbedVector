import sys
import uuid
from index import (
    get_index_list,
    create_named_index,
    pinecone
)
from embbed import generate_embedding

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
        pinecone.delete_index(index_name)
        print(f"Index '{index_name}' deleted successfully!")
    else:
        print("Deletion cancelled")

def create_embedding_mode():
    index_name = input("Enter the name of the index to create embeddings for: ").strip()
    if not index_name:
        print("Error: Index name cannot be empty")
        return
    
    index = pinecone.describe_index(index_name)
    if not index:
        print(f"Error: Index '{index_name}' does not exist")
        return
    
    upsert_data(index)
    print(f"Embeddings created successfully for index '{index_name}'")

def upsert_data(index):
    user_input = input("Enter the text to create embeddings for: ").strip()
    if not user_input:
        print("Error: Text cannot be empty")
        return
    
    embedding = generate_embedding(user_input)
 # Generate a unique ID for each vector.  This is CRUCIAL.
    vector_id = str(uuid.uuid4())

    try:
        index.upsert(
            vectors=[(vector_id, embedding, {"text": user_input})],
            namespace=namespace  # Use the provided namespace
        )
        print(f"Embeddings created successfully for text '{user_input}' with ID '{vector_id}'")
    except Exception as e:
        print(f"Error upserting data: {e}")

    print(f"Embeddings created successfully for text '{user_input}'")




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
    else:
        print(f"Error: Unknown mode '{mode}'")
        show_help()

if __name__ == "__main__":
    main()
