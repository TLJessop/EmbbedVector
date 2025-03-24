import sys
from index import (
    get_index_list,
    create_named_index,
    pinecone
)

def show_help():
    print("Usage: python3 embbed-index-cli.py <mode>")
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
