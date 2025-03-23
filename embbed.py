from openai import OpenAI
import pandas as pd

client = OpenAI()

def generate_embedding(text):
    embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    ).data[0].embedding
    return embedding
        
print(generate_embedding("Hello, world!"))