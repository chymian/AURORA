import ollama
import logging
import chromadb

def generate_embedding(text, embeddings_model, collection, collection_size):
    try:
        response = ollama.embeddings(model=embeddings_model, prompt=text)
        embedding = response["embedding"]
        if not embedding:
            raise ValueError("Generated embedding is empty.")
        collection.add(
            ids=[str(collection_size)],
            embeddings=[embedding],
            documents=[text]
        )
        return embedding
    except Exception as e:
        raise Exception(f"Error generating embedding: {e}")

def add_to_memory(text, embeddings_model, collection, collection_size):
    try:
        embedding = generate_embedding(text, embeddings_model, collection, collection_size)
        collection_size += 1
        return embedding
    except Exception as e:
        raise Exception(f"Error adding to memory: {e}")

def retrieve_relevant_memory(prompt_embedding, collection):
    try:
        results = collection.query(query_embeddings=[prompt_embedding])
        return [doc for doc in results['documents'][0]]
    except Exception as e:
        raise Exception(f"Error retrieving relevant memory: {e}")



def setup_logging():
    """
    Set up logging configuration for the application.
    """
    try:
        logging.basicConfig(
            filename='lobes_log.txt',
            level=logging.INFO,
            format='%(asctime)s %(message)s'
        )
        print("Logging setup completed.")
    except Exception as e:
        print(f"Error setting up logging: {e}")

def setup_embedding_collection():
    """
    Set up the embedding collection using ChromaDB.

    Returns:
        tuple: A tuple containing the created collection and its initial size (0).
    """
    print("Setting up embedding collection.")
    try:
        client = chromadb.Client()
        collection = client.create_collection(name="convo_memory")
        print("Embedding collection setup completed.")
        return collection, 0
    except Exception as e:
        print(f"Error setting up embedding collection: {e}")
        return None, 0
