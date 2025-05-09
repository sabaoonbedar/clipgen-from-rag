from sentence_transformers import SentenceTransformer
import chromadb

# Initialize Chroma and embedder
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("ocr_chunks")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_vector_store(ocr_data):
    texts = [entry["text"] for entry in ocr_data]
    embeddings = embedder.encode(texts).tolist()
    ids = [str(i) for i in range(len(texts))]
    collection.add(documents=texts, embeddings=embeddings, ids=ids)

def query_similar(text, k=3):
    query_emb = embedder.encode([text])[0].tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=k)
    return results["documents"][0]
