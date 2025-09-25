import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
collection = client.get_or_create_collection(name="ingres_gec")
model = SentenceTransformer("all-MiniLM-L6-v2")

q = "What is the recharge in Thanjavur 2024?"
q_emb = model.encode(q, convert_to_numpy=True).tolist()
res = collection.query(query_embeddings=[q_emb], n_results=5, where={"assessment_year":"2024-2025"})
print(res)
