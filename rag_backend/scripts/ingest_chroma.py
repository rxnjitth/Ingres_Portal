"""
ingest_chroma.py
- Reads cleaned CSV (rows -> chunk_text + metadata)
- Generates embeddings with sentence-transformers
- Upserts into a persistent local ChromaDB collection
"""

import pandas as pd
import numpy as np
from uuid import uuid4
from tqdm import tqdm
import os
import glob

# Chroma imports
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# -------- CONFIG ----------
# Use absolute paths to avoid any confusion
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # rag_backend folder
CLEANED_DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")   # directory to store Chroma DB files
COLLECTION_NAME = "ingres_gec"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers model (dim=384)
BATCH_SIZE = 128
# --------------------------

def make_chunk_text(row, district_col="DISTRICT"):
    d = row.get(district_col, "")
    yr = row.get("assessment_year", "")
    ind = row.get("indicator", "")
    val = row.get("value_raw", "")
    # short, consistent sentence
    return f"In district {d} during {yr}, {ind} was {val}."

def load_data(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    return df

def chunk_and_meta(df):
    texts = []
    metadatas = []
    ids = []
    for _, r in df.iterrows():
        txt = make_chunk_text(r)
        meta = {
            "district": r.get("DISTRICT",""),
            "assessment_year": r.get("assessment_year",""),
            "indicator": r.get("indicator",""),
            "source_file": r.get("source_file","")
        }
        texts.append(txt)
        metadatas.append(meta)
        ids.append(str(uuid4()))
    return ids, texts, metadatas

def process_csv_file(csv_path, model, collection):
    """Process a single CSV file and add to ChromaDB collection"""
    print(f"\nProcessing file: {os.path.basename(csv_path)}")
    df = load_data(csv_path)
    print(f"Rows loaded: {len(df)}")

    print("Preparing texts + metadata...")
    ids, texts, metadatas = chunk_and_meta(df)
    if len(texts) == 0:
        print("No rows found in this file. Skipping.")
        return 0

    # encode and upsert in batches
    n = len(texts)
    print(f"Upserting {n} items in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, n, BATCH_SIZE)):
        j = min(i + BATCH_SIZE, n)
        batch_texts = texts[i:j]
        batch_ids = ids[i:j]
        batch_meta = metadatas[i:j]

        # compute embeddings (numpy array)
        embs = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        # convert to Python lists for Chroma
        embs_list = embs.tolist()

        # upsert into collection
        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_meta,
            embeddings=embs_list
        )
    
    return len(texts)

def main():
    # Find all CSV files in the cleaned data directory
    csv_files = glob.glob(os.path.join(CLEANED_DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {CLEANED_DATA_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Load the embedding model once
    print("Loading embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Initialize ChromaDB client
    print("Starting Chroma client...")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    # Process each CSV file
    total_items = 0
    for csv_file in csv_files:
        items_added = process_csv_file(csv_file, model, collection)
        total_items += items_added
        
    print(f"\nChromaDB persisted to: {PERSIST_DIR}")
    print(f"Done. Total items inserted: {total_items}")
    
    # Test a query if items were inserted
    if total_items > 0:
        test_query = "What is the groundwater recharge in Chennai 2024?"
        print(f"\nTesting query: {test_query}")
        
        # Generate embedding for the query
        query_embedding = model.encode(test_query, convert_to_numpy=True).tolist()
        
        # Search the collection
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3,
        )
        
        print("Query results (top):")
        if results and results['documents']:
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                print(f"{i+1}. {doc}")
                print(f"   District: {meta.get('district', 'N/A')}, Year: {meta.get('assessment_year', 'N/A')}")
                print(f"   Indicator: {meta.get('indicator', 'N/A')}\n")

    # With PersistentClient, data is automatically persisted

if __name__ == "__main__":
    main()
