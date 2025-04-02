#!/usr/bin/env python3
"""
AI Security Trainer v3.0 - Code Pattern Discovery
Trains embedding clusters and generates analysis prompts
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import umap
import json
import hashlib

# Configuration
CONFIG = {
    "chunk_file": "code_chunks.txt",
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "min_cluster_size": 5,
    "analysis_samples": 3,
    "output_file": "code_patterns.json"
}

def load_chunks():
    """Load processed code chunks with deduplication"""
    with open(CONFIG["chunk_file"], "r") as f:
        chunks = f.read().splitlines()
    
    # Deduplicate using hashes
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
        if chunk_hash not in seen:
            seen.add(chunk_hash)
            unique_chunks.append(chunk)
    
    return unique_chunks

def train_clusters(chunks):
    """Train code pattern clusters"""
    model = SentenceTransformer(CONFIG["model_name"])
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Dimensionality reduction
    reducer = umap.UMAP(random_state=42)
    reduced_embeds = reducer.fit_transform(embeddings)
    
    # Density-based clustering
    clusterer = DBSCAN(min_samples=CONFIG["min_cluster_size"])
    clusters = clusterer.fit_predict(reduced_embeds)
    
    return chunks, embeddings, clusters

def generate_pattern_dataset(chunks, clusters):
    """Create analysis dataset from clusters"""
    pattern_db = {}
    
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_chunks = [chunks[i] for i in np.where(clusters == cluster_id)[0]]
        samples = cluster_chunks[:CONFIG["analysis_samples"]]
        
        pattern_db[f"pattern_{cluster_id}"] = {
            "frequency": len(cluster_chunks),
            "examples": samples,
            "representative_code": max(samples, key=len)
        }
    
    return pattern_db

def main():
    print("🔍 Training code pattern clusters...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} unique code chunks")
    
    chunks, embeddings, clusters = train_clusters(chunks)
    pattern_db = generate_pattern_dataset(chunks, clusters)
    
    with open(CONFIG["output_file"], "w") as f:
        json.dump(pattern_db, f, indent=2)
    
    print(f"✅ Saved {len(pattern_db)} code patterns to {CONFIG['output_file']}")

if __name__ == "__main__":
    main()