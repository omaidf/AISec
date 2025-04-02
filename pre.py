
#!/usr/bin/env python3
"""
AI Code Processor v2.1 - Robust Embedding Generator
Features:
- Automatic CUDA/CPU fallback
- Dynamic batch sizing
- Comprehensive error recovery
- Progress tracking
- Memory optimization
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

# Configuration
CONFIG = {
    "repo_dir": Path("scanned_repo"),
    "model_name": "microsoft/codebert-base",
    "supported_exts": {".php"},
    "chunk_size": 15,  # Lines per chunk
    "min_chunk_length": 20,
    "max_chunk_length": 512,
    "initial_batch_size": 16,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "log_file": "processing.log"
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)

def clone_repository(repo_url: str) -> None:
    """Safely clone repository with validation"""
    try:
        if CONFIG["repo_dir"].exists():
            logging.warning(f"Directory {CONFIG['repo_dir']} already exists")
            if input("Clear and reclone? (y/n): ").lower() == 'y':
                os.system(f"rm -rf {CONFIG['repo_dir']}")
            else:
                sys.exit(1)
        
        CONFIG["repo_dir"].mkdir(exist_ok=True)
        result = os.system(f"git clone --depth 1 {repo_url} {CONFIG['repo_dir']}")
        if result != 0:
            raise RuntimeError(f"Git clone failed with code {result}")
            
    except Exception as e:
        logging.error(f"Repository cloning failed: {str(e)}")
        sys.exit(1)

def find_code_files() -> list:
    """Discover code files with size and extension validation"""
    valid_files = []
    for path in CONFIG["repo_dir"].rglob("*"):
        try:
            if path.suffix not in CONFIG["supported_exts"]:
                continue
                
            if path.stat().st_size > 10 * 1024 * 1024:  # 10MB max
                logging.warning(f"Skipping large file: {path}")
                continue
                
            valid_files.append(str(path))
            
        except Exception as e:
            logging.error(f"Error accessing {path}: {str(e)}")
    
    return valid_files

def generate_code_chunks(file_path: str) -> list:
    """Generate validated code chunks from files"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [line.rstrip() for line in f if line.strip()]
        
        chunks = []
        current_chunk = []
        for line in lines:
            current_chunk.append(line)
            if len(current_chunk) >= CONFIG["chunk_size"]:
                chunk = '\n'.join(current_chunk)
                if CONFIG["min_chunk_length"] < len(chunk) < CONFIG["max_chunk_length"]:
                    chunks.append(chunk)
                current_chunk = []
        
        return chunks
        
    except Exception as e:
        logging.error(f"Chunking failed for {file_path}: {str(e)}")
        return []

def create_embeddings(code_chunks: list) -> np.ndarray:
    """Generate embeddings with dynamic memory management"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        model = AutoModel.from_pretrained(CONFIG["model_name"]).to(CONFIG["device"])
        model.eval()
    except Exception as e:
        logging.error(f"Model initialization failed: {str(e)}")
        sys.exit(1)

    embeddings = []
    batch_size = CONFIG["initial_batch_size"]
    
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=CONFIG["device"].type == 'cuda'):
        pbar = tqdm(total=len(code_chunks), desc="Processing chunks")
        idx = 0
        
        while idx < len(code_chunks):
            try:
                batch = code_chunks[idx:idx+batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=CONFIG["max_chunk_length"]
                ).to(CONFIG["device"])

                outputs = model(**inputs)
                batch_embeds = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embeds)
                
                idx += batch_size
                pbar.update(len(batch))
                del inputs, outputs, batch_embeds
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e) and batch_size > 1:
                    new_size = max(1, batch_size // 2)
                    logging.warning(f"OOM: Reducing batch size {batch_size} → {new_size}")
                    batch_size = new_size
                    continue
                raise

    pbar.close()
    return np.concatenate(embeddings) if embeddings else np.array([])

def main():
    try:
        repo_url = input("Enter repository URL: ").strip()
        if not repo_url.startswith(("http", "git@", "ssh:")):
            raise ValueError("Invalid repository URL")
        
        clone_repository(repo_url)
        code_files = find_code_files()
        
        if not code_files:
            logging.error("No valid code files found")
            sys.exit(1)
            
        all_chunks = []
        for file in tqdm(code_files, desc="Analyzing files"):
            chunks = generate_code_chunks(file)
            if chunks:
                all_chunks.extend(chunks)
        
        if not all_chunks:
            logging.error("No valid code chunks generated")
            sys.exit(1)
            
        logging.info(f"Processing {len(all_chunks)} chunks")
        embeddings = create_embeddings(all_chunks)
        
        np.save("embeddings.npy", embeddings)
        with open("code_chunks.txt", "w") as f:
            f.write('\n'.join(all_chunks))
            
        logging.info(f"Successfully processed {len(embeddings)} embeddings")
        
    except KeyboardInterrupt:
        logging.info("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Critical failure: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()