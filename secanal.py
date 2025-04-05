import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

# Configuration
MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"
MAX_TOKENS = 4096
DEFAULT_EXTENSIONS = ('.cs', '.py', '.js', '.java', '.php', '.go', '.rb', '.ts')
TEXT_SPLITTER_CHUNK_SIZE = 1000
TEXT_SPLITTER_OVERLAP = 200

def setup_model():
    """Initialize CodeLlama with proper padding configuration"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype="auto"
    )
    return tokenizer, model

def find_code_files(repo_path, extensions):
    """Generator to find code files in repository"""
    for root, _, files in os.walk(repo_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                yield os.path.join(root, file)

def create_repository_embeddings(repo_path, extensions):
    """Create vector store for repository code"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap=TEXT_SPLITTER_OVERLAP
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    docs = []
    for code_file in find_code_files(repo_path, extensions):
        try:
            with open(code_file, 'r', encoding='utf-8') as f:
                content = f.read()
            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                docs.append({"text": chunk, "source": code_file})
        except:
            continue
    
    if docs:
        texts = [d["text"] for d in docs]
        metadatas = [{"source": d["source"]} for d in docs]
        return FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return None

def analyze_code_file(file_path, tokenizer, model, vector_store=None):
    """Analyze a code file for vulnerabilities with optional RAG context"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

    # Add RAG context if available
    rag_context = ""
    if vector_store:
        relevant_docs = vector_store.similarity_search(code, k=2)
        rag_context = "\nRelated code context:\n" + "\n".join(
            [f"From {doc.metadata['source']}:\n{doc.page_content}" 
             for doc in relevant_docs]
        )

    prompt = f"""<s>[INST] Analyze this code for security vulnerabilities. Focus on:
- SQL injection vulnerabilities
- Cross-site scripting (XSS)
- Insecure deserialization
- CSRF vulnerabilities
- Open redirect vulnerabilities
- Authentication/authorization issues
- Hardcoded credentials
- Insecure direct object references
- Use of deprecated/insecure functions
- Weak cryptography
- Memory safety issues (buffer overflows, race conditions)
- Improper error handling
- Sensitive data exposure

Return findings in this format:
- [SEVERITY] [VULNERABILITY_TYPE] (line X): Description

Code:
{code[:5000]}{rag_context}

Potential vulnerabilities: [/INST]"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding=True  # Added padding
    ).to(model.device)

    try:
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Added attention mask
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        analysis = response.split("Potential vulnerabilities:")[-1].strip()
        return analysis
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Code Security Analyzer using CodeLlama")
    parser.add_argument("repo_path", help="Path to code repository")
    parser.add_argument("-o", "--output", default="security_report.txt",
                      help="Output file name")
    parser.add_argument("-e", "--extensions", nargs="+", default=DEFAULT_EXTENSIONS,
                      help="File extensions to analyze")
    parser.add_argument("--rag", action="store_true",
                      help="Enable RAG-based context analysis")
    args = parser.parse_args()

    print("Loading CodeLlama model...")
    tokenizer, model = setup_model()

    vector_store = None
    if args.rag:
        print("Creating repository embeddings...")
        vector_store = create_repository_embeddings(args.repo_path, args.extensions)

    print(f"Scanning files in {args.repo_path}...")
    findings = {}

    for code_file in find_code_files(args.repo_path, args.extensions):
        print(f"Analyzing {code_file}...")
        result = analyze_code_file(code_file, tokenizer, model, vector_store)
        if result and "no vulnerabilities found" not in result.lower():
            findings[code_file] = result

    # Generate report
    with open(args.output, 'w') as f:
        if not findings:
            f.write("No potential vulnerabilities found!")
        else:
            f.write("Code Security Analysis Report\n")
            f.write("=============================\n\n")
            for file, issues in findings.items():
                f.write(f"File: {file}\n")
                f.write(f"Issues:\n{issues}\n")
                f.write("\n" + "-"*50 + "\n")

    print(f"Report generated: {args.output}")

if __name__ == "__main__":
    main()