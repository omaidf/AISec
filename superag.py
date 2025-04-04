import argparse
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class CodeRAGBuilder:
    def __init__(self, repo_path, persist_dir="code_rag_db"):
        self.repo_path = Path(repo_path)
        self.persist_dir = persist_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            separators=[
                "\n\nclass ",
                "\n\nfunction ",
                "\n\ninterface ",
                "\n\ntrait ",
                "\n\nnamespace ",
                "\n\nif ",
                "\n\n<?php",
                "\n\n//",
                "\n\n",
                "\n",
                " ?>",
                ""
            ]
        )

    def _process_file(self, file_path):
        """Process a single code file"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            chunks = self.text_splitter.split_text(content)
            documents = []
            current_line = 1
            
            for chunk in chunks:
                line_count = chunk.count('\n') + 1
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": str(file_path.relative_to(self.repo_path)),
                        "start_line": current_line,
                        "end_line": current_line + line_count - 1
                    }
                ))
                current_line += line_count
            
            return documents
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def build_rag(self):
        """Process entire repository and create RAG index"""
        all_documents = []
        
        # Supported code extensions
        code_extensions = {'.php', '.py', '.js', '.java', '.go', '.rs', '.cpp'}
        
        for code_file in self.repo_path.rglob("*"):
            if code_file.suffix in code_extensions and code_file.is_file():
                all_documents.extend(self._process_file(code_file))

        # Create vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="jinaai/jina-embeddings-v2-base-code",
            model_kwargs={"device": "cpu"}
        )
        
        Chroma.from_documents(
            documents=all_documents,
            embedding=embeddings,
            persist_directory=self.persist_dir
        ).persist()
        
        print(f"Created RAG index with {len(all_documents)} chunks from "
              f"{len(list(self.repo_path.rglob('*')))} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create RAG from code repository')
    parser.add_argument('repo_path', type=str, help='Path to code repository')
    parser.add_argument('--output', type=str, default="code_rag_db",
                      help='Output directory for RAG index')
    args = parser.parse_args()

    builder = CodeRAGBuilder(args.repo_path, args.output)
    builder.build_rag()