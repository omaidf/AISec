from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CodebaseAssistant:
    def __init__(self, persist_dir="roundrag"):
        self.vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=HuggingFaceEmbeddings(
                model_name="jinaai/jina-embeddings-v2-base-code",

            )
        )
        
        # Load DeepSeek Instruct locally
        self.model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
        self.revision="e5d64addd26a6a1db0f9b863abf6ee3141936807"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
    def _format_prompt(self, question, context):
        return f"""You are a senior developer analyzing code. Use this context:
        
        {context}
        
        Answer this question: {question}
        
        Provide a detailed response with code references where applicable.
        Answer:"""

    def query(self, question):
        # Retrieve relevant code context
        docs = self.vector_db.similarity_search(question, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # Format for DeepSeek
        prompt = self._format_prompt(question, context)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    assistant = CodebaseAssistant()
    
    while True:
        query = input("\nAsk about the codebase (or 'exit'): ")
        if query.lower() == "exit":
            break
            
        response = assistant.query(query)
        print(f"\nResponse:\n{response.split('Answer:')[-1].strip()}")