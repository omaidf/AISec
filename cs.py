import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# Configuration
MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"
MAX_TOKENS = 4096
PHP_EXTENSIONS = ('.php', '.php3', '.php4', '.php5', '.phtml')

def setup_model():
    """Initialize CodeLlama for Apple Silicon"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype="auto"
    )
    return tokenizer, model

# ... [keep rest of the code from previous script unchanged] ...
def find_php_files(repo_path):
    """Generator to find PHP files in repository"""
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.lower().endswith(PHP_EXTENSIONS):
                yield os.path.join(root, file)

def analyze_php_file(file_path, tokenizer, model):
    """Analyze a single PHP file for vulnerabilities"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

    # Build security-focused prompt
    prompt = f"""<s>[INST] Analyze this PHP code for security vulnerabilities. Focus on:
- SQL injection vulnerabilities
- Cross-site scripting (XSS)
- File inclusion vulnerabilities (LFI/RFI)
- Insecure file upload handling
- Command injection
- Authentication bypass attempts
- Use of deprecated/unsecure functions

Return findings in this format:
- [SEVERITY] [VULNERABILITY_TYPE] (line X): Description

Code:
{code[:5000]}  # Truncate to prevent overflow

Potential vulnerabilities: [/INST]"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS
    ).to(model.device)

    try:
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the analysis portion
        analysis = response.split("Potential vulnerabilities:")[-1].strip()
        return analysis
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="PHP Security Analyzer using CodeLlama")
    parser.add_argument("repo_path", help="Path to PHP repository")
    parser.add_argument("-o", "--output", default="security_report.txt", 
                      help="Output file name")
    args = parser.parse_args()

    print("Loading CodeLlama model...")
    tokenizer, model = setup_model()

    print(f"Scanning PHP files in {args.repo_path}...")
    findings = {}

    for php_file in find_php_files(args.repo_path):
        print(f"Analyzing {php_file}...")
        result = analyze_php_file(php_file, tokenizer, model)
        if result and "no vulnerabilities found" not in result.lower():
            findings[php_file] = result

    # Generate report
    with open(args.output, 'w') as f:
        if not findings:
            f.write("No potential vulnerabilities found!")
        else:
            f.write("PHP Security Analysis Report\n")
            f.write("============================\n\n")
            for file, issues in findings.items():
                f.write(f"File: {file}\n")
                f.write(f"Issues:\n{issues}\n")
                f.write("\n" + "-"*50 + "\n")

    print(f"Report generated: {args.output}")

if __name__ == "__main__":
    import torch  # Required for BitsAndBytesConfig
    main()
