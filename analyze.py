
#!/usr/bin/env python3
"""
AI Security Analyst v5.0 - Enterprise-Grade Implementation
"""

import os
import sys
import json
import torch
import hashlib
from functools import partial
from typing import Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging
)
from tqdm import tqdm
from huggingface_hub import snapshot_download, login

# Suppress unnecessary warnings
logging.set_verbosity_error()

# Configuration
CONFIG = {
    "model_name": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "revision": "e5d64addd26a6a1db0f9b863abf6ee3141936807",
    "pattern_file": "code_patterns.json",
    "output_file": "security_findings.md",
    "cache_dir": "model_cache",
    "quant_config": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    ),
    "trust_remote_code": True,
    "max_new_tokens": 1024,
    "temperature": 0.2,
    "max_code_length": 2048,
    "min_pattern_size": 3
}

PROMPT_TEMPLATE = """Analyze this code for security vulnerabilities:

{code}

Provide analysis in this format:
1. Potential Risks
2. Recommended Fixes
3. Severity Assessment (Critical/High/Medium/Low)"""

class CustomTqdm(tqdm):
    """Thread-safe progress bar with Hugging Face Hub compatibility"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('unit', 'B')
        kwargs.setdefault('unit_scale', True)
        kwargs.setdefault('leave', False)
        super().__init__(*args, **kwargs)

    @property
    def lock(self):
        return self.get_lock()

def check_dependencies() -> None:
    """Verify and install required packages"""
    required = {
        'accelerate': '>=0.29.0',
        'bitsandbytes': '>=0.43.0',
        'transformers': '>=4.40.0',
        'tqdm': '>=4.66.0',
        'huggingface_hub': '>=0.22.0'
    }
    
    missing = []
    for pkg, version in required.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(f"{pkg}{version}")
    
    if missing:
        print("ERROR: Missing required dependencies")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

class SecurityAnalyst:
    """Core analysis engine with fault tolerance"""
    
    def __init__(self) -> None:
        check_dependencies()
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize model with robust error recovery"""
        try:
            if not self._check_hf_auth():
                login()
                
            self._download_model()
            self._load_tokenizer()
            self._load_model()
            self._configure_tokenizer()
            
        except Exception as e:
            print(f"🚨 Initialization failed: {str(e)}")
            self._print_troubleshooting()
            sys.exit(1)

    def _check_hf_auth(self) -> bool:
        """Check valid Hugging Face authentication"""
        try:
            from huggingface_hub import whoami
            return whoami()["name"] != ""
        except Exception:
            return False

    def _download_model(self) -> None:
        """Download model files with resume support"""
        os.makedirs(CONFIG["cache_dir"], exist_ok=True)
        
        snapshot_download(
            repo_id=CONFIG["model_name"],
            revision=CONFIG["revision"],
            allow_patterns=["*.safetensors", "*.json", "*.model"],
            local_dir=CONFIG["cache_dir"],
            resume_download=True,
            token=True,
            tqdm_class=partial(
                CustomTqdm,
                desc="Downloading model",
                unit_scale=True,
                mininterval=1
            )
        )

    def _load_tokenizer(self) -> None:
        """Load tokenizer with validation"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["cache_dir"],
            trust_remote_code=CONFIG["trust_remote_code"]
        )
        
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self) -> None:
        """Load model with quantized weights"""
        self.model = AutoModelForCausalLM.from_pretrained(
            CONFIG["cache_dir"],
            quantization_config=CONFIG["quant_config"],
            device_map="auto",
            trust_remote_code=CONFIG["trust_remote_code"]
        )
        self.model.eval()

    def _configure_tokenizer(self) -> None:
        """Ensure tokenizer safety settings"""
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def analyze_code(self, code: str) -> str:
        """Analyze code snippet with error boundaries"""
        try:
            if len(code) > CONFIG["max_code_length"]:
                code = code[:CONFIG["max_code_length"]] + "\n// ... (truncated)"
            
            return self._safe_generate(code)
        except Exception as e:
            print(f"⚠️ Analysis error: {str(e)}")
            return "Analysis failed for this pattern"

    def _safe_generate(self, code: str) -> str:
        """Protected generation with device management"""
        prompt = PROMPT_TEMPLATE.format(code=code)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"],
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _print_troubleshooting(self) -> None:
        """Display detailed troubleshooting guide"""
        print("\nTROUBLESHOOTING GUIDE:")
        print("1. Authentication: Run 'huggingface-cli login'")
        print("2. Disk Space: Ensure >50GB available")
        print("3. Network: Check internet connection stability")
        print("4. Cache: Try 'rm -rf model_cache/'")
        print("5. Dependencies: pip install -U transformers accelerate bitsandbytes")
        print("6 Hardware: Use NVIDIA GPU with >16GB VRAM for best results")

def validate_patterns(patterns: Dict[str, Any]) -> None:
    """Validate pattern file structure"""
    if not isinstance(patterns, dict):
        raise ValueError("Pattern file must be a JSON object")
    
    for pid, data in patterns.items():
        if "representative_code" not in data:
            raise ValueError(f"Pattern {pid} missing 'representative_code'")
        if "examples" not in data or len(data["examples"]) < 1:
            raise ValueError(f"Pattern {pid} needs at least one example")

def generate_report(findings: Dict[str, Any]) -> None:
    """Generate comprehensive security report"""
    try:
        with open(CONFIG["output_file"], "w") as f:
            f.write("# Automated Security Audit Report\n\n")
            
            for idx, (pattern_id, data) in enumerate(findings.items(), 1):
                f.write(f"## Finding {idx}: {pattern_id}\n")
                f.write(f"**Frequency**: {data.get('frequency', 'Unknown')} occurrences\n")
                f.write(f"**Severity**: {data['analysis'].split('Severity Assessment')[-1].strip()}\n\n")
                f.write("### Example Code\n```\n")
                f.write(data["example"].strip() + "\n")
                f.write("```\n\n")
                f.write("### Risks\n- " + "\n- ".join(
                    data["analysis"].split("Potential Risks")[-1]
                        .split("Recommended Fixes")[0].strip().split("\n")
                ) + "\n\n")
                f.write("### Recommendations\n- " + "\n- ".join(
                    data["analysis"].split("Recommended Fixes")[-1]
                        .split("Severity Assessment")[0].strip().split("\n")
                ) + "\n\n")
                f.write("---\n")
                
        print(f"\n✅ Report generated: {os.path.abspath(CONFIG['output_file'])}")
        
    except Exception as e:
        print(f"🚨 Critical report error: {str(e)}")
        sys.exit(1)

def main() -> None:
    """Main execution pipeline"""
    print("🔒 Initializing Security Analysis Pipeline")
    
    try:
        # Load and validate patterns
        with open(CONFIG["pattern_file"], "r") as f:
            patterns = json.load(f)
        validate_patterns(patterns)
        
        # Initialize analysis engine
        analyst = SecurityAnalyst()
        findings = {}
        
        # Process patterns
        progress = tqdm(
            patterns.items(),
            desc="Analyzing Code Patterns",
            unit="pattern"
        )
        
        for pattern_id, data in progress:
            try:
                analysis = analyst.analyze_code(data["representative_code"])
                findings[pattern_id] = {
                    "frequency": data.get("frequency", "Unknown"),
                    "example": data["examples"][0],
                    "analysis": analysis
                }
            except Exception as e:
                print(f"⚠️ Skipped {pattern_id}: {str(e)}")
                continue

        # Generate final report
        generate_report(findings)
        
    except Exception as e:
        print(f"🚨 Fatal pipeline error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()