from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import platform
from pathlib import Path

class TextLlmWrapper:
    def __init__(self, model_path: str):
        model_dir = Path(model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir.resolve()}")

        # Device auto-select: prefer CUDA, then MPS on macOS >=14, else CPU
        system = platform.system()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif system == "Darwin" and torch.backends.mps.is_available() and platform.mac_ver()[0] >= "14.0":
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logging.info(f"🧠 Text LLM running on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            local_files_only=True
        ).to(self.device)
        self.model.eval()

        logging.info("✅ Text LLM loaded successfully")

    def run(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device.type != "mps":
                inputs = inputs.to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"❌ Text LLM failed: {e}")
            return "[Error]"