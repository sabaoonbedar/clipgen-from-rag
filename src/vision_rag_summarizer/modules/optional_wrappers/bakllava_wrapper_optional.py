from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image, ImageStat
import torch
from pathlib import Path
import logging
import platform

class BakLlavaWrapper:
    def __init__(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path.resolve()}")

        # Device autodetect
        system = platform.system()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif system == "Darwin" and platform.mac_ver()[0] >= "14.0":
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logging.info(f"🧠 Loading BakLLaVA from {model_path} on {self.device}")

        # Load processor & model locally
        self.processor = LlavaProcessor.from_pretrained(
            model_path, local_files_only=True, use_fast=True
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        ).to(self.device)
        self.model.eval()

        logging.info("✅ BakLLaVA model and processor loaded")

    def run(self, image_path: str, prompt: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")

            # Resize to safe size for ViT patching
            max_size = (768, 768)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Skip blank images
            stat = ImageStat.Stat(image)
            if sum(stat.mean) / len(stat.mean) > 250:
                logging.warning(f"⚠️ Skipping blank image: {image_path}")
                return "[Skipped blank image]"

            # Build inputs: image + prompt
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)

            # Generate a longer summary
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,   # optional to reduce repetition
                )

            # Decode and return
            return self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        except Exception as e:
            logging.error(f"❌ BakLLaVA failed for {image_path}: {e}")
            return "[Error]"
