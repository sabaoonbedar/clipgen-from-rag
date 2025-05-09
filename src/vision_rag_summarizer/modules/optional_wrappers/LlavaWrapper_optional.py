from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image, ImageStat
import torch
from pathlib import Path
import logging

class LlavaWrapper:
    def __init__(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path.resolve()}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"🧠 Loading processor from {model_path}")
        self.processor = LlavaProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True
        )

        logging.info("🧠 Loading LLaVA model...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        ).to(self.device)

        self.model.eval()
        logging.info("✅ Model and processor loaded")

    def run(self, image_path: str, prompt: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")

            # Resize to safe size
            max_size = (1024, 1024)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Check for blank image
            stat = ImageStat.Stat(image)
            if sum(stat.mean) / len(stat.mean) > 250:
                logging.warning(f"⚠️ Skipping nearly blank image: {image_path}")
                return "[Skipped blank image]"

            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=32)

            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        except Exception as e:
            logging.error(f"❌ LLaVA failed for {image_path}: {e}")
            return "[Error]"
