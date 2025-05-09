from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageStat
import torch
from pathlib import Path
import logging
import platform

class BlipWrapper:
    def __init__(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path.resolve()}")

        system = platform.system()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif system == "Darwin" and platform.mac_ver()[0] >= "14.0":
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logging.info(f"🧠 BLIP running on device: {self.device}")

        self.processor = BlipProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

        logging.info("✅ BLIP model and processor loaded")

    def run(self, image_path: str, prompt: str = None) -> str:
        try:
            image = Image.open(image_path).convert("RGB")

            # Resize
            max_size = (768, 768)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Skip blank images
            stat = ImageStat.Stat(image)
            if sum(stat.mean) / len(stat.mean) > 250:
                logging.warning(f"⚠️ Skipping blank image: {image_path}")
                return "[Skipped blank image]"

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                gen_ids = self.model.generate(**inputs, max_new_tokens=64)
            return self.processor.decode(gen_ids[0], skip_special_tokens=True)

        except Exception as e:
            logging.error(f"❌ BLIP failed for {image_path}: {e}")
            return "[Error]"
