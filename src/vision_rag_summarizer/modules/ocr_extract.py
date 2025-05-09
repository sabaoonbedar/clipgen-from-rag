import os
from PIL import Image
import pytesseract
import logging

def extract_text_from_image(image_path, lang="eng"):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=lang)
        logging.info(f"📝 OCR extracted from: {image_path}")
        return text
    except Exception as e:
        logging.error(f"❌ OCR failed for {image_path}: {e}")
        return "[OCR failed]"

def extract_text_with_images(image_folder, lang="eng"):
    data = []
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        text = extract_text_from_image(image_path, lang)
        data.append({
            "image_path": image_path,
            "text": text
        })

    return data
