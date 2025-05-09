import fitz  # pymupdf
import os
import logging

def pdf_to_images(pdf_path, output_folder="images"):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    logging.info(f"📄 PDF has {len(doc)} pages")

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=300)  # high DPI for OCR
        image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
        logging.info(f"🖼️ Saved image: {image_path}")

    return image_paths
