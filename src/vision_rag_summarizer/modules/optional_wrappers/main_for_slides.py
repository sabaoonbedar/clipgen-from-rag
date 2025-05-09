import os
import asyncio
import logging
import time
from pathlib import Path

from vision_rag_summarizer.modules.slides_manifest import create_slide_manifest
from vision_rag_summarizer.modules.pdf_to_images import pdf_to_images
from vision_rag_summarizer.modules.ocr_extract import extract_text_with_images
from vision_rag_summarizer.modules.rag_store import build_vector_store, query_similar
from vision_rag_summarizer.modules.blip_wrapper import BlipWrapper
from vision_rag_summarizer.modules.text_llm_wrapper import TextLlmWrapper
from vision_rag_summarizer.modules.video_generator import generate_video_from_png_slides
from vision_rag_summarizer.utils.time_out import run_with_timeout

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def summarize_page(blip, text_llm, entry, page_number, use_rag):
    caption = blip.run(entry["image_path"])
    rag_ctx = ""
    if use_rag:
        chunks = query_similar(entry["text"], k=3)
        rag_ctx = "\n".join(chunks)

    prompt_parts = [
        f"Visual description: {caption}",
        f"RAG context:\n{rag_ctx}" if rag_ctx else None,
        f"OCR text:\n{entry['text']}",
        "Summarize this for a video script:"
    ]
    prompt = "\n\n".join(p for p in prompt_parts if p)
    summary = await run_with_timeout(text_llm.run, prompt, timeout=180)
    return f"--- Page {page_number} ---\n{summary}\n"

async def main():
    start = time.time()
    pdf_path = Path("src/data/sample.pdf")
    img_folder = "images"
    summary_file = "summary.txt"

    # 1) PDF → images
    logging.info("[1] Converting PDF to images…")
    images = pdf_to_images(pdf_path, img_folder)
    logging.info(f"✅ {len(images)} images created.")

    # 2) OCR extraction
    logging.info("[2] Extracting OCR text…")
    ocr_data = extract_text_with_images(img_folder)
    logging.info(f"✅ OCR done for {len(ocr_data)} pages.")

    # 3) Optional RAG
    use_rag = len(ocr_data) > 1
    if use_rag:
        logging.info("[3] Building RAG store…")
        build_vector_store(ocr_data)
        logging.info("✅ RAG store ready.")

    # 4) Load models
    logging.info("[4] Loading models…")
    blip = BlipWrapper("src/models/blip-image-captioning-base")
    text_llm = TextLlmWrapper("src/models/tinyllama-1.1B-chat")
    logging.info("✅ Models loaded.")

    # 5) Summarize pages
    logging.info("[5] Summarizing pages…")
    tasks = [
        summarize_page(blip, text_llm, entry, i+1, use_rag)
        for i, entry in enumerate(ocr_data)
    ]
    summaries = await asyncio.gather(*tasks)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.writelines(summaries)
    logging.info(f"✅ Summary written to {summary_file}")

    # 6) Clean up images
    logging.info("[6] Deleting temporary images…")
    for img_path in images:
        os.remove(img_path)
    os.rmdir(img_folder)
    logging.info("✅ Cleaned up images.")

    # 7) Create slide PNGs & manifest
    logging.info("[7] Creating slides…")
    png_slides = create_slide_manifest(
        summary_file,
        output_folder="slides",
        lines_per_slide=5,
        manifest_path="slides/manifest.json",
        slide_size=(640, 480),
        fontsize=24
    )
    logging.info(f"✅ Created {len(png_slides)} PNG slides.")

    # 8) Verify PNG slides exist
    slide_folder = Path("slides")
    slide_images = sorted(slide_folder.glob("slide_*.png"))
    if not slide_images:
        raise FileNotFoundError(f"No slide PNGs found in {slide_folder.resolve()}")
    logging.info(f"✅ Found {len(slide_images)} slide images:")
    for img in slide_images:
        logging.info(f"   • {img.name}")

    # 9) Generate video
    logging.info("[8] Generating video…")
    os.makedirs("videos", exist_ok=True)
    video_path = "videos/summary_video.mp4"
    generate_video_from_png_slides(
        slides_folder="slides",
        output_path=video_path
    )
    logging.info(f"✅ Video saved to {Path(video_path).resolve()}")

    logging.info(f"🎉 Done in {time.time() - start:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
