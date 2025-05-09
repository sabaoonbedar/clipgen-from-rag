import os
import asyncio
import logging
import time
from pathlib import Path

from vision_rag_summarizer.modules.pdf_to_images import pdf_to_images
from vision_rag_summarizer.modules.ocr_extract import extract_text_with_images
from vision_rag_summarizer.modules.rag_store import build_vector_store, query_similar
from vision_rag_summarizer.modules.blip_wrapper import BlipWrapper
from vision_rag_summarizer.modules.text_llm_wrapper import TextLlmWrapper
from vision_rag_summarizer.modules.video_generator import generate_video_from_pages
from vision_rag_summarizer.utils.time_out import run_with_timeout

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def summarize_page(blip, text_llm, entry, page_number, use_rag):
    
    
    caption = blip.run(entry["image_path"])
    rag_ctx = ""
    if use_rag:
        chunks = query_similar(entry["text"], k=3)
        rag_ctx = "\n".join(chunks)
    prompt = "\n\n".join(filter(None, [
        f"This information is about: {caption}",
        f"RAG context:\n{rag_ctx}" if rag_ctx else None,
        f"The information we \n{entry['text']}",
        "In summary:"
    ]))
    summary = await run_with_timeout(text_llm.run, prompt, timeout=180)
    return f"--- Page {page_number} ---\n{summary}\n"

async def main():
    start = time.time()
    pdf_path = Path("src/data/sample2.pdf")
    img_folder = "images"
    summary_file = "summary.txt"

    # 1) PDF → images
    logging.info("[1] Converting PDF to images…")
    images = pdf_to_images(pdf_path, img_folder)
    logging.info(f"✅ {len(images)} page images created.")

    # 2) OCR
    logging.info("[2] Extracting OCR text…")
    ocr_data = extract_text_with_images(img_folder)
    logging.info(f"✅ OCR done for {len(ocr_data)} pages.")

    # 3) RAG?
    use_rag = len(ocr_data) > 1
    if use_rag:
        logging.info("[3] Building RAG store…")
        build_vector_store(ocr_data)
        logging.info("✅ RAG store ready.")

    # 4) Load models
    logging.info("[4] Loading vision+text models…")
    blip = BlipWrapper("src/models/blip-image-captioning-base")
    text_llm = TextLlmWrapper("src/models/tinyllama-1.1B-chat")

    # 5) Summarize
    logging.info("[5] Summarizing pages…")
    tasks = [
        summarize_page(blip, text_llm, entry, i+1, use_rag)
        for i, entry in enumerate(ocr_data)
    ]
    summaries = await asyncio.gather(*tasks)
    Path(summary_file).write_text("".join(summaries), encoding="utf-8")
    logging.info(f"✅ Summary written to {summary_file}")

    # 6) Generate narrated video from original page images
    logging.info("[6] Generating narrated video from pages…")
    os.makedirs("videos", exist_ok=True)
    generate_video_from_pages(
        images_folder=img_folder,
        summary_path=summary_file,
        output_path="videos/summary_video.mp4"
    )
    logging.info(f"✅ Video saved to {Path('videos/summary_video.mp4').resolve()}")

    # 7) (Optional) Remove the page images folder
    # for p in images: os.remove(p)
    # os.rmdir(img_folder)

    logging.info(f"🎉 Done in {time.time() - start:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
