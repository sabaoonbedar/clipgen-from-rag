# slides_manifest.py

import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def create_slide_manifest(
    summary_path: str,
    output_folder: str = "slides",
    lines_per_slide: int = 5,
    manifest_path: str = "slides/manifest.json",
    slide_size=(640, 480),
    fontsize=24,
    font_path=None  # you can point to a .ttf if you like
):
    """
    Splits summary.txt into text & PNG slides, then writes a manifest.json of PNGs.
    Returns list of PNG file paths.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load summary lines
    with open(summary_path, "r", encoding="utf-8") as f:
        lines = [l.rstrip() for l in f if l.strip()]

    # Prepare font
    if font_path:
        font = ImageFont.truetype(font_path, fontsize)
    else:
        font = ImageFont.load_default()

    png_files = []

    # Chunk into slides
    for idx in range(0, len(lines), lines_per_slide):
        chunk_lines = lines[idx : idx + lines_per_slide]
        slide_num = idx // lines_per_slide + 1
        num_str = f"{slide_num:03d}"
        
        # 1) write the .txt (optional)
        txt_path = Path(output_folder) / f"slide_{num_str}.txt"
        with open(txt_path, "w", encoding="utf-8") as sf:
            sf.write("\n".join(chunk_lines))

        # 2) render the PNG
        img = Image.new("RGB", slide_size, "black")
        draw = ImageDraw.Draw(img)
        y = 10
        for line in chunk_lines:
            draw.text((10, y), line, font=font, fill="white")
            y += fontsize + 4

        png_path = Path(output_folder) / f"slide_{num_str}.png"
        img.save(png_path)
        png_files.append(str(png_path))

    # 3) write manifest of PNGs
    manifest = {"slides": png_files}
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    return png_files
