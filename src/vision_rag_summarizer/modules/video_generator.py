import os
import logging
import subprocess
import re
from pathlib import Path
from gtts import gTTS
import imageio_ffmpeg  

# bundled ffmpeg path
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

def generate_video_from_pages(
    images_folder: str = "images",
    summary_path: str = "summary.txt",
    output_path: str = "videos/summary_video.mp4"
):
    # 1) Load summaries into a dict: { page_num: text }
    raw = Path(summary_path).read_text(encoding='utf-8')
    parts = re.split(r'^--- Page (\d+) ---$', raw, flags=re.MULTILINE)
    summaries = {}
    for i in range(1, len(parts), 2):
        num = int(parts[i])
        summaries[num] = parts[i+1].strip()

    # 2) Discover all page images, sorted by the number in their filename
    image_files = sorted(
        Path(images_folder).glob("page_*.png"),
        key=lambda p: int(re.search(r'page_(\d+)\.png', p.name).group(1))
    )
    if not image_files:
        logging.error(f"No images found in {images_folder}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    segments = []

    # 3) One segment per image
    for img_path in image_files:
        page_num = int(re.search(r'page_(\d+)\.png', img_path.name).group(1))
        text = summaries.get(page_num, "")

        audio_path = f"page_{page_num}.mp3"
        segment_path = f"segment_page_{page_num}.mp4"

        # a) Generate audio (or a brief silent placeholder)
        if text:
            logging.info(f"🔊 Generating audio for page {page_num}…")
            gTTS(text).save(audio_path)
        else:
            # 0.5s of silence so the slide still appears
            subprocess.run([
                FFMPEG_EXE, '-y',
                '-f', 'lavfi', '-i',
                'anullsrc=channel_layout=stereo:sample_rate=44100',
                '-t', '0.5',
                audio_path
            ], check=True)

        # b) Create the video segment
        cmd = [
            FFMPEG_EXE, '-y',
            '-loop', '1',
            '-i', str(img_path),
            '-i', audio_path,
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            segment_path
        ]
        logging.info(f"🔨 Creating segment for page {page_num}")
        subprocess.run(cmd, check=True)
        segments.append(segment_path)

    # 4) Write the concat list
    list_file = 'segments.txt'
    with open(list_file, 'w', encoding='utf-8') as lf:
        for seg in segments:
            lf.write(f"file '{os.path.abspath(seg)}'\n")

    # 5) Stitch them all together
    logging.info(f"🔗 Concatenating {len(segments)} segments…")
    subprocess.run([
        FFMPEG_EXE, '-y',
        '-f', 'concat', '-safe', '0',
        '-i', list_file,
        '-c', 'copy',
        output_path
    ], check=True)
    logging.info(f"✅ Final video saved to {output_path}")

    # 6) Cleanup intermediates
    to_remove = segments + [list_file] + [f"page_{n}.mp3" for n in summaries]
    for p in to_remove:
        try:
            os.remove(p)
        except OSError:
            logging.warning(f"Could not delete {p}")
