import os
import logging
import subprocess
import glob
from gtts import gTTS
import imageio_ffmpeg  # Bundled FFmpeg binary

# Path to bundled ffmpeg executable
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def generate_video_from_png_slides(
    slides_folder: str = 'slides',
    output_path: str = 'videos/summary_video.mp4',
    fps: int = 1
):
    """
    Assembles a narrated video from PNG slides and matching TXT content.

    Steps:
      1) Discover PNG slides in slides_folder
      2) Generate TTS audio via gTTS for each TXT slide if missing
      3) Create one MP4 segment per slide, looping image and audio until shortest ends
      4) Concatenate all segments into the final MP4
      5) Cleanup intermediate files
    """
    # 1) Discover slide images
    pattern = os.path.join(slides_folder, 'slide_*.png')
    slide_images = sorted(glob.glob(pattern))
    if not slide_images:
        logging.error(f"No slide PNGs found in '{slides_folder}'")
        return
    logging.info(f"Found {len(slide_images)} slide images.")

    # 2) Prepare output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 3) Generate audio and create segments
    temp_segments = []
    for img_path in slide_images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(slides_folder, f"{base}.txt")
        audio_path = os.path.join(slides_folder, f"{base}.mp3")
        segment_path = os.path.join(slides_folder, f"segment_{base}.mp4")

        # Generate audio if it doesn't exist
        if not os.path.exists(audio_path) and os.path.exists(txt_path):
            text = open(txt_path, 'r', encoding='utf-8').read().strip()
            logging.info(f"🔊 Generating audio for {base} via gTTS...")
            tts = gTTS(text)
            tts.save(audio_path)
            logging.info(f"🎵 Audio saved to {audio_path}")

        if not os.path.exists(audio_path):
            logging.warning(f"Skipping {base}: audio not found and no text to generate")
            continue

        # Create video segment: loop image and include audio, stop when audio ends
        cmd = [
            FFMPEG_EXE,
            '-y',            # overwrite
            '-loop', '1',    # loop image
            '-i', img_path,  # input image
            '-i', audio_path,# input audio
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            '-shortest',     # end when shortest input ends (audio)
            segment_path
        ]
        logging.info(f"🔨 Creating segment for {base}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        temp_segments.append(segment_path)

    if not temp_segments:
        logging.error("No segments created; aborting.")
        return

    # 4) Concatenate segments
    list_file = os.path.join(slides_folder, 'segments.txt')
    with open(list_file, 'w', encoding='utf-8') as lf:
        for seg in temp_segments:
            lf.write(f"file '{os.path.abspath(seg)}'\n")
    logging.info(f"🔗 Concatenating {len(temp_segments)} segments into final video...")
    cmd_concat = [
        FFMPEG_EXE,
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-c', 'copy',
        output_path
    ]
    logging.info(f"CMD: {' '.join(cmd_concat)}")
    subprocess.run(cmd_concat, check=True)
    logging.info(f"✅ Video saved to {os.path.abspath(output_path)}")

    # 5) Cleanup
    logging.info("🗑️ Cleaning up temporary files...")
    for path in temp_segments + [list_file]:
        try:
            os.remove(path)
            logging.info(f"Deleted: {path}")
        except OSError:
            pass