import os
from TTS.api import TTS
from moviepy.editor import ImageClip, AudioFileClip

def generate_narrated_video(summary_file: str, image_folder: str, output_path: str = "summary_video.mp4"):
    print("[*] Reading summary text...")
    with open(summary_file, "r") as f:
        summary = f.read()

    if not summary.strip():
        raise ValueError("Summary text is empty.")

    print("[*] Generating TTS audio...")
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
    tts.tts_to_file(text=summary, file_path="summary.wav")

    print("[*] Locating image for video...")
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    if not image_files:
        raise FileNotFoundError("No images found in image folder.")

    first_image = os.path.join(image_folder, image_files[0])
    audio = AudioFileClip("summary.wav")

    print("[*] Creating video...")
    video_clip = ImageClip(first_image).set_duration(audio.duration).set_audio(audio)
    video_clip = video_clip.set_fps(24)

    print("[*] Writing video to disk...")
    video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    print(f"[✓] Video created: {output_path}")
