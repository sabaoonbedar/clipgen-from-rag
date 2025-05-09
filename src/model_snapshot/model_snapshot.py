from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="TheBloke/LLaVA-1.5-3B-GGUF",
#     local_dir="src/models/llava-1.5-3b-gguf",
#     local_dir_use_symlinks=False,
#     allow_patterns=["*.gguf"]  # Only download model files
# )





# snapshot_download(
#     repo_id="Salesforce/blip2-opt-2.7b",
#     local_dir="src/models/blip2-opt-2.7b",
#     local_dir_use_symlinks=False
# )
# download_models.py

# Download BLIP (vision)
snapshot_download(
    repo_id="Salesforce/blip-image-captioning-base",
    local_dir="src/models/blip-image-captioning-base",
    local_dir_use_symlinks=False
)

# Download TinyLLaMA (text)
snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir="src/models/tinyllama-1.1B-chat",
    local_dir_use_symlinks=False
)