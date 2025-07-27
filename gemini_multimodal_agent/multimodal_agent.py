import os
import time
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

# --------- 1. Configure API Key Safely ---------
from google.generativeai import configure

api_key = "your_key"
if not api_key:
    raise EnvironmentError("Please set the GOOGLE_API_KEY environment variable.")

configure(api_key=api_key)

# --------- 2. Helper Functions for Uploading ---------
def safe_upload(file_path, upload_fn, get_fn, label):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{label} file not found at {file_path}")
    file_obj = upload_fn(file_path)
    
    retries = 10
    while file_obj.state.name == "PROCESSING" and retries > 0:
        time.sleep(2)
        file_obj = get_fn(file_obj.name)
        retries -= 1

    if file_obj.state.name != "ACTIVE":
        raise RuntimeError(f"{label} upload failed or timed out.")
    
    print(f"{label} uploaded successfully.")
    return file_obj

# --------- 3. Initialize Agent ---------
agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True,
)

# --------- 4. Upload Files ---------
from google.generativeai import upload_file, get_file # ✅ Make sure your agno version has this

image_file_path = "resources/sample_image.jpg"
video_file_path = "resources/sample_video.mp4"
audio_file_path = "resources/sample_audio.mp3"

image_file = safe_upload(image_file_path, upload_file, get_file, "Image")
video_file = safe_upload(video_file_path, upload_file, get_file, "Video")

# 🔍 Agno might not support audio yet, so we include it as a placeholder
if not os.path.exists(audio_file_path):
    raise FileNotFoundError("Audio file not found.")
with open(audio_file_path, "rb") as f:
    audio_bytes = f.read()

# --------- 5. Multimodal Query ---------
query = """
Combine insights from the inputs:
1. **Image**: Describe the scene and its significance.  
2. **Audio**: Extract key messages that relate to the visual.  
3. **Video**: Look at the video input and provide insights that connect with the image and audio context.  
4. **Web Search**: Find the latest updates or events linking all these topics.

Summarize the overall theme or story these inputs convey.
"""

# --------- 6. Generate Response ---------
response = agent.print_response(
    query,
    images=[image_file],
    audio=audio_bytes,           # ✅ if Agno supports raw bytes for audio input
    videos=[video_file],
    stream=True
)
