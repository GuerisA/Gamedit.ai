from flask import Flask, request, jsonify
from moviepy.video.io.VideoFileClip import VideoFileClip
import os

# Initialize Flask app
app = Flask(__name__)

# Directory to save uploaded videos
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return "AI Video Processing Service is Running!"

@app.route("/process", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
    video.save(video_path)

    try:
        # Load the video using MoviePy
        clip = VideoFileClip(video_path)
        
        # Get video duration
        duration = clip.duration
        print(f"Duration: {duration} seconds")

        # Example placeholder: Analyze video for silence (add your logic here)
        # For now, just return the duration
        return jsonify({
            "message": "Video processed successfully",
            "duration": duration
        })
    except Exception as e:
        print(f"Error processing video: {e}")
        return jsonify({"error": "Failed to process video"}), 500
    finally:
        # Clean up: Close the video file and delete it
        clip.close()
        os.remove(video_path)

# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True, port=5001)
