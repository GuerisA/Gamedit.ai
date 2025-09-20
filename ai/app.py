# ai/app.py

from flask import Flask, request, jsonify  # add send_file later if you want to return the mp4 directly
from werkzeug.utils import secure_filename
from moviepy.video.io.VideoFileClip import VideoFileClip
import os, uuid, tempfile, subprocess
import numpy as np
import cv2
import threading, time
import subprocess, time, threading

def ffprobe_duration_seconds(path: str) -> float:
    """Return media duration in seconds using ffprobe, or 0.0 if unknown."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        if not out or out.upper() == "N/A":
            return 0.0
        return float(out)
    except Exception:
        return 0.0


def run_ffmpeg_with_progress(cmd, total_duration: float):
    """
    Run ffmpeg with `-progress pipe:1` and print readable percent progress.
    total_duration is seconds (used to compute %).
    """
    start = time.time()
    last_print = 0.0
    # ensure progress keys are emitted to stdout
    cmd_with_progress = cmd[:] + ["-progress", "pipe:1", "-nostats"]

    print("[ffmpeg]", " ".join(cmd_with_progress))
    proc = subprocess.Popen(
        cmd_with_progress,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            # ffmpeg progress keys look like: out_time_ms=123456789
            if line.startswith("out_time_ms="):
                t_ms = int(line.split("=", 1)[1])
                t = t_ms / 1_000_000.0  # microseconds -> seconds
                pct = 0 if total_duration <= 0 else min(99, int((t / total_duration) * 100))
                # throttle prints to ~2/sec
                now = time.time()
                if now - last_print > 0.5:
                    print(f"[progress] {t:6.1f}s ({pct:2d}%)")
                    last_print = now
            elif line.startswith("progress="):
                # progress=continue | end
                if line.endswith("end"):
                    print(f"[progress] done in {time.time()-start:0.1f}s")
            else:
                # keep other ffmpeg lines visible (bitrate, speed, etc.)
                if "speed=" in line or "bitrate=" in line:
                    print("[ffmpeg]", line)
    finally:
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed (code {ret})")

def run_with_heartbeat(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    start = time.time()
    alive = True

    def heartbeat():
        while alive:
            print(f"[progress] elapsed={int(time.time()-start)}s")
            time.sleep(2)

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()

    try:
        for line in proc.stdout:
            if line.strip():
                print("[ffmpeg]", line.strip())
    finally:
        ret = proc.wait()
        alive = False
        t.join(timeout=1)
        if ret != 0:
            raise RuntimeError("ffmpeg failed")

# ---- Optional CORS ----
try:
    from flask_cors import CORS
    USE_CORS = True
except Exception:
    USE_CORS = False

# ---------------- Config ----------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTS = {"mp4", "mov", "mkv", "webm", "m4v"}
MAX_CONTENT_LENGTH_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB

# --------------- App Init ---------------
app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_BYTES

if USE_CORS:
    CORS(app, resources={r"/*": {"origins": "*"}})

# --------------- Helpers ----------------
def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

# ---------------- Routes ----------------
@app.route("/")
def home():
    return "AI Video Processing Service is Running!"

@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/routes")
def routes():
    return {"routes": [str(r) for r in app.url_map.iter_rules()]}

@app.route("/process", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided (field name must be 'video')"}), 400

    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTS)}"}), 400

    from werkzeug.utils import secure_filename
    safe_name = secure_filename(file.filename)
    ext = safe_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    clip = None
    try:
        file.save(video_path)
        clip = VideoFileClip(video_path)
        duration = float(clip.duration or 0.0)
        fps = float(clip.fps or 0.0)
        width, height = clip.size if clip.size else (None, None)
        has_audio = bool(clip.audio)
        return jsonify({
            "message": "Video processed successfully",
            "file": unique_name,
            "original_filename": safe_name,
            "duration_sec": duration,
            "fps": fps,
            "resolution": {"width": width, "height": height},
            "has_audio": has_audio
        }), 200
    except Exception as e:
        print(f"[process_video] error: {e}")
        return jsonify({"error": "Failed to process video"}), 500
    finally:
        try:
            if clip is not None:
                clip.close()
        except Exception as close_err:
            print(f"[cleanup] clip.close warning: {close_err}")
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as rm_err:
            print(f"[cleanup] remove warning: {rm_err}")

# -------- Face-cam detection (simple heuristic) --------
def detect_facecam_bbox(video_path, sample_every_sec=0.5, min_persistence=6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        cap.release()
        return None

    step = int(max(1, round(fps * sample_every_sec)))
    detections = []
    corner_margin_x = int(W * 0.25)
    corner_margin_y = int(H * 0.35)

    frame_idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_idx % step == 0:
            ret2, frame = cap.retrieve()
            if not ret2:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            for (x, y, w, h) in faces:
                cx = x + w // 2; cy = y + h // 2
                if (cx < corner_margin_x or cx > W - corner_margin_x or
                    cy < corner_margin_y or cy > H - corner_margin_y):
                    detections.append((x, y, w, h))
        frame_idx += 1
    cap.release()
    if not detections:
        return None

    # Cluster by simple IOU / median bbox
    def iou(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        x1 = max(ax, bx); y1 = max(ay, by)
        x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
        if x1 >= x2 or y1 >= y2: return 0.0
        inter = (x2-x1)*(y2-y1)
        union = aw*ah + bw*bh - inter
        return inter/union if union>0 else 0.0

    clusters = []
    for det in detections:
        placed = False
        for c in clusters:
            if iou(det, c["bbox"]) > 0.2:
                c["items"].append(det)
                xs=[d[0] for d in c["items"]]; ys=[d[1] for d in c["items"]]
                ws=[d[2] for d in c["items"]]; hs=[d[3] for d in c["items"]]
                c["bbox"] = (int(np.median(xs)), int(np.median(ys)),
                             int(np.median(ws)), int(np.median(hs)))
                placed = True; break
        if not placed:
            clusters.append({"items":[det], "bbox":det})
    clusters.sort(key=lambda c: len(c["items"]), reverse=True)
    best = clusters[0]
    if len(best["items"]) < min_persistence:
        return None

    x, y, w, h = best["bbox"]
    pad_x = int(w * 0.15); pad_y = int(h * 0.15)
    x = max(0, x - pad_x); y = max(0, y - pad_y)
    w = min(W - x, w + 2*pad_x); h = min(H - y, h + 2*pad_y)
    return (x, y, w, h, W, H)

# -------- FFmpeg composition to 1080x1920 --------
def render_tiktok_layout(src_path, out_path, face_bbox=None, fast=False, use_nvenc=False):
    CANVAS_W, CANVAS_H = 1080, 1920
    game_crop = "crop=iw*0.6:ih*0.6:iw*0.2:ih*0.2"  # center 60%

    # Background: scale to "cover" using 'increase', then crop, then blur
    bg = (
        f"[0:v]scale={CANVAS_W}:{CANVAS_H}:force_original_aspect_ratio=increase,"
        f"crop={CANVAS_W}:{CANVAS_H},boxblur=30[bg];"
    )
    game = f"[0:v]{game_crop},scale={CANVAS_W}:{CANVAS_W}[game];"

    filter_complex = bg + game
    overlays = "[bg][game]overlay=(W-w)/2:(H-h)/2[tmp]"

    if face_bbox:
        x, y, w, h, W, H = face_bbox
        face_w = 480
        filter_complex += f"[0:v]crop={w}:{h}:{x}:{y},scale={face_w}:-1[face];"
        overlays = overlays.replace("[tmp]", "[base]")
        overlays += ";[base][face]overlay=(W-w)/2:80[tmp]"

    filter_complex += overlays

    # speed/quality knobs
    preset = "ultrafast" if fast else "veryfast"
    crf = "24" if fast else "20"

    # choose encoder
    if use_nvenc:
        vcodec = ["-c:v","h264_nvenc","-rc","vbr","-cq","23","-preset","p5","-pix_fmt","yuv420p"]
    else:
        vcodec = ["-c:v","libx264","-pix_fmt","yuv420p","-preset",preset,"-crf",crf]

    cmd = [
         "ffmpeg","-y","-hide_banner",
    "-i", src_path,
    "-filter_complex", filter_complex,
    "-map","[tmp]","-map","0:a?",
    *vcodec,
    "-c:a","aac","-b:a","128k",
    "-movflags","+faststart",
    "-progress","pipe:1","-nostats",   # <<< move here
    out_path
    ]

    # get total duration for % calc (ok to use input duration)
    total = ffprobe_duration_seconds(src_path)
    run_ffmpeg_with_progress(cmd, total_duration=total)



@app.route("/tiktok", methods=["POST"])
def tiktok():
    """
    POST a file field named 'video' and this will:
      - save it to a temp file
      - (optionally) detect a face-cam box
      - render a 1080x1920 vertical video with blurred BG + centered gameplay
      - return JSON with where the output file is

    Console will show progress like:
      [progress]  6.5s (32%)   or ffmpeg lines if percent isn't available.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided (field name must be 'video')"}), 400

    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # ---- Simple knobs (change True/False here, no query flags needed) ----
    FAST_MODE = True          # quick encode (good for testing). Set False for better quality.
    USE_NVENC = False         # set True if you have an NVIDIA GPU + NVENC available.
    DETECT_FACECAM = False    # start False (fast). Set True to enable face-cam detection.

    # temp paths
    suffix = os.path.splitext(file.filename)[1].lower()
    tmp_in  = os.path.join(tempfile.gettempdir(), f"in_{uuid.uuid4().hex}{suffix}")
    tmp_out = os.path.join(tempfile.gettempdir(), f"tiktok_{uuid.uuid4().hex}.mp4")

    try:
        print("[tiktok] upload received:", file.filename)
        file.save(tmp_in)
        try:
            size = os.path.getsize(tmp_in)
        except Exception:
            size = -1
        print(f"[tiktok] saved to {tmp_in} ({size} bytes)")

        # face-cam (optional)
        face_bbox = detect_facecam_bbox(tmp_in) if DETECT_FACECAM else None
        print(f"[tiktok] facecam_found={bool(face_bbox)}")

        # render (prints ffmpeg/progress to console)
        render_tiktok_layout(
            tmp_in,
            tmp_out,
            face_bbox,
            fast=FAST_MODE,
            use_nvenc=USE_NVENC
        )

        print(f"[tiktok] render done -> {tmp_out}")
        return jsonify({
            "message": "Rendered TikTok format",
            "fast_mode": FAST_MODE,
            "nvenc": USE_NVENC,
            "facecam_found": bool(face_bbox),
            "output_path": tmp_out.replace("\\", "/")   # file kept so you can open it
        }), 200

    except FileNotFoundError as e:
        # usually means ffmpeg not on PATH
        print(f"[tiktok] ffmpeg missing: {e}")
        return jsonify({"error": "FFmpeg not found on PATH"}), 500
    except RuntimeError as e:
        print(f"[tiktok] ffmpeg error: {e}")
        return jsonify({"error": "Failed during encoding"}), 500
    except Exception as e:
        print(f"[tiktok] error: {e}")
        return jsonify({"error": "Failed to render TikTok format"}), 500
    finally:
        # keep tmp_out so you can check the result; just delete the input
        try:
            if os.path.exists(tmp_in):
                os.remove(tmp_in)
        except Exception:
            pass


# --------------- Main (MUST BE LAST) -------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)
