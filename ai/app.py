# ai/app.py

from flask import Flask, request, jsonify  # add send_file later if you want to return the mp4 directly
from werkzeug.utils import secure_filename
from moviepy.video.io.VideoFileClip import VideoFileClip
import os, uuid, tempfile, subprocess
import numpy as np
import cv2
import threading, time
import subprocess, time, threading
import re
from collections import Counter
CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")

def autocrop_region(video_path: str,
                    probe_seconds: int = 8,
                    limit: float = 0.18,      # ↑ more sensitive than 0.094
                    round_val: int = 2):
    """
    Try ffmpeg cropdetect; if nothing is returned, fall back to OpenCV edge scan.
    Returns (w, h, x, y) or None.
    """
    # -------- try ffmpeg cropdetect --------
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-ss", "0", "-t", str(probe_seconds),
        "-i", video_path,
        "-vf", f"cropdetect=limit={limit}:round={round_val}:reset=0",
        "-f", "null", "-"
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
        text = (out.stderr or "") + (out.stdout or "")
        crops = CROP_RE.findall(text)
        if crops:
            w, h, x, y = map(int, Counter(crops).most_common(1)[0][0])
            if w > 0 and h > 0:
                print(f"[autocrop] ffmpeg cropdetect -> {w}x{h}+{x}+{y}")
                return (w, h, x, y)
    except Exception as e:
        print(f"[autocrop] cropdetect error: {e}")

    # -------- fallback: OpenCV edge scan --------
    try:
        import numpy as np, cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[autocrop] cv2 open failed")
            return None

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # sample a few frames over the first probe_seconds
        end_frame = min(frames - 1, int(fps * probe_seconds))
        idxs = np.linspace(0, max(0, end_frame), 5, dtype=int)

        # luma threshold (0..255). 22~28 works well for “almost black” bars.
        thr = 24

        x1 = 0; y1 = 0; x2 = W; y2 = H
        any_frame = False

        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                continue
            any_frame = True
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            col_mean = gray.mean(axis=0)
            row_mean = gray.mean(axis=1)

            # find first/last columns/rows that are above threshold
            left = int(np.argmax(col_mean > thr))
            right = int(W - np.argmax(col_mean[::-1] > thr))
            top = int(np.argmax(row_mean > thr))
            bottom = int(H - np.argmax(row_mean[::-1] > thr))

            # intersect across samples (be conservative)
            x1 = max(x1, left)
            y1 = max(y1, top)
            x2 = min(x2, right)
            y2 = min(y2, bottom)

        cap.release()

        if not any_frame:
            return None

        # safety / rounding to even
        x1 = max(0, min(x1, W - 2))
        y1 = max(0, min(y1, H - 2))
        x2 = max(x1 + 2, min(x2, W))
        y2 = max(y1 + 2, min(y2, H))
        w = ((x2 - x1) // 2) * 2
        h = ((y2 - y1) // 2) * 2

        if w > 0 and h > 0 and (w < W or h < H):
            print(f"[autocrop] opencv edge -> {w}x{h}+{x1}+{y1} (from {W}x{H})")
            return (w, h, x1, y1)

    except Exception as e:
        print(f"[autocrop] opencv error: {e}")

    print("[autocrop] no crop detected")
    return None


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


def _parse_hms_to_seconds(hms: str) -> float:
    # "00:01:23.45" -> seconds
    try:
        hh, mm, ss = hms.split(":")
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    except Exception:
        return 0.0

def run_ffmpeg_with_progress(cmd, total_duration: float):
    """
    Run ffmpeg and print readable progress. Accepts both:
      - out_time_ms=123456789
      - out_time=HH:MM:SS.micro
    Gracefully ignores N/A.
    """
    # IMPORTANT: put progress flags BEFORE the output file in the command the caller gives us.
    start = time.time()
    last_print = 0.0

    # Make sure progress flags are present; caller builds base cmd without them.
    cmd_with_progress = cmd[:]  # expect -progress pipe:1 -nostats already included by caller
    print("[ffmpeg]", " ".join(cmd_with_progress))

    proc = subprocess.Popen(
        cmd_with_progress,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue

            t_sec = None
            if line.startswith("out_time_ms="):
                val = line.split("=", 1)[1].strip()
                # guard against N/A
                if val.upper() != "N/A":
                    try:
                        t_sec = int(val) / 1_000_000.0
                    except Exception:
                        t_sec = None
            elif line.startswith("out_time="):
                # e.g. 00:00:12.34
                t_sec = _parse_hms_to_seconds(line.split("=", 1)[1].strip())

            # print throttled progress
            if t_sec is not None:
                pct = 0
                if total_duration and total_duration > 0:
                    pct = min(99, int((t_sec / total_duration) * 100))
                now = time.time()
                if now - last_print > 0.5:
                    if total_duration > 0:
                        print(f"[progress] {t_sec:6.1f}s ({pct:2d}%)")
                    else:
                        # duration unknown—show elapsed only
                        print(f"[progress] {t_sec:6.1f}s")
                    last_print = now

            # keep useful ffmpeg lines
            if "speed=" in line or "bitrate=" in line:
                print("[ffmpeg]", line)

            if line.startswith("progress=") and line.endswith("end"):
                elapsed = time.time() - start
                print(f"[progress] done in {elapsed:0.1f}s")

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
    game_crop_center = "crop=iw*0.6:ih*0.6:iw*0.2:ih*0.2"  # central 60% after any auto-crop

    # ---------- [AUTO-CROP] detect black bars and prep a [src] stream ----------
    ac = autocrop_region(src_path)  # returns (w,h,x,y) or None
    if ac:
        aw, ah, ax, ay = ac
        pre = f"[0:v]crop={aw}:{ah}:{ax}:{ay}[src];"
        src = "[src]"
        # If we detected a face box on the original frame, shift it into the cropped coords
        if face_bbox:
            x, y, w, h, W, H = face_bbox
            # translate so (0,0) is at crop top-left
            x, y = max(0, x - ax), max(0, y - ay)
            # clamp within new bounds
            x = min(x, max(0, aw - 1))
            y = min(y, max(0, ah - 1))
            w = min(w, aw - x)
            h = min(h, ah - y)
            face_bbox = (x, y, w, h, aw, ah)
    else:
        pre = ""
        src = "[0:v]"

    # --------- build filter graph using src (either cropped or original) -------
    # Background: cover 1080x1920 then blur
    bg = (
        f"{src}scale={CANVAS_W}:{CANVAS_H}:force_original_aspect_ratio=increase,"
        f"crop={CANVAS_W}:{CANVAS_H},boxblur=30[bg];"
    )

    # Foreground gameplay (square area centered) after auto-crop
    game = (
        f"{src}{game_crop_center},scale={CANVAS_W}:{CANVAS_W}[game];"
    )

    filter_complex = pre + bg + game
    overlays = "[bg][game]overlay=(W-w)/2:(H-h)/2[tmp]"

    # Face-cam overlay (optional)
    if face_bbox:
        x, y, w, h, Wn, Hn = face_bbox  # Wn/Hn correspond to the src stream dimensions (possibly auto-cropped)
        face_w = 480
        filter_complex += f"{src}crop={w}:{h}:{x}:{y},scale={face_w}:-1[face];"
        overlays = overlays.replace("[tmp]", "[base]") + ";[base][face]overlay=(W-w)/2:80[tmp]"

    filter_complex += overlays

    # speed/quality knobs
    preset = "ultrafast" if fast else "veryfast"
    crf = "24" if fast else "20"
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
        "-progress","pipe:1","-nostats",   # keep before output to see progress
        out_path
    ]

    total = ffprobe_duration_seconds(src_path)  # may be 0.0; we handle that in the parser
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
    OUTPUT_FOLDER = r"D:\Videos"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    tmp_in  = os.path.join(OUTPUT_FOLDER, f"in_{uuid.uuid4().hex}{suffix}")
    tmp_out = os.path.join(OUTPUT_FOLDER, f"tiktok_{uuid.uuid4().hex}.mp4")


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
