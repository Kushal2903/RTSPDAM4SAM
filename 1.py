import subprocess
import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image

from dam4sam_tracker import DAM4SAMTracker
from utils.visualization_utils import overlay_mask
from utils.box_selector import BoxSelector


# ============================================================
# CONFIG
# ============================================================

WIDTH = 1280
HEIGHT = 720

# Use Tiny for lowest GPU cost
MODEL = "sam21pp-T"

# Don't use 25 threads for this.
# Start with 2; increase only if decoding becomes the bottleneck.
FFMPEG_THREADS = 2

# ============================================================
# GLOBAL LATEST-FRAME BUFFER
# ============================================================

latest_frame = None
frame_lock = threading.Lock()

running = True


# ============================================================
# FFMPEG RTSP
# ============================================================

def start_ffmpeg(rtsp_url):

    frame_size = WIDTH * HEIGHT * 3

    command = [
        "ffmpeg",

        # ----------------------------------------------------
        # RTSP
        # ----------------------------------------------------
        "-rtsp_transport", "tcp",

        # ----------------------------------------------------
        # LOW LATENCY
        # ----------------------------------------------------
        "-fflags", "nobuffer+discardcorrupt",
        "-flags", "low_delay",

        # Don't wait for stream analysis
        "-analyzeduration", "0",
        "-probesize", "32",

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------
        "-i", rtsp_url,

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------
        "-an",

        # Resize directly in FFmpeg
        "-vf", f"scale={WIDTH}:{HEIGHT}",

        # Raw BGR frames for OpenCV
        "-pix_fmt", "bgr24",

        # Raw video through stdout
        "-f", "rawvideo",

        "-loglevel", "error",

        "-"
    ]

    print("\nStarting FFmpeg...")
    print("RTSP:", rtsp_url)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,

        # Small pipe buffer helps prevent excessive buffering
        bufsize=1024 * 1024
    )

    return process, frame_size


# ============================================================
# RTSP READER THREAD
# ============================================================

def read_rtsp(process, frame_size):

    global latest_frame
    global running

    print("RTSP reader started.")

    while running:

        try:

            raw = process.stdout.read(frame_size)

            if len(raw) != frame_size:

                if not running:
                    break

                continue

            frame = np.frombuffer(
                raw,
                dtype=np.uint8
            ).reshape(
                HEIGHT,
                WIDTH,
                3
            )

            # ------------------------------------------------
            # CRITICAL:
            #
            # Never queue frames.
            #
            # Always replace the old frame with the newest one.
            # ------------------------------------------------

            with frame_lock:

                latest_frame = frame

        except Exception as e:

            print("RTSP error:", e)
            break

    print("RTSP reader stopped.")


# ============================================================
# GET NEWEST FRAME
# ============================================================

def get_latest_frame():

    with frame_lock:

        if latest_frame is None:
            return None

        return latest_frame.copy()


# ============================================================
# MAIN
# ============================================================

def main():

    global running

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        print("ERROR: CUDA is not available.")
        return

    print("\n======================================")
    print("DAM4SAM RTSP - Jetson Orin")
    print("======================================")

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "Model:",
        MODEL
    )

    print(
        "Resolution:",
        f"{WIDTH}x{HEIGHT}"
    )

    # --------------------------------------------------------
    # RTSP INPUT
    # --------------------------------------------------------

    rtsp_url = input(
        "\nEnter RTSP URL: "
    ).strip()

    if not rtsp_url:

        print("ERROR: RTSP URL is empty.")
        return

    # --------------------------------------------------------
    # START FFMPEG
    # --------------------------------------------------------

    process, frame_size = start_ffmpeg(
        rtsp_url
    )

    reader = threading.Thread(
        target=read_rtsp,
        args=(process, frame_size),
        daemon=True
    )

    reader.start()

    # --------------------------------------------------------
    # WAIT FOR FIRST FRAME
    # --------------------------------------------------------

    print("\nWaiting for RTSP stream...")

    first_frame = None

    while running:

        first_frame = get_latest_frame()

        if first_frame is not None:
            break

        time.sleep(0.005)

    if first_frame is None:

        print("ERROR: No RTSP frames received.")

        running = False

        process.kill()

        return

    print("RTSP connected.")

    # --------------------------------------------------------
    # FIRST FRAME
    # --------------------------------------------------------

    first_rgb = cv2.cvtColor(
        first_frame,
        cv2.COLOR_BGR2RGB
    )

    first_pil = Image.fromarray(
        first_rgb
    )

    # --------------------------------------------------------
    # SELECT OBJECT
    # --------------------------------------------------------

    print("\nSelect the object to track.")

    box_selector = BoxSelector()

    init_box = box_selector.select_box(
        first_rgb
    )

    if not init_box:

        print("ERROR: No bounding box selected.")

        running = False
        process.kill()

        return

    # --------------------------------------------------------
    # CREATE DAM4SAM
    # --------------------------------------------------------

    print("\nLoading DAM4SAM Tiny...")

    tracker = DAM4SAMTracker(
        MODEL
    )

    print("DAM4SAM loaded.")

    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    print("\nInitializing tracker...")

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            outputs = tracker.initialize(
                first_pil,
                None,
                bbox=init_box
            )

    torch.cuda.synchronize()

    init_time = (
        time.perf_counter() - start
    ) * 1000

    print(
        f"Initialization: {init_time:.1f} ms"
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    window_name = "DAM4SAM RTSP"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps = 0.0
    fps_counter = 0

    fps_timer = time.perf_counter()

    inference_ms = 0.0

    # --------------------------------------------------------
    # TRACK LOOP
    # --------------------------------------------------------

    print("\nTracking started.")
    print("ESC = Exit")
    print("SPACE = Pause\n")

    paused = False

    while running:

        # ----------------------------------------------------
        # ALWAYS GET THE NEWEST FRAME
        # ----------------------------------------------------

        frame = get_latest_frame()

        if frame is None:

            time.sleep(0.001)

            continue

        # ----------------------------------------------------
        # PAUSE
        # ----------------------------------------------------

        if paused:

            cv2.imshow(
                window_name,
                frame
            )

            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break

            if key == 32:
                paused = False

            continue

        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

        # ----------------------------------------------------
        # DAM4SAM
        # ----------------------------------------------------

        torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = tracker.track(
                    image
                )

        torch.cuda.synchronize()

        inference_ms = (
            time.perf_counter() - start
        ) * 1000

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        pred_mask = outputs[
            "pred_mask"
        ]

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        display = rgb.copy()

        overlay_mask(
            display,
            pred_mask,
            (255, 255, 0),
            line_width=1,
            alpha=0.55
        )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        fps_counter += 1

        elapsed = (
            time.perf_counter()
            - fps_timer
        )

        if elapsed >= 1.0:

            fps = fps_counter / elapsed

            fps_counter = 0

            fps_timer = time.perf_counter()

        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Inference: {inference_ms:.1f} ms",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            "SAM2.1 Tiny FP16",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # RGB → BGR
        # ----------------------------------------------------

        display = cv2.cvtColor(
            display,
            cv2.COLOR_RGB2BGR
        )

        cv2.imshow(
            window_name,
            display
        )

        # ----------------------------------------------------
        # KEY
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == 32:
            paused = True

    # ========================================================
    # CLEANUP
    # ========================================================

    print("\nStopping...")

    running = False

    try:
        process.kill()
        process.wait(timeout=2)
    except Exception:
        pass

    cv2.destroyAllWindows()

    print("Done.")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
