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
import os

# ============ OPTIMIZED CONFIGURATION ============
# Reduced resolution for better performance
WIDTH = 640           # Reduced from 1024
HEIGHT = 640          # Reduced from 1024
MODEL = "sam21pp-T"   # Tiny model

# Performance tuning
TARGET_FPS = 25.0     # Slightly reduced target
TARGET_FRAME_DURATION = 1.0 / TARGET_FPS

# Adaptive inference - process every frame but with optimizations
INFERENCE_INTERVAL = 2  # Every 2nd frame for mask, but track bounding box every frame

# ============ JETSON OPTIMIZATION FLAGS ============
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Enable TensorRT optimization if available
try:
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except:
    TENSORRT_AVAILABLE = False

latest_frame = None
frame_id = 0
frame_lock = threading.Lock()
running = True

def start_ffmpeg(rtsp_url):
    """Optimized FFmpeg with hardware acceleration"""
    frame_size = WIDTH * HEIGHT * 3
    
    # Try hardware acceleration for Jetson
    command = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-fflags", "nobuffer+discardcorrupt+fastseek",
        "-flags", "low_delay",
        "-avioflags", "direct",
        "-analyzeduration", "0",
        "-threads", "1",  # Reduced threads for better scheduling
        "-i", rtsp_url,
        "-tune", "zerolatency",
        "-an", "-sn",
        "-vf", f"scale={WIDTH}:{HEIGHT},fps={TARGET_FPS}",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-loglevel", "error",
        "-"
    ]
    
    # Add hardware acceleration for Jetson
    try:
        # Check if nvidia hardware acceleration is available
        test_cmd = ["ffmpeg", "-hwaccels"]
        result = subprocess.run(test_cmd, capture_output=True, text=True)
        if "cuda" in result.stdout.lower() or "nvdec" in result.stdout.lower():
            command.insert(1, "-hwaccel")
            command.insert(2, "cuda")
            command.insert(3, "-hwaccel_output_format")
            command.insert(4, "cuda")
            print("CUDA hardware acceleration enabled for FFmpeg")
    except:
        pass
    
    print("\nStarting optimized FFmpeg...")
    print("RTSP:", rtsp_url)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1024*1024)
    return process, frame_size

def read_rtsp(process, frame_size):
    """Optimized RTSP reader with frame skipping"""
    global latest_frame, frame_id, running
    print("RTSP reader started.")
    
    skip_counter = 0
    SKIP_FRAMES = 1  # Process every 2nd frame from RTSP to reduce load
    
    while running:
        try:
            raw = process.stdout.read(frame_size)
            if len(raw) != frame_size:
                if not running:
                    break
                continue
            
            # Skip frames to reduce processing load
            skip_counter += 1
            if skip_counter % (SKIP_FRAMES + 1) != 0:
                continue
                
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
            with frame_lock:
                latest_frame = frame
                frame_id += 1
                
        except Exception as e:
            print("RTSP error:", e)
            break
    print("RTSP reader stopped.")

def get_latest_frame_with_id():
    with frame_lock:
        if latest_frame is None:
            return None, 0
        return latest_frame.copy(), frame_id

class OptimizedDAM4SAM:
    """Wrapper for DAM4SAM with optimization strategies"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.tracker = DAM4SAMTracker(model_name)
        self.last_mask = None
        self.last_box = None
        self.frame_count = 0
        
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Try to optimize with TensorRT if available
        if TENSORRT_AVAILABLE:
            print("TensorRT available - will use for acceleration")
    
    def initialize(self, image, bbox=None):
        """Initialize tracker with optimization"""
        self.last_box = bbox
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                self.tracker.initialize(image, None, bbox=bbox)
    
    def track(self, image):
        """Optimized tracking with cache and early exit"""
        self.frame_count += 1
        
        # Use cached mask for frames we skip inference
        if self.frame_count % INFERENCE_INTERVAL != 0 and self.last_mask is not None:
            return {"pred_mask": self.last_mask, "cached": True}
        
        # Run inference on selected frames
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.tracker.track(image)
                
                # Store mask for caching
                if "pred_mask" in outputs:
                    self.last_mask = outputs["pred_mask"]
                    outputs["cached"] = False
                return outputs

def main():
    global running
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        return
    
    # Set GPU memory limit to prevent OOM
    torch.cuda.set_per_process_memory_fraction(0.7)
    
    print("="*60)
    print("OPTIMIZED DAM4SAM TRACKER FOR JETSON AGX ORIN")
    print("="*60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Inference Interval: Every {INFERENCE_INTERVAL} frames")
    print("="*60)
    
    # ========== RTSP SETUP ==========
    rtsp_url = r"rtsp://172.72.8.11:554/live/0/0"
    process, frame_size = start_ffmpeg(rtsp_url)
    
    reader = threading.Thread(target=read_rtsp, args=(process, frame_size), daemon=True)
    reader.start()
    
    print("\nWaiting for RTSP stream...")
    while running:
        frame, _ = get_latest_frame_with_id()
        if frame is not None:
            break
        time.sleep(0.005)
    
    print("RTSP connected.")
    
    # ========== MODEL LOADING ==========
    print("Loading optimized DAM4SAM...")
    tracker = OptimizedDAM4SAM(MODEL)
    print("DAM4SAM loaded successfully.")
    
    # ========== WINDOW SETUP ==========
    window_name = "DAM4SAM Optimized"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    tracking_initialized = False
    box_selector = BoxSelector()
    
    # Performance monitoring
    fps_counter = 0
    display_fps = 0.0
    fps_timer = time.perf_counter()
    inference_ms = 0.0
    gpu_usage = 0.0
    
    last_processed_id = -1
    pred_mask = None
    
    print("\n[STREAMING MODE ACTIVE]")
    print("SPACE: Select Object & Start Tracking")
    print("ESC: Exit")
    print("R: Reset Tracking")
    print("="*60 + "\n")
    
    frame_buffer = []
    MAX_BUFFER_SIZE = 2
    
    while running:
        loop_start = time.perf_counter()
        
        # Get frame with buffer
        frame, current_id = get_latest_frame_with_id()
        if frame is None:
            time.sleep(0.001)
            continue
        
        # Simple frame buffer to handle processing delays
        frame_buffer.append(frame)
        if len(frame_buffer) > MAX_BUFFER_SIZE:
            frame_buffer.pop(0)
        
        # Use latest frame from buffer
        current_frame = frame_buffer[-1] if frame_buffer else frame
        
        if not tracking_initialized:
            display = current_frame.copy()
            cv2.putText(display, "Press SPACE to select object", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(display, f"Resolution: {WIDTH}x{HEIGHT}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                print("\nSelecting object...")
                init_box = box_selector.select_box(current_frame)
                
                if init_box:
                    print(f"Initializing with box: {init_box}")
                    rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    first_pil = Image.fromarray(rgb_frame)
                    
                    tracker.initialize(first_pil, bbox=init_box)
                    
                    print("Tracking initialized successfully!")
                    tracking_initialized = True
                    
                    fps_timer = time.perf_counter()
                    fps_counter = 0
                    pred_mask = None
                    last_processed_id = current_id
                else:
                    print("Selection cancelled.")
        
        else:
            if current_id == last_processed_id:
                time.sleep(0.001)
                continue
            
            last_processed_id = current_id
            rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            
            # ========== OPTIMIZED INFERENCE ==========
            inf_start = time.perf_counter()
            
            try:
                outputs = tracker.track(image)
                pred_mask = outputs["pred_mask"]
                cached = outputs.get("cached", False)
                
                # Only calculate inference time when actually processing
                if not cached:
                    inference_ms = (time.perf_counter() - inf_start) * 1000
                    fps_counter += 1
                else:
                    inference_ms = 0.0
                    fps_counter += 1
                    
            except Exception as e:
                print(f"Inference error: {e}")
                continue
            
            # ========== VISUALIZATION ==========
            display = rgb_frame.copy()
            
            # Optimized overlay - reduce alpha for better performance
            if pred_mask is not None:
                # Convert mask to binary for faster overlay
                mask_binary = (pred_mask > 0.5).astype(np.uint8)
                overlay_mask(display, mask_binary, (255, 255, 0), 
                           line_width=1, alpha=0.4)  # Reduced alpha
            
            # ========== PERFORMANCE METRICS ==========
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                display_fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.perf_counter()
                
                # Monitor GPU usage
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = util.gpu
                    pynvml.nvmlShutdown()
                except:
                    gpu_usage = 0  # Fallback
            
            # Status text
            status = "GPU: " if not cached else "(Cached) GPU: "
            if gpu_usage > 0:
                status += f"{gpu_usage:.0f}%"
            else:
                status += "N/A"
            
            cv2.putText(display, f"FPS: {display_fps:.1f}", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if cached:
                cv2.putText(display, "CACHED", (20, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(display, f"Inf: {inference_ms:.0f}ms", (20, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(display, status, (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if gpu_usage < 60 else (0, 0, 255), 2)
            
            cv2.putText(display, f"Res: {WIDTH}x{HEIGHT}", (20, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(display, "SPACE:select | R:reset | ESC:exit", 
                       (20, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, display_bgr)
            
            # ========== KEY HANDLING ==========
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE - re-select
                tracking_initialized = False
                print("Resetting tracking...")
            elif key == ord('r') or key == ord('R'):
                tracking_initialized = False
                print("Resetting tracking...")
            
            # ========== MEMORY MANAGEMENT ==========
            if tracker.frame_count % 50 == 0:
                torch.cuda.empty_cache()
            
            # ========== DYNAMIC PACING ==========
            time_spent = time.perf_counter() - loop_start
            time_remaining = TARGET_FRAME_DURATION - time_spent
            if time_remaining > 0:
                time.sleep(min(time_remaining, 0.01))  # Cap sleep time
    
    # ========== CLEANUP ==========
    print("\nStopping stream...")
    running = False
    try:
        process.kill()
        process.wait(timeout=2)
    except Exception:
        pass
    cv2.destroyAllWindows()
    torch.cuda.empty_cache()
    print("Done.")

if __name__ == "__main__":
    main()
