import os
import glob
import time
import cv2
import numpy as np
import torch
import gc
from pipeline_manager import PipelineManager

# ── CẤU HÌNH ──────────────────────────────────────────────────────────────────
FOLDER_PATH  = "datasets/phenikaa_dataset/CAM_P_F"
IMG_FORMATS  = ("*.jpg", "*.jpeg", "*.png", "*.webp")

RESOLUTIONS_TO_TEST = [
    (518, 294), (700, 392), (840, 476), (924, 518), (1120, 630)
]

K = np.array([[1837.189936, 0, 912.272553], [0, 1844.888039, 728.975910], [0, 0, 1]], dtype=np.float64)
E_v2c = np.array([[0.050593, -0.998719, 0.000873, -0.067766], [4.4e-05, -0.000872, -1.000000, 1.880442], [0.998719, 0.050593, 0, -1.305326], [0, 0, 0, 1]], dtype=np.float64)

def run_sweep_benchmark():
    image_paths = []
    for fmt in IMG_FORMATS:
        image_paths.extend(glob.glob(os.path.join(FOLDER_PATH, fmt)))
    
    if not image_paths:
        print(f"❌ Thư mục trống: {FOLDER_PATH}")
        return

    image_paths.sort() 
    results_summary = []

    for res in RESOLUTIONS_TO_TEST:
        W, H = res
        print(f"\n🔄 TESTING RESOLUTION: {W}x{H} " + "="*50)
        
        pipeline = PipelineManager(
            seg_weight="weights/best_yolov8l.pt",
            depth_weight="weights/depth_anything_v2_metric_vkitti_vits.pth",
            K=K, E_v2c=E_v2c, device="cuda:0", target_size=(W, H)
        )

        warmup_img = cv2.imread(image_paths[0])
        for _ in range(5):
            pipeline.run(warmup_img)
        torch.cuda.synchronize()

        # Mở rộng dictionary để chứa TOÀN BỘ CÁC BƯỚC
        metrics = {
            "e2e": [], 
            "s1_resize": [], 
            "s2a_yolo": [], 
            "s2b_mask": [], 
            "s3_depth": [], 
            "s4_geo": [], 
            "s5_post": [], 
            "objs": 0
        }
        
        test_count = len(image_paths)
        
        for i in range(test_count):
            img = cv2.imread(image_paths[i])
            if img is None: continue
            
            torch.cuda.synchronize()
            detections, p = pipeline.run(img)
            
            # Lưu lại tất cả dữ liệu
            metrics["e2e"].append(p.get('TOTAL_LATENCY', 0))
            metrics["s1_resize"].append(p.get('Step 1: Resize & Calib', 0))
            metrics["s2a_yolo"].append(p.get('Step 2a: YOLO Inference', 0))
            metrics["s2b_mask"].append(p.get('Step 2b: Mask Build (CPU)', 0))
            metrics["s3_depth"].append(p.get('Step 3: Depth Inference', 0))
            metrics["s4_geo"].append(p.get('Step 4: Geometry (CPU/NP)', 0))
            metrics["s5_post"].append(p.get('Step 5: Post-Process (CPU)', 0))
            metrics["objs"] += len(detections)

            if i % 10 == 0:
                print(f"\r  ⏳ Frame {i}/{test_count} | Latency: {p.get('TOTAL_LATENCY',0):.2f}ms", end="")

        avg_e2e = np.mean(metrics["e2e"])
        results_summary.append({
            "Res": f"{W}x{H}",
            "FPS": 1000 / avg_e2e if avg_e2e > 0 else 0,
            "E2E": avg_e2e,
            "S1": np.mean(metrics["s1_resize"]),
            "S2a": np.mean(metrics["s2a_yolo"]),
            "S2b": np.mean(metrics["s2b_mask"]),
            "S3": np.mean(metrics["s3_depth"]),
            "S4": np.mean(metrics["s4_geo"]),
            "S5": np.mean(metrics["s5_post"]),
            "Objs": metrics["objs"]
        })

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

    # ── IN BẢNG TỔNG KẾT CHI TIẾT ──
    print("\n\n" + "★"*130)
    print(f"{'RTX 5090 FULL PIPELINE PROFILING SWEEP':^130}")
    print("★"*130)
    
    # Header format cực rộng để chứa đủ các cột
    header = f"| {'Res':<10} | {'FPS':<7} | {'E2E':<7} | {'S1 (ms)':<7} | {'S2a (ms)':<8} | {'S2b (ms)':<8} | {'S3 (ms)':<8} | {'S4 (ms)':<7} | {'S5 (ms)':<7} | {'Objs':<5} |"
    print(header)
    print("-" * len(header))
    
    for r in results_summary:
        line = (f"| {r['Res']:<10} | {r['FPS']:<7.2f} | {r['E2E']:<7.2f} | "
                f"{r['S1']:<7.2f} | {r['S2a']:<8.2f} | {r['S2b']:<8.2f} | "
                f"{r['S3']:<8.2f} | {r['S4']:<7.2f} | {r['S5']:<7.2f} | {r['Objs']:<5} |")
        print(line)
        
    print("-" * len(header))
    print("\n💡 CHÚ THÍCH CÁC BƯỚC (STEPS):")
    print("  - S1 : Step 1 - Resize & Auto-Calibration (CPU)")
    print("  - S2a: Step 2a - YOLO Semantic Inference (GPU)")
    print("  - S2b: Step 2b - Mask Extraction & Bitwise (CPU)")
    print("  - S3 : Step 3 - Depth Anything Inference (GPU)")
    print("  - S4 : Step 4 - Geometry & Surface Normals (CPU/NP)")
    print("  - S5 : Step 5 - OOD Filtering & Connected Components (CPU)")

if __name__ == "__main__":
    run_sweep_benchmark()