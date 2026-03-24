import argparse
import json
import numpy as np
from pathlib import Path
from pipeline_manager import ObstacleDetectionPipeline

def parse_args():
    p = argparse.ArgumentParser(description="Obstacle Detection Pipeline (Refactored)")
    p.add_argument("--image", required=True, help="Đường dẫn ảnh đầu vào")
    p.add_argument("--yolo", default="weights/best_yolov8n.pt", help="Weight YOLOv8-seg")
    p.add_argument("--depth", default="weights/depth_anything_v2_metric_vkitti_vits.pth", help="Weight DepthAnythingV2 metric")
    p.add_argument("--save-dir", default="output", help="Thư mục lưu output")
    p.add_argument("--fx", type=float, required=True, help="Focal length x")
    p.add_argument("--fy", type=float, required=True, help="Focal length y")
    p.add_argument("--cx", type=float, required=True, help="Principal point x")
    p.add_argument("--cy", type=float, required=True, help="Principal point y")
    p.add_argument("--yolo-conf", type=float, default=0.4)
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--plane-dist-thr", type=float, default=0.05)
    p.add_argument("--elevation-thr", type=float, default=0.15)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--save-json", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    K = np.array([
        [args.fx, 0,       args.cx],
        [0,       args.fy, args.cy],
        [0,       0,       1      ]
    ], dtype=np.float64)

    # Khởi tạo Pipeline
    pipeline = ObstacleDetectionPipeline(
        yolo_weight=args.yolo,
        depth_weight=args.depth,
        K=K
    )

    # Chạy
    result = pipeline.run(
        image_path=args.image,
        yolo_conf=args.yolo_conf,
        alpha=args.alpha,
        plane_dist_thr=args.plane_dist_thr,
        elevation_thr=args.elevation_thr,
        visualize=args.visualize,
        save_dir=args.save_dir
    )

    detections = result["detections"]
    print(f"\n=> Tổng vật cản phát hiện: {len(detections)}")
    if detections:
        depths = [d for _, _, d in detections]
        print(f"=> Khoảng cách: {min(depths):.2f}m ~ {max(depths):.2f}m")

    if args.save_json:
        stem = Path(args.image).stem
        json_path = Path(args.save_dir) / f"{stem}_detections.json"
        with open(json_path, "w") as f:
            json.dump([{"row": r, "col": c, "depth_m": d} for r, c, d in detections], f, indent=2)
        print(f"=> Detections saved: {json_path}")

if __name__ == "__main__":
    main()