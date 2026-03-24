import sys
from pathlib import Path
import numpy as np
import cv2
import torch

# (Phần import và setup DEPTH_REPO giữ nguyên)
DEPTH_REPO = Path(__file__).parent.parent / "Depth-Anything-V2"
if DEPTH_REPO.exists():
    METRIC_PATH = DEPTH_REPO / "metric_depth"
    if METRIC_PATH.exists(): sys.path.insert(0, str(METRIC_PATH))
    else: sys.path.insert(0, str(DEPTH_REPO))

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print("Warning: Không tìm thấy thư viện depth_anything_v2.")

class DepthEstimator:
    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(self, weight_path: str, device: str = "cuda", max_depth: float = 60.0):
        self.device = device
        self.device_type = "cuda" if "cuda" in device else "cpu"
        self.max_depth = max_depth
        self.encoder = self._detect_encoder(weight_path)
        
        cfg = self.MODEL_CONFIGS[self.encoder].copy()
        cfg['max_depth'] = self.max_depth
        
        self.model = DepthAnythingV2(**cfg)
        
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        
        # Tối ưu bộ nhớ nội tại của Torch
        if self.device_type == "cuda":
            torch.backends.cudnn.benchmark = True
            
        print(f"[DepthEstimator] Loaded Metric Depth ({self.encoder}) trên {self.device} (AMP Enabled)")

    def _detect_encoder(self, weight_path: str) -> str:
        name = Path(weight_path).stem.lower()
        for enc in ("vitl", "vitb", "vits"):
            if enc in name: return enc
        return "vitl"

    def predict(self, image_bgr: np.ndarray, K: np.ndarray = None) -> np.ndarray:
        if K is None:
            # Tối ưu hóa: Dùng inference_mode và autocast(FP16)
            with torch.inference_mode(), torch.autocast(device_type=self.device_type, dtype=torch.float16):
                return self.model.infer_image(image_bgr)

        H, W = image_bgr.shape[:2]
        fx = K[0, 0]
        fy = K[1, 1]

        ratio_x_kitti = 620.5 / 725.0087
        ratio_y_kitti = 187.0 / 725.0087

        target_W = int(2 * fx * ratio_x_kitti)
        target_H = int(2 * fy * ratio_y_kitti)

        paste_W = min(W, target_W)
        paste_H = min(H, target_H)

        img_y1, img_y2 = (H - paste_H) // 2, (H - paste_H) // 2 + paste_H
        img_x1, img_x2 = (W - paste_W) // 2, (W - paste_W) // 2 + paste_W

        can_y1, can_y2 = (target_H - paste_H) // 2, (target_H - paste_H) // 2 + paste_H
        can_x1, can_x2 = (target_W - paste_W) // 2, (target_W - paste_W) // 2 + paste_W

        canvas = np.zeros((target_H, target_W, 3), dtype=np.uint8)
        canvas[can_y1:can_y2, can_x1:can_x2] = image_bgr[img_y1:img_y2, img_x1:img_x2]

        # Tối ưu hóa: Dùng inference_mode và autocast(FP16)
        with torch.inference_mode(), torch.autocast(device_type=self.device_type, dtype=torch.float16):
            pred_depth = self.model.infer_image(canvas)

        final_depth = np.zeros((H, W), dtype=np.float32)
        final_depth[img_y1:img_y2, img_x1:img_x2] = pred_depth[can_y1:can_y2, can_x1:can_x2]

        return final_depth