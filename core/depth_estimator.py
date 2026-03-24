import sys
from pathlib import Path
import numpy as np
import cv2
import torch

# Đảm bảo bạn đã clone repo Depth-Anything-V2 cùng cấp
DEPTH_REPO = Path(__file__).parent.parent / "Depth-Anything-V2"
if DEPTH_REPO.exists():
    METRIC_PATH = DEPTH_REPO / "metric_depth"
    if METRIC_PATH.exists():
        sys.path.insert(0, str(METRIC_PATH))
    else:
        sys.path.insert(0, str(DEPTH_REPO))

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
        self.max_depth = max_depth
        self.encoder = self._detect_encoder(weight_path)
        
        cfg = self.MODEL_CONFIGS[self.encoder].copy()
        cfg['max_depth'] = self.max_depth
        
        self.model = DepthAnythingV2(**cfg)
        
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(f"[DepthEstimator] Loaded Metric Depth ({self.encoder}) trên {self.device}")

    def _detect_encoder(self, weight_path: str) -> str:
        name = Path(weight_path).stem.lower()
        for enc in ("vitl", "vitb", "vits"):
            if enc in name: return enc
        return "vitl"

    def predict(self, image_bgr: np.ndarray, K: np.ndarray = None) -> np.ndarray:
        """
        Dự đoán Depth Map có bù trừ FOV tự động (Bảo toàn tuyệt đối Local Geometry).
        Dựa trên thông số FOV chuẩn của Virtual KITTI 2 (81.1° ngang, 28.9° dọc).
        
        :param image_bgr: Ảnh đầu vào.
        :param K: Ma trận Camera Intrinsics (3x3).
        """
        if K is None:
            with torch.no_grad():
                return self.model.infer_image(image_bgr)

        H, W = image_bgr.shape[:2]
        fx = K[0, 0]
        fy = K[1, 1]

        # 1. Tỉ lệ chuẩn của KITTI (tan(FOV/2))
        ratio_x_kitti = 620.5 / 725.0087
        ratio_y_kitti = 187.0 / 725.0087

        # 2. Kích thước khung hình mục tiêu để đạt FOV chuẩn với tiêu cự hiện tại
        target_W = int(2 * fx * ratio_x_kitti)
        target_H = int(2 * fy * ratio_y_kitti)

        # 3. Khởi tạo Canvas
        canvas = np.zeros((target_H, target_W, 3), dtype=np.uint8)

        # 4. Tính toán kích thước khối ảnh thực tế sẽ được giao thoa (copy/paste)
        # Giúp triệt tiêu hoàn toàn lỗi chênh lệch số lẻ 1 pixel
        paste_W = min(W, target_W)
        paste_H = min(H, target_H)

        # 5. Tọa độ cắt từ Ảnh Gốc
        img_y1 = (H - paste_H) // 2
        img_y2 = img_y1 + paste_H
        img_x1 = (W - paste_W) // 2
        img_x2 = img_x1 + paste_W

        # 6. Tọa độ dán lên Canvas
        can_y1 = (target_H - paste_H) // 2
        can_y2 = can_y1 + paste_H
        can_x1 = (target_W - paste_W) // 2
        can_x2 = can_x1 + paste_W

        # Dán phần hữu ích của ảnh gốc vào Canvas
        canvas[can_y1:can_y2, can_x1:can_x2] = image_bgr[img_y1:img_y2, img_x1:img_x2]

        # 7. Cho AI suy luận trên Canvas
        with torch.no_grad():
            pred_depth = self.model.infer_image(canvas)

        # 8. Khôi phục lại kích thước ảnh gốc (W x H)
        final_depth = np.zeros((H, W), dtype=np.float32)
        
        # Bóc phần độ sâu thật gắn trở lại vị trí cũ trên ảnh gốc
        final_depth[img_y1:img_y2, img_x1:img_x2] = pred_depth[can_y1:can_y2, can_x1:can_x2]

        return final_depth