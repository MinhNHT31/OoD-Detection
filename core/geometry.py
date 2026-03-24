import numpy as np
import cv2
import torch

class GeometryEngine:
    def __init__(self, K: np.ndarray, E_v2c: np.ndarray, device: str = "cuda"):
        """
        Khởi tạo với Camera Intrinsics và Sensor Extrinsics.
        Tối ưu hóa: Pre-compute và lưu trữ trên GPU (PyTorch).
        """
        self.device = torch.device(device)
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]

        # Đưa ma trận Extrinsics lên GPU
        self.E_c2v = np.linalg.inv(E_v2c)
        self.R_c2v = torch.tensor(self.E_c2v[:3, :3], dtype=torch.float32, device=self.device)
        self.T_c2v = torch.tensor(self.E_c2v[:3, 3], dtype=torch.float32, device=self.device)

        # Cache cho Ray Grid để tái sử dụng
        self.ray_grid = None

    def _get_ray_grid(self, H: int, W: int) -> torch.Tensor:
        """Tính toán và lưu lại (Cache) lưới tia nhìn 3D."""
        if self.ray_grid is not None and self.ray_grid.shape[:2] == (H, W):
            return self.ray_grid
            
        v, u = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32), 
            indexing='ij'
        )
        X_ray = (u - self.cx) / self.fx
        Y_ray = (v - self.cy) / self.fy
        
        # Shape: (H, W, 3)
        self.ray_grid = torch.stack((X_ray, Y_ray, torch.ones_like(X_ray)), dim=-1)
        return self.ray_grid

    def get_alpha_shape_mask_cv2(self, road_mask: np.ndarray) -> np.ndarray:
        """Trám lỗ hổng bằng Convex Hull (Giữ nguyên vì cv2 đã rất nhanh: ~1ms)"""
        mask_uint8 = (road_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask_uint8)
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                hull = cv2.convexHull(cnt)
                cv2.drawContours(filled_mask, [hull], -1, 255, thickness=-1)
        return filled_mask > 0

    def scale_depth_by_focal_length(self, depth_map: np.ndarray, ai_train_fx: float = 721.5377) -> np.ndarray:
        focal_scale = self.fx / ai_train_fx
        return depth_map * focal_scale

    def get_obstacle_mask_normals_numpy(self, depth_map_real: np.ndarray, mask: np.ndarray, 
                                        angle_thr_deg: float = 60.0) -> tuple:
        """
        Tìm vật cản bằng Vector Pháp Tuyến (Phiên bản PyTorch - CUDA)
        Thời gian dự kiến: Giảm từ ~263ms xuống ~5-15ms
        """
        H, W = depth_map_real.shape

        # 1. Chuyển dữ liệu lên GPU (Dùng non_blocking để không block CPU)
        Z_c = torch.from_numpy(depth_map_real).to(self.device, non_blocking=True)
        mask_t = torch.from_numpy(mask).to(self.device, non_blocking=True)

        # 2. Tạo điểm 3D trong hệ Camera
        rays = self._get_ray_grid(H, W)
        P_cam = rays * Z_c.unsqueeze(-1)

        # 3. Đưa về Hệ Tọa Độ Xe
        P_veh = torch.matmul(P_cam, self.R_c2v.T) + self.T_c2v

        # 4. Tính Vector Pháp tuyến bằng Finite Difference (Nhanh hơn np.gradient)
        dP_dx = torch.zeros_like(P_veh)
        dP_dx[:, 1:-1] = P_veh[:, 2:] - P_veh[:, :-2]

        dP_dy = torch.zeros_like(P_veh)
        dP_dy[1:-1, :] = P_veh[2:, :] - P_veh[:-2, :]

        normals = torch.cross(dP_dx, dP_dy, dim=-1)
        norm_mag = torch.norm(normals, dim=-1, keepdim=True) + 1e-9
        normals = normals / norm_mag

        # 5. Tính góc nghiêng (Trục Z của n_road = [0,0,1], nên Dot Product chính là kênh Z)
        dot_prods = torch.clamp(normals[..., 2], -1.0, 1.0)
        angles_rad = torch.acos(torch.abs(dot_prods))
        angles_deg = torch.rad2deg(angles_rad)

        # 6. Lọc mask
        flags = (angles_deg > angle_thr_deg) & mask_t
        flags[-15:, :] = False 
        
        # Đẩy về lại CPU cho luồng phía sau
        return flags.cpu().numpy(), angles_deg[mask_t].cpu().numpy()