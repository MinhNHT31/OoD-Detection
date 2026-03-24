import numpy as np
import cv2

class GeometryEngine:
    def __init__(self, K: np.ndarray, E_v2c: np.ndarray):
        """
        Khởi tạo với Camera Intrinsics và Sensor Extrinsics.
        """
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]

        self.E_c2v = np.linalg.inv(E_v2c)
        self.R_c2v = self.E_c2v[:3, :3]
        self.T_c2v = self.E_c2v[:3, 3]

    def get_alpha_shape_mask_cv2(self, road_mask: np.ndarray) -> np.ndarray:
        """Trám lỗ hổng bằng Convex Hull (Tốc độ ~1-2ms)"""
        mask_uint8 = (road_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask_uint8)
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                hull = cv2.convexHull(cnt)
                cv2.drawContours(filled_mask, [hull], -1, 255, thickness=-1)
        return filled_mask > 0

    def scale_depth_by_focal_length(self, depth_map: np.ndarray, ai_train_fx: float = 721.5377) -> np.ndarray:
        """
        Bù trừ sự sai lệch không gian do khác biệt tiêu cự thấu kính.
        ai_train_fx = 721.5377 (Lấy chuẩn từ ma trận P2 của dataset KITTI).
        """
        focal_scale = self.fx / ai_train_fx
        return depth_map * focal_scale

    def get_obstacle_mask_normals_numpy(self, depth_map_real: np.ndarray, mask: np.ndarray, 
                                        angle_thr_deg: float = 60.0) -> tuple:
        """
        Tìm vật cản bằng Vector Pháp Tuyến trên không gian 3D đã chuẩn tỉ lệ.
        """
        H, W = depth_map_real.shape

        # 1. Tạo điểm 3D trong hệ Camera
        v, u = np.indices((H, W))
        Z_c = depth_map_real
        X_c = (u - self.cx) * Z_c / self.fx
        Y_c = (v - self.cy) * Z_c / self.fy
        P_cam = np.stack((X_c, Y_c, Z_c), axis=-1)

        # 2. Đưa về Hệ Tọa Độ Xe
        P_veh = P_cam @ self.R_c2v.T + self.T_c2v

        # 3. Tính Vector Pháp tuyến
        dP_dy, dP_dx = np.gradient(P_veh, axis=(0, 1))
        normals = np.cross(dP_dx, dP_dy)
        norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9
        normals = normals / norm_mag

        n_road = np.array([0., 0., 1.], dtype=np.float64)

        # 4. Tính góc nghiêng
        dot_prods = np.sum(normals * n_road, axis=-1)
        dot_prods = np.clip(dot_prods, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(np.abs(dot_prods)))

        flags = (angles_deg > angle_thr_deg) & mask
        flags[-15:, :] = False 
        
        return flags, angles_deg[mask]
    def calculate_distance_by_raycasting(self, u: int, v: int) -> float:
        """
        ĐO LƯỜNG THUẦN VẬT LÝ: Bắn tia từ Camera qua pixel (u, v) xuống mặt đường (Z=0).
        - Hoạt động độc lập 100% với AI Depth Map. Kháng mọi sai số do AI bị giãn không gian.
        """
        # 1. Tạo vector hướng của tia sáng trong không gian Camera
        X_c = (u - self.cx) / self.fx
        Y_c = (v - self.cy) / self.fy
        d_cam = np.array([X_c, Y_c, 1.0])

        # 2. Quay vector hướng này sang hệ trục tọa độ của Xe
        d_veh = self.R_c2v @ d_cam

        # 3. Tìm giao điểm của tia với mặt phẳng đường
        cam_z = self.T_c2v[2]  # Chiều cao camera (vd: 1.88m)
        ray_z_dir = d_veh[2]   # Hướng Z của tia (phải âm thì mới đâm xuống đất)

        if ray_z_dir >= -1e-6: return -1.0 

        # t là hệ số kéo dài tia cho đến khi chạm Z = 0
        t = -cam_z / ray_z_dir

        # 4. Tọa độ (X, Y) trên mặt đất
        intersection_pt = self.T_c2v + t * d_veh
        
        # Khoảng cách Euclide 2D từ gốc tọa độ Xe đến điểm chạm đất
        distance = np.sqrt(intersection_pt[0]**2 + intersection_pt[1]**2)

        return float(distance)