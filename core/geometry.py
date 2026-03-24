import numpy as np
import open3d as o3d
import cv2
import alphashape
from shapely.geometry import Polygon, MultiPolygon
from shapely import prepare, contains_xy

class GeometryEngine:
    def __init__(self, K: np.ndarray):
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]

    def backproject(self, depth_map: np.ndarray, pixel_mask: np.ndarray) -> tuple:
        rows, cols = np.where(pixel_mask)
        Z = depth_map[rows, cols]

        valid = (Z > 0) & np.isfinite(Z)
        rows, cols, Z = rows[valid], cols[valid], Z[valid]

        X = (cols - self.cx) * Z / self.fx
        Y = (rows - self.cy) * Z / self.fy

        points_3d = np.stack([X, Y, Z], axis=1).astype(np.float64)
        pixel_coords = np.stack([rows, cols], axis=1)
        return points_3d, pixel_coords

    def fit_ground_plane(self, points_3d: np.ndarray, dist_thr: float = 0.05, ransac_n: int = 3) -> np.ndarray:
        if len(points_3d) < ransac_n:
            raise ValueError(f"Không đủ {ransac_n} điểm để fit plane.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        plane_model, inliers = pcd.segment_plane(distance_threshold=dist_thr, ransac_n=ransac_n, num_iterations=1000)
        return np.array(plane_model, dtype=np.float64) # [a, b, c, d]

    def get_alpha_shape_mask(self, road_mask: np.ndarray, alpha: float = 0.02, downsample: int = 4) -> np.ndarray:
        H, W = road_mask.shape
        rows, cols = np.where(road_mask)
        if len(rows) < 4: return road_mask.copy()

        idx = np.arange(0, len(rows), downsample)
        pts_2d = np.stack([cols[idx], rows[idx]], axis=1).astype(float)
        shape = alphashape.alphashape(pts_2d, alpha)

        if shape is None or shape.is_empty: return road_mask.copy()

        prepare(shape)
        minx, miny, maxx, maxy = shape.bounds
        row_lo, row_hi = max(0, int(miny)), min(H, int(maxy) + 1)
        col_lo, col_hi = max(0, int(minx)), min(W, int(maxx) + 1)

        rows_all, cols_all = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        sub_rows = rows_all[row_lo:row_hi, col_lo:col_hi].ravel()
        sub_cols = cols_all[row_lo:row_hi, col_lo:col_hi].ravel()

        inside_mask = np.zeros((H, W), dtype=bool)
        flags = contains_xy(shape, sub_cols.astype(float), sub_rows.astype(float))
        inside_mask[sub_rows[flags], sub_cols[flags]] = True

        return inside_mask

    def get_signed_distance(self, points_3d: np.ndarray, plane_eq: np.ndarray) -> np.ndarray:
        a, b, c, d = plane_eq
        norm = np.sqrt(a**2 + b**2 + c**2) + 1e-9
        X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        return (a * X + b * Y + c * Z + d) / norm

    def get_elevated_mask(self, dist: np.ndarray, pixel_coords: np.ndarray, depth_vals: np.ndarray, 
                          H: int, W: int, threshold: float = 0.15, sign: str = "above") -> tuple:
        if sign == "above": flags = dist > threshold
        elif sign == "below": flags = dist < -threshold
        else: flags = np.abs(dist) > threshold

        elevated_mask = np.zeros((H, W), dtype=bool)
        rows, cols = pixel_coords[flags, 0], pixel_coords[flags, 1]
        elevated_mask[rows, cols] = True

        detections = [(int(r), int(c), float(z)) for r, c, z in zip(rows, cols, depth_vals[flags])]
        return elevated_mask, detections
    
    def get_obstacle_mask_with_normals(self, points_3d: np.ndarray, pixel_coords: np.ndarray, depth_vals: np.ndarray,
                                       plane_eq: np.ndarray, H: int, W: int, 
                                       angle_thr_deg: float = 30.0) -> tuple:
        """
        Phát hiện vật cản dựa trên sự chênh lệch góc giữa Vector pháp tuyến cục bộ
        và Vector pháp tuyến của mặt phẳng đường (RANSAC).
        """
        # 1. Lấy vector pháp tuyến của đường từ phương trình RANSAC (a, b, c)
        a, b, c, d = plane_eq
        n_road = np.array([a, b, c], dtype=np.float64)
        n_road = n_road / (np.linalg.norm(n_road) + 1e-9) # Chuẩn hóa về độ dài 1

        # Đảm bảo vector pháp tuyến hướng về phía Camera (Camera nhìn theo trục +Z)
        # Trong hệ tọa độ OpenCV, Z hướng về phía trước, Y hướng xuống đất.
        # Mặt đường thường có pháp tuyến hướng lên (Y âm).
        if n_road[1] > 0: 
            n_road = -n_road

        # 2. Xây dựng PointCloud và tính Normal cho từng điểm
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        # Dùng KNN = 30 để làm mượt nhiễu từ mô hình Depth
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        
        # Đồng bộ hướng của tất cả vector cục bộ về phía Camera
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        normals = np.asarray(pcd.normals)

        # 3. Tính góc lệch giữa normals cục bộ và n_road
        # cos(theta) = |n_point . n_road|
        dot_prods = np.sum(normals * n_road, axis=1)
        dot_prods = np.clip(dot_prods, -1.0, 1.0) # Tránh lỗi floating point khi arccos
        
        # Dùng trị tuyệt đối để chỉ quan tâm độ song song/vuông góc, không phân biệt chiều
        angles_rad = np.arccos(np.abs(dot_prods))
        angles_deg = np.degrees(angles_rad)

        # 4. Lọc vật cản bằng ngưỡng góc (Ví dụ: lệch quá 30 độ)
        flags = angles_deg > angle_thr_deg

        # Tạo mask 2D
        obstacle_mask = np.zeros((H, W), dtype=bool)
        rows, cols = pixel_coords[flags, 0], pixel_coords[flags, 1]
        obstacle_mask[rows, cols] = True

        # Trả về format detections tương tự hàm cũ
        detections = [(int(r), int(c), float(z)) for r, c, z in zip(rows, cols, depth_vals[flags])]
        
        return obstacle_mask, detections, angles_deg
    
    def get_alpha_shape_mask_cv2(self, road_mask: np.ndarray) -> np.ndarray:
        """
        Dùng Convex Hull (Bao lồi) để bắc cầu qua các vết lõm ở rìa đường.
        Giải quyết triệt để trường hợp vật cản nằm ở mép mask YOLO.
        """
        mask_uint8 = (road_mask * 255).astype(np.uint8)
        
        # 1. Tìm các đường viền ngoài cùng
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filled_mask = np.zeros_like(mask_uint8)
        
        # 2. Tính Convex Hull cho các mảng đường lớn
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                # Lệnh này tạo ra một đa giác lồi, tự động "kéo thẳng" qua các vết lõm
                hull = cv2.convexHull(cnt)
                
                # Tô kín vùng bao lồi
                cv2.drawContours(filled_mask, [hull], -1, 255, thickness=-1)
                
        return filled_mask > 0
    
    def get_obstacle_mask_normals_numpy(self, depth_map: np.ndarray, mask: np.ndarray, 
                                    plane_eq: np.ndarray, angle_thr_deg: float = 30.0) -> tuple:
        H, W = depth_map.shape
        
        # 1. Tạo ma trận tọa độ X, Y, Z cho toàn bộ ảnh
        v, u = np.indices((H, W))
        Z = depth_map
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        
        # Gom thành một ma trận 3D ảnh: shape (H, W, 3)
        P = np.stack((X, Y, Z), axis=-1)
        
        # 2. Tính đạo hàm theo trục X và Y bằng Numpy (Tốc độ cực nhanh)
        # Lấy gradient dọc theo trục Y (axis=0) và trục X (axis=1)
        dP_dy, dP_dx = np.gradient(P, axis=(0, 1))
        
        # 3. Tính Vector Pháp tuyến bằng tích có hướng (Cross Product)
        normals = np.cross(dP_dx, dP_dy)
        
        # Chuẩn hóa độ dài vector về 1
        norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9
        normals = normals / norm_mag
        
        # 4. So sánh với Ground Plane (a, b, c)
        n_road = np.array(plane_eq[:3], dtype=np.float64)
        n_road = n_road / (np.linalg.norm(n_road) + 1e-9)
        if n_road[1] > 0: n_road = -n_road
        
        # Tính tích vô hướng để tìm góc lệch cho TOÀN BỘ ảnh cùng lúc
        dot_prods = np.sum(normals * n_road, axis=-1)
        dot_prods = np.clip(dot_prods, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(np.abs(dot_prods)))
        
        # 5. Lọc vật cản (Chỉ xét những pixel nằm trong mask)
        flags = (angles_deg > angle_thr_deg) & mask
        
        rows, cols = np.where(flags)
        detections = [(int(r), int(c), float(Z[r, c])) for r, c in zip(rows, cols)]
        
        # Lấy angles của các điểm hợp lệ để visualize nếu cần
        valid_angles = angles_deg[mask] 
        
        return flags, detections, valid_angles