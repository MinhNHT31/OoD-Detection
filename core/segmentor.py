import numpy as np
import cv2
from ultralytics import YOLO

class Segmentor:
    DEFAULT_COLOR_MAP = {
        'road': '#8B4513',           
        'sidewalk': '#808080',       
        'building': '#4169E1',       
        '2_wheel': '#FF0000',        
        '4_wheel': '#69FF7D',        
        'person': '#FF6614',         
        'background': '#000000'      
    }

    def __init__(self, weight_path: str, color_map: dict = None):
        """
        Khởi tạo Segmentor để xuất ảnh color mask.
        :param weight_path: Đường dẫn tới weight của YOLO.
        :param color_map: Dictionary quy định màu HEX cho từng class.
        """
        self.model = YOLO(weight_path)
        self.color_map = color_map if color_map else self.DEFAULT_COLOR_MAP
        
        # Tiền xử lý: Chuyển đổi mã HEX sang chuẩn BGR của OpenCV
        self.bgr_colors = {}
        for cls, hex_color in self.color_map.items():
            hex_color = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            self.bgr_colors[cls] = (b, g, r) # OpenCV dùng BGR
            
        # print(f"[Segmentor] Loaded YOLO model từ {weight_path}")

    def predict(self, image_bgr: np.ndarray, conf_threshold: float = 0.4) -> np.ndarray:
        """
        Trả về colored mask (BGR) dựa trên các vật thể phát hiện được.
        """
        H, W = image_bgr.shape[:2]
        
        # 1. Khởi tạo canvas với màu background (mặc định là đen)
        bg_color = self.bgr_colors.get('background', (0, 0, 0))
        colored_mask = np.full((H, W, 3), bg_color, dtype=np.uint8)

        results = self.model(image_bgr, conf=conf_threshold, verbose=False)
        for result in results:
            # print(f"[Segmentor] Phát hiện {len(result.boxes)} đối tượng với conf >= {conf_threshold}")
            if result.masks is None: continue
            names = result.names
            
            for i, cls_id in enumerate(result.boxes.cls.cpu().numpy().astype(int)):
                cls_name = names.get(cls_id, "").lower()
                conf_score = result.boxes.conf[i]
                # print(f"[Segmentor] Phát hiện: {cls_name} (ID: {cls_id}) - Conf: {conf_score:.2f}")
                
                # 2. Kiểm tra xem class này có trong color_map không
                if cls_name in self.bgr_colors:
                    color = self.bgr_colors[cls_name]
                    
                    # 3. Lấy mask nhị phân và resize về ảnh gốc
                    mask_i = result.masks.data[i].cpu().numpy()
                    mask_i = cv2.resize(mask_i, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    
                    # 4. Áp màu BGR vào những pixel thuộc vật thể
                    colored_mask[mask_i] = color
                    
        return colored_mask