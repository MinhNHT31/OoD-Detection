import cv2
import numpy as np

def save_visualizations(result: dict, save_dir: str, stem: str) -> None:
    img        = result["image_bgr"]
    depth_map  = result["depth_map"]
    road_mask  = result["road_mask"]
    alpha_mask = result["alpha_mask"]
    elevated   = result["elevated_mask"]

    # Depth map
    depth_vis = (depth_map - depth_map.min()) / (np.ptp(depth_map) + 1e-6)
    depth_colored = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    cv2.imwrite(f"{save_dir}/{stem}_depth.png", depth_colored)

    # Road & Alpha Masks
    def overlay_mask(base_img, mask, color):
        out = base_img.copy()
        out[mask] = (out[mask] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
        return out

    cv2.imwrite(f"{save_dir}/{stem}_road_mask.png", overlay_mask(img, road_mask, [0, 200, 0]))
    cv2.imwrite(f"{save_dir}/{stem}_alpha_mask.png", overlay_mask(img, alpha_mask, [0, 200, 255]))

    # Elevated Mask & Bounding Boxes
    result_img = overlay_mask(img, elevated, [0, 0, 255])
    contours, _ = cv2.findContours(elevated.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 50: continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imwrite(f"{save_dir}/{stem}_elevated.png", result_img)
    print(f"[Visualize] Đã lưu ảnh kết quả vào thư mục: {save_dir}/")