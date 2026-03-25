"""
PipelineManager
===============
Tối ưu hóa: Hardcode Target Resolution (Ví dụ: 924x518).
Fix: Lưu trữ intermediate output để tránh chạy lại AI gây lãng phí thời gian.
Đo lường: Tính tổng thời gian End-to-End chính xác.
"""

import time
import numpy as np
import cv2
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from core.segmentor import Segmentor
from core.depth_estimator import DepthEstimator
from core.geometry import GeometryEngine

# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    label      : str
    bbox       : Tuple[int, int, int, int]
    distance_m : float
    mask       : Optional[np.ndarray] = field(default=None, repr=False)

# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class PipelineManager:
    BG_CLASSES = {"road", "sidewalk", "building"}
    OBSTACLE_CLASSES = {"2_wheel", "4_wheel", "person"}

    MIN_AREA_KNOWN_PX = 500
    MIN_AREA_OOD_PX   = 1500
    MAX_AREA_PX       = 50000
    MATCH_OVERLAP_RATIO = 0.25
    MIN_DEPTH_M        = 0.5
    MAX_DEPTH_M        = 60.0

    def __init__(
        self,
        seg_weight   : str,
        depth_weight : str,
        K            : np.ndarray,
        E_v2c        : np.ndarray,
        device       : str   = "cuda",
        max_depth    : float = 60.0,
        angle_thr_deg: float = 60.0,
        conf_threshold: float = 0.4,
        target_size  : Tuple[int, int] = (924, 518)
    ):
        self.device        = device
        self.K_orig        = K
        self.E_v2c         = E_v2c
        self.angle_thr_deg = angle_thr_deg
        self.conf_threshold= conf_threshold
        self.target_size   = target_size

        print(f"[PipelineManager] Khởi tạo hệ thống. Target Resolution: {target_size[0]}x{target_size[1]}")
        self.segmentor = Segmentor(seg_weight)
        self.depth_est = DepthEstimator(depth_weight, device=device, max_depth=max_depth)
        
        self.geo = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.K_scaled = None
        
        # Biến cache để lưu dữ liệu vẽ đồ thị, tránh chạy lại AI
        self.last_colored_mask = None
        self.last_depth_map = None
        
        print("[PipelineManager] Sẵn sàng.")

    def _color_to_binary(self, colored_mask_bgr: np.ndarray, color_bgr: tuple) -> np.ndarray:
        c = np.array(color_bgr, dtype=np.uint8)
        return cv2.inRange(colored_mask_bgr, c, c)

    def _build_non_bg_mask(self, colored_mask_bgr: np.ndarray) -> np.ndarray:
        H, W = colored_mask_bgr.shape[:2]
        bg_acc = np.zeros((H, W), dtype=np.uint8)
        for cls in self.BG_CLASSES:
            color = self.segmentor.bgr_colors.get(cls)
            if color is None: continue
            layer = self._color_to_binary(colored_mask_bgr, color)
            bg_acc = cv2.bitwise_or(bg_acc, layer)
        return cv2.bitwise_not(bg_acc)

    def _build_known_masks(self, colored_mask_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        for cls in self.OBSTACLE_CLASSES:
            color = self.segmentor.bgr_colors.get(cls)
            if color is None: continue
            out[cls] = self._color_to_binary(colored_mask_bgr, color)
        return out

    def _build_road_mask(self, colored_mask_bgr: np.ndarray) -> np.ndarray:
        color = self.segmentor.bgr_colors.get("road")
        if color is None: return np.zeros(colored_mask_bgr.shape[:2], dtype=bool)
        binary = self._color_to_binary(colored_mask_bgr, color)
        return binary > 0

    def _get_distance(self, depth_map: np.ndarray, instance_mask: np.ndarray, bbox: Tuple) -> float:
        x, y, w, h = bbox
        y_lower = y + h // 2
        roi_depth = depth_map[y_lower:y+h, x:x+w]
        roi_mask = instance_mask[y_lower:y+h, x:x+w].astype(bool)
        valid = roi_depth[roi_mask]
        valid = valid[(valid > self.MIN_DEPTH_M) & (valid < self.MAX_DEPTH_M)]
        return float(np.min(valid)) if len(valid) else -1.0

    def _match_label(self, instance_mask: np.ndarray, known_masks: Dict[str, np.ndarray], area: int) -> str:
        best_label = "OOD"
        best_overlap = 0
        inst_bool = instance_mask.astype(bool)
        for cls_name, cls_mask in known_masks.items():
            overlap = int(np.count_nonzero(inst_bool & (cls_mask > 0)))
            if overlap > best_overlap and overlap >= area * self.MATCH_OVERLAP_RATIO:
                best_overlap = overlap
                best_label = cls_name
        return best_label

    def run(self, image_bgr: np.ndarray) -> List[Detection]:
        profiling_times = {}
        total_start = time.perf_counter()

        # ── AUTO-CALIBRATION ─────────────────────
        if self.geo is None:
            orig_H, orig_W = image_bgr.shape[:2]
            target_W, target_H = self.target_size
            
            self.scale_x = target_W / orig_W
            self.scale_y = target_H / orig_H
            
            self.K_scaled = self.K_orig.copy()
            self.K_scaled[0, :] *= self.scale_x  
            self.K_scaled[1, :] *= self.scale_y  
            
            self.geo = GeometryEngine(self.K_scaled, self.E_v2c, self.device)

        # ── TIỀN XỬ LÝ RESIZE ─────────────────────────────────────────────────
        img_infer = cv2.resize(image_bgr, self.target_size, interpolation=cv2.INTER_LINEAR)

        # ── STEP 2: Semantic ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        colored_mask = self.segmentor.predict(img_infer, conf_threshold=self.conf_threshold)
        non_bg_mask  = self._build_non_bg_mask(colored_mask)
        known_masks  = self._build_known_masks(colored_mask)
        road_mask    = self._build_road_mask(colored_mask)
        profiling_times['Step 2 (Semantic)'] = (time.perf_counter() - t0) * 1000

        # ── STEP 3: Depth estimation ──────────────────────────────────────────
        t0 = time.perf_counter()
        depth_map_real = self.depth_est.predict(img_infer, K=self.K_scaled)
        profiling_times['Step 3 (Depth)'] = (time.perf_counter() - t0) * 1000

        # CACHE LẠI ĐỂ DÙNG CHO HÀM VISUALIZE
        self.last_colored_mask = colored_mask
        self.last_depth_map = depth_map_real

        # ── STEP 4: Geometry Engine ───────────────────────────────────────────
        t0 = time.perf_counter()
        alpha_mask = self.geo.get_alpha_shape_mask_cv2(road_mask)
        obstacle_flags, _ = self.geo.get_obstacle_mask_normals_numpy(
            depth_map_real, alpha_mask, self.angle_thr_deg
        )
        profiling_times['Step 4 (Geometry)'] = (time.perf_counter() - t0) * 1000

        # ── STEP 5: Lọc OOD + BBox + Distance ────────────────────────────────
        t0 = time.perf_counter()
        filtered = (obstacle_flags & (non_bg_mask > 0)).astype(np.uint8) * 255
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN,  kernel)

        num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(filtered, connectivity=8)
        detections: List[Detection] = []

        scale_area = self.scale_x * self.scale_y
        min_area_known = int(self.MIN_AREA_KNOWN_PX * scale_area)
        min_area_ood   = int(self.MIN_AREA_OOD_PX * scale_area)
        max_area       = int(self.MAX_AREA_PX * scale_area)

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area_known or area > max_area: continue

            x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                          int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            bbox_infer = (x, y, w, h)
            instance_mask = (labels_map == i).astype(np.uint8)

            label = self._match_label(instance_mask, known_masks, area)
            if label == "OOD" and area < min_area_ood: continue

            dist = self._get_distance(depth_map_real, instance_mask, bbox_infer)
            if dist < 0: continue

            orig_bbox = (
                int(x / self.scale_x),
                int(y / self.scale_y),
                int(w / self.scale_x),
                int(h / self.scale_y)
            )

            detections.append(Detection(
                label      = label,
                bbox       = orig_bbox,
                distance_m = round(dist, 2),
                mask       = instance_mask, 
            ))

        detections.sort(key=lambda d: d.distance_m)
        profiling_times['Step 5 (Filtering)'] = (time.perf_counter() - t0) * 1000

        # ── IN BÁO CÁO MODULES ─────────────────────────────────────────────
        total_time = (time.perf_counter() - total_start) * 1000
        print(f"\n{'='*50}")
        print(f"⏱️  MODULES TIMING ({self.target_size[0]}x{self.target_size[1]})")
        print(f"{'-'*50}")
        for step_name, t_ms in profiling_times.items(): print(f"{step_name:<28} : {t_ms:>7.2f} ms")
        print(f"{'-'*50}")
        print(f"{'Sum of Modules':<28} : {total_time:>7.2f} ms")
        print(f"{'='*50}\n")

        return detections

# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION & EXPORT
# ──────────────────────────────────────────────────────────────────────────────

_LABEL_COLORS_BGR = {
    "person" : (  0, 200,   0),
    "2_wheel": (  0, 200, 255),
    "4_wheel": (  0, 180, 255),
    "OOD"    : (  0,   0, 230),
}

def draw_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    vis = image_bgr.copy()
    for det in detections:
        x, y, w, h = det.bbox
        color = _LABEL_COLORS_BGR.get(det.label, (180, 180, 180))
        text  = f"{det.label}  {det.distance_m:.1f}m"

        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(vis, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return vis

def export_json(detections: List[Detection]) -> List[dict]:
    return [
        {
            "label"     : d.label,
            "bbox"      : {"x": d.bbox[0], "y": d.bbox[1], "w": d.bbox[2], "h": d.bbox[3]},
            "distance_m": d.distance_m,
        }
        for d in detections
    ]

# ──────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def visualize(
    image_bgr    : np.ndarray,
    colored_mask : np.ndarray,
    depth_map    : np.ndarray,
    detections   : List[Detection],
    save_path    : str = "output_detection.jpg",
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    img_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_H, orig_W = image_bgr.shape[:2]
    
    mask_vis = cv2.resize(colored_mask, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
    mask_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
    depth_map_resized = cv2.resize(depth_map, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
    depth_vis = np.clip(depth_map_resized, 0, 60)

    det_img = cv2.cvtColor(draw_detections(image_bgr, detections), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(22, 10), facecolor="#0f0f0f")
    fig.suptitle(f"CV Pipeline — {orig_W}x{orig_H} (Infer: {colored_mask.shape[1]}x{colored_mask.shape[0]})", 
                 fontsize=16, fontweight="bold", color="white", y=0.98)

    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1.2], hspace=0.35, wspace=0.08,
                          left=0.03, right=0.97, top=0.93, bottom=0.04)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_seg = fig.add_subplot(gs[0, 1])
    ax_dep = fig.add_subplot(gs[0, 2])
    ax_det = fig.add_subplot(gs[0, 3])
    ax_bar = fig.add_subplot(gs[1, :])

    label_kw = dict(fontsize=10, color="#aaaaaa", pad=6, loc="left", fontweight="bold")
    ax_img.imshow(img_rgb)
    ax_img.set_title("① Input (Original HD)", **label_kw)
    ax_img.axis("off")

    ax_seg.imshow(mask_rgb)
    ax_seg.set_title("② colored_mask (Up-scaled)", **label_kw)
    ax_seg.axis("off")

    im = ax_dep.imshow(depth_vis, cmap="plasma", vmin=0, vmax=60)
    ax_dep.set_title("③ depth_map_real (0 – 60 m)", **label_kw)
    ax_dep.axis("off")
    cbar = fig.colorbar(im, ax=ax_dep, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("mét", color="#aaaaaa", fontsize=8)

    ax_det.imshow(det_img)
    ax_det.set_title("④ Detections (Accurate BBox)", **label_kw)
    ax_det.axis("off")

    ax_bar.set_facecolor("#111111")
    for spine in ax_bar.spines.values(): spine.set_edgecolor("#333333")

    _MPLCOLORS = {"person": "#00c864", "2_wheel": "#ffd700", "4_wheel": "#00aaff", "OOD": "#ff3333"}
    top = detections[:15]

    if top:
        labels = [f"{d.label}\n({d.distance_m:.1f}m)" for d in top]
        dists = [d.distance_m for d in top]
        colors = [_MPLCOLORS.get(d.label, "#888888") for d in top]
        bars = ax_bar.barh(range(len(top)), dists, color=colors, height=0.6, edgecolor="#222222", linewidth=0.4)
        ax_bar.set_yticks(range(len(top)))
        ax_bar.set_yticklabels(labels, fontsize=8.5, color="white")
        ax_bar.set_xlabel("Khoảng cách (mét)", color="#aaaaaa", fontsize=9)
        ax_bar.set_title(f"⑤ Distance ranking (top {len(top)} vật thể gần nhất)", **label_kw)
        ax_bar.tick_params(axis="x", colors="#aaaaaa", labelsize=8)
        ax_bar.invert_yaxis()
        for bar, dist in zip(bars, dists):
            ax_bar.text(dist + 0.3, bar.get_y() + bar.get_height() / 2, f"{dist:.1f} m", va="center", fontsize=8, color="white")
        legend_handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in _MPLCOLORS.items()]
        ax_bar.legend(handles=legend_handles, loc="lower right", fontsize=8, facecolor="#222222", edgecolor="#444444", labelcolor="white")
    else:
        ax_bar.text(0.5, 0.5, "Không phát hiện vật thể nào", ha="center", va="center", color="#666666", fontsize=12, transform=ax_bar.transAxes)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[Visualize] Đã lưu → {save_path}")
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2 

    K = np.array([
        [1837.189936,    0.000000, 912.272553],
        [   0.000000, 1844.888039, 728.975910],
        [   0.000000,    0.000000,   1.000000],
    ], dtype=np.float64)

    E_v2c = np.array([
        [ 0.050593, -0.998719,  0.000873, -0.067766],
        [ 4.4e-05,  -0.000872, -1.000000,  1.880442],
        [ 0.998719,  0.050593,  0.000000, -1.305326],
        [ 0.000000,  0.000000,  0.000000,  1.000000],
    ], dtype=np.float64)

    pipeline = PipelineManager(
        seg_weight    = "weights/best_yolov8l.pt",
        depth_weight  = "weights/depth_anything_v2_metric_vkitti_vits.pth",
        K             = K,
        E_v2c         = E_v2c,
        device        = "cuda",
        max_depth     = 60.0,
        angle_thr_deg = 50.0,
        conf_threshold= 0.2,
        target_size   = (924, 518)  
    )

    IMG_PATH = "datasets/phenikaa_dataset/CAM_P_F/1758620947-999741793.jpg"
    image = cv2.imread(IMG_PATH)
    assert image is not None, f"Không đọc được ảnh: {IMG_PATH}"

    print("\n⏳ Đang khởi động (Warmup) các mô hình AI...")
    _ = pipeline.run(image)

    print("\n🚀 BẮT ĐẦU ĐO LƯỜNG THỰC TẾ...")
    total_prog_start = time.perf_counter()
    
    # Chạy thực tế một lần duy nhất
    detections = pipeline.run(image)
    
    # Tính tổng thời gian hệ thống End-to-End
    total_prog_time = (time.perf_counter() - total_prog_start) * 1000

    print(f"\n{'─'*50}")
    print(f"  TỔNG KẾT VẬT THỂ: {len(detections)}")
    print(f"{'─'*50}")
    for d in detections:
        print(f"  [{d.label:8s}]  bbox={d.bbox}  dist={d.distance_m:.2f}m")
    print(f"{'─'*50}")
    
    # ĐÂY LÀ CHỈ SỐ FPS CHUẨN XÁC NHẤT
    print(f"  ⏱️ TỔNG THỜI GIAN END-TO-END: {total_prog_time:.2f} ms ({1000/total_prog_time:.2f} FPS)")
    print(f"{'─'*50}\n")

    # Lấy luôn kết quả đã lưu trong pipeline, KHÔNG CHẠY LẠI MÔ HÌNH!
    colored_mask = pipeline.last_colored_mask
    depth_raw    = pipeline.last_depth_map
    depth_map    = pipeline.geo.scale_depth_by_focal_length(depth_raw)

    visualize(
        image_bgr    = image,
        colored_mask = colored_mask,
        depth_map    = depth_map,
        detections   = detections,
        save_path    = "output_detection_924x518.jpg",
    )