"""
PipelineManager
===============
Kết nối Segmentor · DepthEstimator · GeometryEngine
theo đúng luồng pipeline đã thiết kế.

Luồng xử lý:
  img_bgr
    │
    ├─[Step 2] Segmentor.predict()
    │           → colored_mask (BGR H×W×3)
    │           → non_bg_mask  (bool H×W)
    │
    ├─[Step 3] DepthEstimator.predict(img, K=K)
    │           → depth_map_real (float32 H×W, metric mét, FOV đã căn chỉnh nội bộ)
    │
    ├─[Step 4] GeometryEngine
    │           → alpha_mask (Convex Hull lấp lỗ YOLO)
    │           → obstacle_flags (Surface Normals > 60°)
    │
    └─[Step 5] Lọc OOD + BBox + Distance
                → List[Detection] sorted by distance_m ASC
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
    label      : str                        # class name hoặc "OOD"
    bbox       : Tuple[int, int, int, int]  # (x, y, w, h) pixel
    distance_m : float                      # khoảng cách thực tế (mét)
    mask       : Optional[np.ndarray] = field(default=None, repr=False)
    # mask: HxW uint8 binary — dùng nội bộ, không export ra JSON


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class PipelineManager:
    """
    Tham số khởi tạo
    ─────────────────
    seg_weight   : path tới weight YOLOv8-seg (.pt)
    depth_weight : path tới weight DepthAnythingV2 (.pth)
    K            : Camera Intrinsics (3×3 float64)
    E_v2c        : Extrinsics vehicle→camera (4×4 float64)
    device       : "cuda" | "cpu"
    max_depth    : ngưỡng depth tối đa (mét)
    angle_thr_deg: ngưỡng góc pháp tuyến để coi là vật đứng thẳng (độ)
    conf_threshold: ngưỡng confidence của YOLO
    """

    # ── Class phân loại ───────────────────────────────────────────────────────
    # Background thật sự (nền cứng) → dùng để build non_bg_mask
    # LƯU Ý: KHÔNG đưa "background" vào đây —
    #   pixel màu background = YOLO không nhận ra = ứng viên OOD, phải GIỮ LẠI
    BG_CLASSES = {"road", "sidewalk", "building"}
    # Known obstacle → đối chiếu khi gán nhãn
    OBSTACLE_CLASSES = {"2_wheel", "4_wheel", "person"}

    # ── Ngưỡng lọc ────────────────────────────────────────────────────────────
    MIN_AREA_KNOWN_PX = 500     # diện tích tối thiểu (pixel²) cho known class
    MIN_AREA_OOD_PX   = 1500    # diện tích tối thiểu cho OOD (tránh nhiễu nhỏ)
    MAX_AREA_PX       = 50000   # loại vùng quá lớn (thường là nhiễu nền rộng)
    MATCH_OVERLAP_RATIO = 0.25  # IoU overlap tối thiểu để coi là "đã biết"
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
    ):
        self.K             = K
        self.angle_thr_deg = angle_thr_deg
        self.conf_threshold = conf_threshold

        print("[PipelineManager] Khởi tạo các module...")
        self.segmentor  = Segmentor(seg_weight)
        self.depth_est  = DepthEstimator(depth_weight, device=device, max_depth=max_depth)
        self.geo        = GeometryEngine(K, E_v2c)
        print("[PipelineManager] Sẵn sàng.")

    # ── Helpers Step 2 ────────────────────────────────────────────────────────

    def _color_to_binary(self, colored_mask_bgr: np.ndarray,
                         color_bgr: tuple) -> np.ndarray:
        """Tách một class ra khỏi colored_mask theo màu BGR → binary mask."""
        c = np.array(color_bgr, dtype=np.uint8)
        return cv2.inRange(colored_mask_bgr, c, c)  # 0 hoặc 255

    def _build_non_bg_mask(self, colored_mask_bgr: np.ndarray) -> np.ndarray:
        """
        Gom tất cả class background → đảo bit → non_bg_mask.
        Pixel=255: vùng có thể là vật thể.
        """
        H, W   = colored_mask_bgr.shape[:2]
        bg_acc = np.zeros((H, W), dtype=np.uint8)

        for cls in self.BG_CLASSES:
            color = self.segmentor.bgr_colors.get(cls)
            if color is None:
                continue
            layer  = self._color_to_binary(colored_mask_bgr, color)
            bg_acc = cv2.bitwise_or(bg_acc, layer)

        return cv2.bitwise_not(bg_acc)   # non_bg_mask (HxW, uint8 255/0)

    def _build_known_masks(self, colored_mask_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """Trả về dict {class_name: binary_mask} cho từng OBSTACLE class."""
        out = {}
        for cls in self.OBSTACLE_CLASSES:
            color = self.segmentor.bgr_colors.get(cls)
            if color is None:
                continue
            out[cls] = self._color_to_binary(colored_mask_bgr, color)  # uint8 255/0
        return out

    def _build_road_mask(self, colored_mask_bgr: np.ndarray) -> np.ndarray:
        """Binary mask của class 'road' → dùng cho Convex Hull."""
        color = self.segmentor.bgr_colors.get("road")
        if color is None:
            return np.zeros(colored_mask_bgr.shape[:2], dtype=bool)
        binary = self._color_to_binary(colored_mask_bgr, color)
        return binary > 0  # bool

    # ── Helper Step 5 ─────────────────────────────────────────────────────────

    def _get_distance(self, depth_map: np.ndarray,
                      instance_mask: np.ndarray,
                      bbox: Tuple) -> float:
        """
        Khoảng cách đại diện của một instance.

        Dùng nửa dưới của bbox (thân xe, gầm, lốp) vì:
        - Phần trên thường là kính/bầu trời → depth không đáng tin
        - Nửa dưới là khối đặc → np.min() cho khoảng cách sát nhất

        Notebook cell 3: Z_c_roi = depth_map[y_lower:y+h, x:x+w][roi_mask]
                         real_distance = np.min(Z_c_roi)
        """
        x, y, w, h = bbox

        # Nửa dưới bbox
        y_lower     = y + h // 2
        roi_depth   = depth_map[y_lower:y+h, x:x+w]
        roi_mask    = instance_mask[y_lower:y+h, x:x+w].astype(bool)

        valid = roi_depth[roi_mask]
        valid = valid[(valid > self.MIN_DEPTH_M) & (valid < self.MAX_DEPTH_M)]

        return float(np.min(valid)) if len(valid) else -1.0

    def _match_label(self, instance_mask: np.ndarray,
                     known_masks: Dict[str, np.ndarray],
                     area: int) -> str:
        """
        So khớp instance với YOLO class bằng pixel overlap.
        Trả về tên class nếu overlap >= MATCH_OVERLAP_RATIO × area,
        ngược lại trả về "OOD".
        """
        best_label   = "OOD"
        best_overlap = 0
        inst_bool    = instance_mask.astype(bool)

        for cls_name, cls_mask in known_masks.items():
            overlap = int(np.count_nonzero(inst_bool & (cls_mask > 0)))
            if overlap > best_overlap and overlap >= area * self.MATCH_OVERLAP_RATIO:
                best_overlap = overlap
                best_label   = cls_name

        return best_label

    # ── MAIN RUN ──────────────────────────────────────────────────────────────

    def run(self, image_bgr: np.ndarray) -> List[Detection]:
        """
        Chạy toàn bộ pipeline trên một frame kèm Profiling thời gian.
        """
        profiling_times = {}
        total_start = time.perf_counter()

        # ── STEP 2: Semantic ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        print("[Step 2] Semantic segmentation...")
        colored_mask = self.segmentor.predict(
            image_bgr, conf_threshold=self.conf_threshold
        )                                              # BGR H×W×3 uint8

        non_bg_mask  = self._build_non_bg_mask(colored_mask)    # uint8 H×W
        known_masks  = self._build_known_masks(colored_mask)     # dict
        road_mask    = self._build_road_mask(colored_mask)       # bool H×W
        
        profiling_times['Step 2 (Semantic & Masks)'] = (time.perf_counter() - t0) * 1000

        # ── STEP 3: Depth estimation ──────────────────────────────────────────
        t0 = time.perf_counter()
        print("[Step 3] Depth estimation...")
        depth_map_real = self.depth_est.predict(image_bgr, K=self.K)  # float32 H×W
        
        profiling_times['Step 3 (Depth Estimation)'] = (time.perf_counter() - t0) * 1000

        # ── STEP 4: Geometry Engine ───────────────────────────────────────────
        t0 = time.perf_counter()
        print("[Step 4] Geometry / Surface Normals...")

        alpha_mask = self.geo.get_alpha_shape_mask_cv2(road_mask)   # bool H×W
        obstacle_flags, _ = self.geo.get_obstacle_mask_normals_numpy(
            depth_map_real, alpha_mask, self.angle_thr_deg
        )                                                            # bool H×W
        
        profiling_times['Step 4 (Geometry Engine)'] = (time.perf_counter() - t0) * 1000

        # ── STEP 5: Lọc OOD + BBox + Distance ────────────────────────────────
        t0 = time.perf_counter()
        print("[Step 5] OOD filtering + BBox + Distance...")

        filtered = (obstacle_flags & (non_bg_mask > 0)).astype(np.uint8) * 255

        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN,  kernel)

        num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
            filtered, connectivity=8
        )

        detections: List[Detection] = []

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < self.MIN_AREA_KNOWN_PX or area > self.MAX_AREA_PX:
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            bbox          = (x, y, w, h)
            instance_mask = (labels_map == i).astype(np.uint8)

            label = self._match_label(instance_mask, known_masks, area)

            if label == "OOD" and area < self.MIN_AREA_OOD_PX:
                continue

            dist = self._get_distance(depth_map_real, instance_mask, bbox)
            if dist < 0:
                continue

            detections.append(Detection(
                label      = label,
                bbox       = bbox,
                distance_m = round(dist, 2),
                mask       = instance_mask,
            ))

        detections.sort(key=lambda d: d.distance_m)
        profiling_times['Step 5 (Filtering & BBox)'] = (time.perf_counter() - t0) * 1000

        # ── IN BÁO CÁO THỜI GIAN ─────────────────────────────────────────────
        total_time = (time.perf_counter() - total_start) * 1000
        print(f"\n{'='*50}")
        print(f"⏱️  PIPELINE TIMING REPORT (ms)")
        print(f"{'-'*50}")
        for step_name, t_ms in profiling_times.items():
            print(f"{step_name:<28} : {t_ms:>7.2f} ms")
        print(f"{'-'*50}")
        print(f"{'TOTAL PIPELINE':<28} : {total_time:>7.2f} ms ({1000/total_time:.2f} FPS)")
        print(f"{'='*50}\n")

        print(f"[PipelineManager] Kết quả: {len(detections)} vật thể "
              f"({sum(1 for d in detections if d.label == 'OOD')} OOD)")
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

def draw_detections(image_bgr: np.ndarray,
                    detections: List[Detection]) -> np.ndarray:
    """Vẽ bbox + label + distance lên ảnh gốc."""
    vis = image_bgr.copy()
    for det in detections:
        x, y, w, h = det.bbox
        color       = _LABEL_COLORS_BGR.get(det.label, (180, 180, 180))
        text        = f"{det.label}  {det.distance_m:.1f}m"

        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(vis, text, (x + 3, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def export_json(detections: List[Detection]) -> List[dict]:
    """Serialize kết quả ra list dict (JSON-safe)."""
    return [
        {
            "label"     : d.label,
            "bbox"      : {"x": d.bbox[0], "y": d.bbox[1],
                           "w": d.bbox[2], "h": d.bbox[3]},
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
    """
    Hiển thị 5 panel debug trực tiếp bằng matplotlib:
      [0] Ảnh gốc (RGB)
      [1] Colored mask (YOLO semantic)
      [2] Depth map (colormap plasma)
      [3] Detection result: bbox + label + distance
      [4] Distance bar chart (top-N vật thể gần nhất)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    # ── Chuẩn bị dữ liệu ────────────────────────────────────────────────────
    img_rgb  = cv2.cvtColor(image_bgr,   cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)

    # Depth: clip + normalize để hiển thị đẹp
    depth_vis = np.clip(depth_map, 0, 60)

    # Detection result image (vẽ bbox lên ảnh gốc)
    det_img  = cv2.cvtColor(draw_detections(image_bgr, detections), cv2.COLOR_BGR2RGB)

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 10), facecolor="#0f0f0f")
    fig.suptitle(
        "CV Pipeline — Debug View",
        fontsize=16, fontweight="bold", color="white", y=0.98
    )

    # GridSpec: hàng trên 4 panel ngang, hàng dưới 1 panel bar chart
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1.2],
                          hspace=0.35, wspace=0.08,
                          left=0.03, right=0.97, top=0.93, bottom=0.04)

    ax_img   = fig.add_subplot(gs[0, 0])   # Ảnh gốc
    ax_seg   = fig.add_subplot(gs[0, 1])   # Colored mask
    ax_dep   = fig.add_subplot(gs[0, 2])   # Depth map
    ax_det   = fig.add_subplot(gs[0, 3])   # Detection result
    ax_bar   = fig.add_subplot(gs[1, :])   # Bar chart full width

    panel_style = dict(facecolor="#1a1a1a", frameon=True)
    label_kw    = dict(fontsize=10, color="#aaaaaa",
                       pad=6, loc="left", fontweight="bold")

    # ── Panel 0: Ảnh gốc ────────────────────────────────────────────────────
    ax_img.imshow(img_rgb)
    ax_img.set_title("① Input  img_rgb", **label_kw)
    ax_img.axis("off")
    ax_img.set_facecolor("#1a1a1a")

    # ── Panel 1: Colored mask ────────────────────────────────────────────────
    ax_seg.imshow(mask_rgb)
    ax_seg.set_title("② colored_mask  (YOLO Semantic)", **label_kw)
    ax_seg.axis("off")

    # ── Panel 2: Depth map ───────────────────────────────────────────────────
    im = ax_dep.imshow(depth_vis, cmap="plasma", vmin=0, vmax=60)
    ax_dep.set_title("③ depth_map_real  (0 – 60 m)", **label_kw)
    ax_dep.axis("off")
    cbar = fig.colorbar(im, ax=ax_dep, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("mét", color="#aaaaaa", fontsize=8)

    # ── Panel 3: Detection result ────────────────────────────────────────────
    ax_det.imshow(det_img)
    ax_det.set_title("④ Detections  (label + dist)", **label_kw)
    ax_det.axis("off")

    # ── Panel 4: Distance bar chart ──────────────────────────────────────────
    ax_bar.set_facecolor("#111111")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#333333")

    _MPLCOLORS = {
        "person" : "#00c864",
        "2_wheel": "#ffd700",
        "4_wheel": "#00aaff",
        "OOD"    : "#ff3333",
    }
    TOP_N = 15
    top   = detections[:TOP_N]

    if top:
        labels  = [f"{d.label}\n({d.distance_m:.1f}m)" for d in top]
        dists   = [d.distance_m for d in top]
        colors  = [_MPLCOLORS.get(d.label, "#888888") for d in top]

        bars = ax_bar.barh(
            range(len(top)), dists, color=colors,
            height=0.6, edgecolor="#222222", linewidth=0.4
        )
        ax_bar.set_yticks(range(len(top)))
        ax_bar.set_yticklabels(labels, fontsize=8.5, color="white")
        ax_bar.set_xlabel("Khoảng cách (mét)", color="#aaaaaa", fontsize=9)
        ax_bar.set_title(
            f"⑤ Distance ranking  (top {len(top)} vật thể gần nhất)",
            **label_kw
        )
        ax_bar.tick_params(axis="x", colors="#aaaaaa", labelsize=8)
        ax_bar.invert_yaxis()  # gần nhất ở trên

        # Gắn nhãn số trên mỗi bar
        for bar, dist in zip(bars, dists):
            ax_bar.text(
                dist + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{dist:.1f} m", va="center",
                fontsize=8, color="white"
            )

        # Legend
        legend_handles = [
            mpatches.Patch(color=c, label=lbl)
            for lbl, c in _MPLCOLORS.items()
        ]
        ax_bar.legend(
            handles=legend_handles, loc="lower right",
            fontsize=8, facecolor="#222222", edgecolor="#444444",
            labelcolor="white"
        )
    else:
        ax_bar.text(0.5, 0.5, "Không phát hiện vật thể nào",
                    ha="center", va="center",
                    color="#666666", fontsize=12,
                    transform=ax_bar.transAxes)

    # ── Lưu + hiển thị ──────────────────────────────────────────────────────
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[Visualize] Đã lưu → {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2 as _cv2

    # ── Thông số camera thực tế CAM_P_F (từ notebook cell 2) ─────────────────
    K = np.array([
        [1837.189936,    0.000000, 912.272553],
        [   0.000000, 1844.888039, 728.975910],
        [   0.000000,    0.000000,   1.000000],
    ], dtype=np.float64)

    # Ma trận ngoại vehicle→camera thực tế (từ notebook cell 2)
    E_v2c = np.array([
        [ 0.050593, -0.998719,  0.000873, -0.067766],
        [ 4.4e-05,  -0.000872, -1.000000,  1.880442],
        [ 0.998719,  0.050593,  0.000000, -1.305326],
        [ 0.000000,  0.000000,  0.000000,  1.000000],
    ], dtype=np.float64)

    # ── Khởi tạo pipeline ────────────────────────────────────────────────────
    pipeline = PipelineManager(
        seg_weight    = "weights/best_yolov8l.pt",
        depth_weight  = "weights/depth_anything_v2_metric_vkitti_vits.pth",
        K             = K,
        E_v2c         = E_v2c,
        device        = "cuda",
        max_depth     = 60.0,
        angle_thr_deg = 50.0,
        conf_threshold= 0.2,   # notebook cell 1: conf_threshold=0.2
    )

    # ── Load ảnh ─────────────────────────────────────────────────────────────
    IMG_PATH = "datasets/dataset_ObstacleTrack/images/validation_40.webp"
    image = _cv2.imread(IMG_PATH)
    assert image is not None, f"Không đọc được ảnh: {IMG_PATH}"

    # ── Chạy pipeline (lấy thêm intermediate outputs để visualize) ───────────
    colored_mask = pipeline.segmentor.predict(image, conf_threshold=pipeline.conf_threshold)
    depth_raw    = pipeline.depth_est.predict(image, K=pipeline.K)
    depth_map    = pipeline.geo.scale_depth_by_focal_length(depth_raw)
    detections   = pipeline.run(image)

    # ── In kết quả ra terminal ────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Tổng: {len(detections)} vật thể")
    print(f"{'─'*50}")
    for d in detections:
        print(f"  [{d.label:8s}]  bbox={d.bbox}  dist={d.distance_m:.2f}m")
    print(f"{'─'*50}\n")

    # ── Visualize 5 panel ─────────────────────────────────────────────────────
    visualize(
        image_bgr    = image,
        colored_mask = colored_mask,
        depth_map    = depth_map,
        detections   = detections,
        save_path    = "output_detection.jpg",
    )

    # ── Export JSON ───────────────────────────────────────────────────────────
    print(json.dumps(export_json(detections), ensure_ascii=False, indent=2))