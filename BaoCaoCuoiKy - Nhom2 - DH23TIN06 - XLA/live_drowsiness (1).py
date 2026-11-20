#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phát hiện ngủ gật TRỰC TIẾP kết hợp YOLOv8 (tìm khuôn mặt) + MediaPipe FaceMesh (đo độ mở mắt EAR & miệng MAR).

Màu khung (theo trạng thái của TỪNG người):  Normal = XANH LÁ | Yawn = VÀNG | Drowsy = ĐỎ
Thanh trên cùng đổi màu và đổi chữ theo trạng thái nguy hiểm nhất đang có trong khung hình;
khi ngủ gật kéo dài thì chuyển sang DANGER (nhấp nháy đỏ) và kêu chuông.

Cài đặt (một lần):
    pip install ultralytics opencv-python numpy mediapipe

Chạy:
    python "live_drowsiness (1).py" --model best.pt

Phím tắt:
    q hoặc ESC = thoát | c = bật/tắt CLAHE | + / - = chỉnh ngưỡng conf YOLO | m = lật gương
    [ / ] = giảm/tăng ngưỡng nhắm mắt (EAR) | ; / ' = giảm/tăng ngưỡng ngáp (MAR)
    b = ẩn/hiện thanh trên cùng | s = chụp ảnh | r = quay video | d = in gỡ lỗi ra Terminal
"""
import argparse
import glob
import os
import platform
import sys
import threading
import time
import unicodedata
import urllib.request
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    sys.exit('Chưa cài mediapipe. Hãy chạy lệnh:  pip install mediapipe')

# ----------------------------------------------------------------------------------------------------------------------
# Màu (BGR của OpenCV) và tên hiển thị của ba trạng thái
# ----------------------------------------------------------------------------------------------------------------------
COLORS = {'normal': (0, 190, 0), 'yawn': (0, 215, 255), 'drowsy': (0, 0, 255)}        # xanh lá / vàng / đỏ
TEXT_COLORS = {'normal': (255, 255, 255), 'yawn': (0, 0, 0), 'drowsy': (255, 255, 255)}  # chữ trên nền màu
STATE_TEXT = {'normal': 'NORMAL', 'yawn': 'YAWN', 'drowsy': 'DROWSY'}
BANNER_TEXT = {'normal': 'NORMAL', 'yawn': 'YAWNING', 'drowsy': 'DROWSY'}
PRIORITY = {'drowsy': 3, 'yawn': 2, 'normal': 1}
STATES = ('normal', 'yawn', 'drowsy')

ALIASES = {
    'normal': ['normal', 'awake', 'alert', 'binh thuong', 'tinh tao', 'open'],
    'yawn': ['yawn', 'ngap'],
    'drowsy': ['drowsy', 'drowsiness', 'sleepy', 'sleep', 'ngu gat', 'buon ngu', 'nodding', 'closed'],
}

# Chỉ số các điểm mốc khuôn mặt (468 điểm MediaPipe) để tính EAR (mắt) và MAR (miệng)
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
MOUTH_IDX = [78, 81, 13, 311, 308, 402, 14, 178]


class ModelError(Exception):
    pass


def norm_name(s):
    s = unicodedata.normalize('NFD', str(s)).encode('ascii', 'ignore').decode('ascii')
    return s.lower().replace('_', ' ').replace('-', ' ').strip()


def resolve_classes(model_names, override=None):
    """Ánh xạ chỉ số lớp của mô hình -> 'normal' / 'yawn' / 'drowsy'."""
    names = {int(k): str(v) for k, v in dict(model_names).items()}
    shown = ', '.join(f'{k}:{v}' for k, v in list(names.items())[:12]) + (' ...' if len(names) > 12 else '')
    if override:
        o = [x.strip().lower() for x in override.split(',')]
        if sorted(o) != sorted(STATES) or len(names) != 3:
            raise ModelError('--classes phải liệt kê đúng 3 lớp normal,yawn,drowsy theo thứ tự chỉ số lớp của mô hình.')
        return {i: o[i] for i in range(3)}, f'ghép theo --classes: {o}'
    mapping = {}
    for k, v in names.items():
        nv = norm_name(v)
        for canon, al in ALIASES.items():
            if any(a in nv for a in al):
                mapping[k] = canon
                break
    if sorted(mapping.values()) == sorted(STATES):
        return mapping, 'ghép theo tên lớp'
    if len(names) == 3 and all(norm_name(v).isdigit() for v in names.values()):
        return ({i: s for i, s in zip(sorted(names), STATES)},
                'tên lớp là số 0/1/2: giả định thứ tự normal, yawn, drowsy')
    coco = 'person' in [norm_name(v) for v in names.values()] or len(names) >= 20
    hint = ('Đây là mô hình COCO mặc định (yolov8n.pt hoặc tương tự), KHÔNG phải mô hình ngủ gật. ' if coco else '')
    raise ModelError(f'Mô hình này không có đúng 3 lớp normal / yawn / drowsy.\nCác lớp tìm thấy: {shown}\n{hint}'
                     'Hãy dùng tệp best.pt do notebook huấn luyện ra.')


def locate_model(path):
    p = Path(path)
    if p.exists():
        return str(p)
    here = Path(__file__).resolve().parent
    cands = [here / 'best.pt', here / 'models' / 'best.pt', Path.cwd() / 'best.pt', Path.cwd() / 'models' / 'best.pt']
    cands += [Path(x) for x in glob.glob(str(here / '**' / 'weights' / 'best.pt'), recursive=True)]
    cands = [c for c in cands if c.exists()]
    if cands:
        cands.sort(key=lambda c: c.stat().st_mtime, reverse=True)
        print(f'Không thấy "{path}", dùng tệp tìm được: {cands[0]}')
        return str(cands[0])
    raise ModelError(f'Không tìm thấy tệp mô hình "{path}". Chép best.pt vào cùng thư mục với script này.')


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------------------------------------------------------
# Bộ trích xuất điểm mốc MediaPipe & tính chỉ số EAR (mắt) / MAR (miệng)
# ----------------------------------------------------------------------------------------------------------------------
class FaceMeshAnalyzer:
    def __init__(self, max_faces=5):
        self.use_solutions = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh')
        if self.use_solutions:
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_faces,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            task_path = Path(__file__).resolve().parent / 'face_landmarker.task'
            if not task_path.exists():
                url = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/'
                       'face_landmarker/float16/1/face_landmarker.task')
                print('Đang tải tệp hỗ trợ MediaPipe (face_landmarker.task)...')
                urllib.request.urlretrieve(url, str(task_path))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=max_faces,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _eye_ear(pts, idx):
        p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]
        horiz = np.linalg.norm(p1 - p4)
        if horiz < 1e-6:
            return 0.3
        vert = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        return float(vert / (2.0 * horiz))

    @staticmethod
    def _mouth_mar(pts, idx):
        p_left, p_u1, p_u2, p_u3, p_right, p_d3, p_d2, p_d1 = [pts[i] for i in idx]
        horiz = np.linalg.norm(p_left - p_right)
        if horiz < 1e-6:
            return 0.0
        vert = np.linalg.norm(p_u1 - p_d1) + np.linalg.norm(p_u2 - p_d2) + np.linalg.norm(p_u3 - p_d3)
        return float(vert / (2.0 * horiz))

    def analyze(self, frame_bgr):
        """Trả về danh sách dict {'nose': (nx, ny), 'ear': float, 'mar': float} cho từng khuôn mặt."""
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces_lms = []
        if self.use_solutions:
            res = self.mesh.process(rgb)
            if res.multi_face_landmarks:
                for fl in res.multi_face_landmarks:
                    faces_lms.append(fl.landmark)
        else:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect(mp_img)
            if res.face_landmarks:
                faces_lms = res.face_landmarks

        results = []
        for lms in faces_lms:
            pts = np.array([(lm.x * W, lm.y * H) for lm in lms], dtype=np.float32)
            ear = 0.5 * (self._eye_ear(pts, LEFT_EYE_IDX) + self._eye_ear(pts, RIGHT_EYE_IDX))
            mar = self._mouth_mar(pts, MOUTH_IDX)
            nose = (float(pts[1][0]), float(pts[1][1]))
            results.append({'nose': nose, 'ear': ear, 'mar': mar})
        return results


# ----------------------------------------------------------------------------------------------------------------------
# Theo dõi từng khuôn mặt + biểu quyết đa số: chống nhấp nháy màu khung giữa các khung hình
# ----------------------------------------------------------------------------------------------------------------------
class Track:
    _next_id = 1

    def __init__(self, box, label, conf, history):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = tuple(box)
        self.hist = deque([(label, conf)], maxlen=history)
        self.missing = 0
        self.age = 0

    def update(self, box, label, conf):
        self.box = tuple(0.6 * n + 0.4 * o for n, o in zip(box, self.box))
        self.hist.append((label, conf))
        self.missing = 0
        self.age += 1

    @property
    def stable(self):
        counts = Counter(l for l, _ in self.hist)
        label = max(counts, key=lambda l: (counts[l], PRIORITY.get(l, 0)))
        confs = [c for l, c in self.hist if l == label]
        return label, float(np.mean(confs))


class FaceTracker:
    def __init__(self, history=5, iou_thr=0.25, max_missing=8):
        self.tracks, self.history, self.iou_thr, self.max_missing = [], history, iou_thr, max_missing

    def update(self, dets):
        pairs = sorted(((iou(t.box, d[0]), ti, di) for ti, t in enumerate(self.tracks) for di, d in enumerate(dets)), reverse=True)
        used_t, used_d = set(), set()
        for v, ti, di in pairs:
            if v < self.iou_thr:
                break
            if ti in used_t or di in used_d:
                continue
            self.tracks[ti].update(*dets[di])
            used_t.add(ti)
            used_d.add(di)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missing += 1
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(d[0], d[1], d[2], self.history))
        self.tracks = [t for t in self.tracks if t.missing <= self.max_missing]
        return sorted((t for t in self.tracks if t.missing == 0), key=lambda t: t.box[0])


# ----------------------------------------------------------------------------------------------------------------------
# Logic cảnh báo theo thời gian (tính theo GIÂY, không phụ thuộc FPS của camera)
# ----------------------------------------------------------------------------------------------------------------------
class TimeAlert:
    def __init__(self, window_s=3.0, drowsy_ratio=0.6, yawn_window_s=60.0, yawn_needed=3, yawn_gap_s=2.0, exit_s=2.0, exit_ratio=0.8):
        self.window_s, self.drowsy_ratio = window_s, drowsy_ratio
        self.yawn_window_s, self.yawn_needed, self.yawn_gap_s = yawn_window_s, yawn_needed, yawn_gap_s
        self.exit_s, self.exit_ratio = exit_s, exit_ratio
        self.samples = deque()
        self.yawn_events = deque()
        self.last_label, self.last_yawn_t = None, -1e9
        self.level = 'NORMAL'

    def update(self, label, t):
        self.samples.append((t, label))
        while self.samples and t - self.samples[0][0] > max(self.window_s, self.exit_s):
            self.samples.popleft()
        if label == 'yawn' and self.last_label != 'yawn' and t - self.last_yawn_t >= self.yawn_gap_s:
            self.yawn_events.append(t)
            self.last_yawn_t = t
        self.last_label = label
        while self.yawn_events and t - self.yawn_events[0] > self.yawn_window_s:
            self.yawn_events.popleft()

        win = [l for ts, l in self.samples if t - ts <= self.window_s]
        ratio = sum(1 for l in win if l == 'drowsy') / max(1, len(win))
        covered = (t - self.samples[0][0]) >= 0.8 * self.window_s
        n_yawn = len(self.yawn_events)
        if covered and ratio >= self.drowsy_ratio:
            self.level = 'DANGER'
        elif self.level == 'DANGER':
            ex = [l for ts, l in self.samples if t - ts <= self.exit_s]
            if ex and sum(1 for l in ex if l == 'normal') / len(ex) >= self.exit_ratio and (t - self.samples[0][0]) >= 0.8 * self.exit_s:
                self.level = 'WARNING' if n_yawn >= self.yawn_needed else 'NORMAL'
        elif n_yawn >= self.yawn_needed:
            self.level = 'WARNING'
        else:
            self.level = 'NORMAL'
        return self.level, ratio, n_yawn


class Alarm:
    def __init__(self, enabled=True):
        self.enabled, self.last = enabled, 0.0

    def ring(self):
        if not self.enabled or time.time() - self.last < 1.0:
            return
        self.last = time.time()
        threading.Thread(target=self._beep, daemon=True).start()

    @staticmethod
    def _beep():
        try:
            system = platform.system()
            if system == 'Windows':
                import winsound
                winsound.Beep(1200, 400)
            elif system == 'Darwin':
                import subprocess
                subprocess.run(['afplay', '/System/Library/Sounds/Sosumi.aiff'], check=False)
            else:
                sys.stdout.write('\a')
                sys.stdout.flush()
        except Exception:
            pass


# ----------------------------------------------------------------------------------------------------------------------
# Vẽ giao diện
# ----------------------------------------------------------------------------------------------------------------------
def text_size(text, fs, th=2):
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    return tw, tht, base


def draw_face(img, t, top_limit):
    label, conf = t.stable
    col, tcol = COLORS[label], TEXT_COLORS[label]
    H, W = img.shape[:2]
    x1, y1, x2, y2 = t.box
    g = 0.7 + 0.3 * min(1.0, t.age / 5.0)
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * g, (y2 - y1) * g
    x1, y1, x2, y2 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2)), int(min(W - 1, cx + w / 2)), int(min(H - 1, cy + h / 2))
    bw = max(1, x2 - x1)
    lw = max(3, W // 280)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, lw)
    text = f'#{t.id} {STATE_TEXT[label]} {conf:.0%}'
    fs = float(np.clip(bw / 250.0, 0.55, 1.1))
    tw, tht, base = text_size(text, fs)
    band_w, band_h = max(bw, tw + 14), tht + base + 12
    by1 = y1 - band_h
    if by1 < top_limit:
        by1 = max(y1, top_limit)
    bx2 = min(W - 1, x1 + band_w)
    cv2.rectangle(img, (x1, by1), (bx2, by1 + band_h), col, -1)
    cv2.putText(img, text, (x1 + 7, by1 + tht + 5), cv2.FONT_HERSHEY_SIMPLEX, fs, tcol, 2, cv2.LINE_AA)


def draw_banner(img, worst, n_faces, level, blink):
    H, W = img.shape[:2]
    bar_h = max(40, H // 12)
    if worst is None:
        col, tcol, msg = (110, 110, 110), (255, 255, 255), 'NO FACE DETECTED'
    else:
        col, tcol = COLORS[worst], TEXT_COLORS[worst]
        msg = f'{BANNER_TEXT[worst]}  ({n_faces} {"person" if n_faces == 1 else "people"})'
        if level == 'DANGER':
            col, tcol, msg = ((0, 0, 255) if blink else (0, 0, 130)), (255, 255, 255), 'DROWSY - DANGER! WAKE UP!'
        elif level == 'WARNING' and worst != 'drowsy':
            msg += '  - frequent yawning'
    cv2.rectangle(img, (0, 0), (W, bar_h), col, -1)
    cv2.putText(img, msg, (14, int(bar_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, max(0.8, W / 1000), tcol, 2, cv2.LINE_AA)
    return bar_h


def draw_legend(img, counts):
    H, W = img.shape[:2]
    x0, y0, w, h = 10, H - 36 - 3 * 26 - 14, 190, 3 * 26 + 12
    roi = img[y0:y0 + h, x0:x0 + w]
    img[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(roi, 0.45, np.zeros_like(roi), 0.55, 0)
    for i, s in enumerate(STATES):
        y = y0 + 24 + i * 26
        cv2.rectangle(img, (x0 + 8, y - 14), (x0 + 28, y + 4), COLORS[s], -1)
        cv2.putText(img, f'{s.capitalize()}: {counts.get(s, 0)}', (x0 + 38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_info(img, fps, ear, mar, ear_thr, mar_thr, ratio, n_yawn, recording):
    H, W = img.shape[:2]
    info = (f'FPS {fps:4.1f} | EAR {ear:.2f} (<{ear_thr:.2f}) | MAR {mar:.2f} (>{mar_thr:.2f}) | '
            f'drowsy {ratio:.0%} | yawns/60s {n_yawn}')
    fs = max(0.48, W / 1900)
    cv2.putText(img, info, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, info, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    if recording:
        cv2.circle(img, (W - 30, 70), 10, (0, 0, 255), -1)


def apply_clahe(img_bgr, clip=2.0, tile=(8, 8)):
    h, s, v = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
    v = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


# ----------------------------------------------------------------------------------------------------------------------
# Bộ xử lý một khung hình kết hợp YOLOv8 + MediaPipe
# ----------------------------------------------------------------------------------------------------------------------
class LiveDetector:
    def __init__(self, model, class_map, conf=0.40, imgsz=640, device='cpu', smooth=5, use_clahe=False,
                 alarm=True, show_banner=True, debug=False, ear_thr=0.21, mar_thr=0.50):
        self.model, self.class_map, self.conf, self.imgsz, self.device = model, class_map, conf, imgsz, device
        self.tracker, self.alert, self.alarm = FaceTracker(history=smooth), TimeAlert(), Alarm(alarm)
        self.mesh_analyzer = FaceMeshAnalyzer(max_faces=5)
        self.use_clahe, self.show_banner, self.debug = use_clahe, show_banner, debug
        self.ear_thr = ear_thr
        self.mar_thr = mar_thr
        self.last_ear = 0.0
        self.last_mar = 0.0
        self.fps = 0.0
        self.prev = time.time()
        self.t0 = time.time()
        self.recording = False
        self.frame_idx = 0

    def classify_by_landmarks(self, ear, mar):
        # Ưu tiên bắt ngáp trước (khi ngáp mắt cũng thường nheo lại)
        if mar > self.mar_thr:
            conf = min(0.99, 0.65 + (mar - self.mar_thr) * 0.8)
            return 'yawn', conf
        if ear < self.ear_thr:
            conf = min(0.99, 0.65 + (self.ear_thr - ear) * 2.5)
            return 'drowsy', conf
        conf = min(0.99, 0.65 + (ear - self.ear_thr) * 2.0)
        return 'normal', conf

    def detect(self, frame):
        proc = apply_clahe(frame) if self.use_clahe else frame
        res = self.model.predict(proc, conf=self.conf, iou=0.5, imgsz=self.imgsz, device=self.device,
                                 agnostic_nms=True, verbose=False)[0]
        mp_faces = self.mesh_analyzer.analyze(proc)

        dets = []
        for b in res.boxes:
            idx = int(b.cls[0]) if hasattr(b.cls, '__len__') else int(b.cls)
            if idx not in self.class_map:
                continue
            yolo_conf = float(b.conf[0]) if hasattr(b.conf, '__len__') else float(b.conf)
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]

            # Ghép kết quả MediaPipe nằm trong khung khuôn mặt của YOLOv8
            matched = None
            best_dist = 1e9
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            for mf in mp_faces:
                nx, ny = mf['nose']
                if (x1 - 20) <= nx <= (x2 + 20) and (y1 - 20) <= ny <= (y2 + 20):
                    d = (nx - cx) ** 2 + (ny - cy) ** 2
                    if d < best_dist:
                        best_dist = d
                        matched = mf

            if matched is not None:
                self.last_ear = matched['ear']
                self.last_mar = matched['mar']
                label, conf = self.classify_by_landmarks(matched['ear'], matched['mar'])
            else:
                label, conf = self.class_map[idx], yolo_conf

            dets.append(((x1, y1, x2, y2), label, conf))
        return dets

    def step(self, frame, now=None):
        now = time.time() if now is None else now
        dets = self.detect(frame)
        if self.debug and self.frame_idx % 10 == 0:
            print(f'[khung {self.frame_idx}] EAR={self.last_ear:.2f} MAR={self.last_mar:.2f} ->',
                  [(l, round(c, 2)) for _, l, c in dets])
        visible = self.tracker.update(dets)
        worst = max((t.stable[0] for t in visible), key=lambda l: PRIORITY[l], default=None)
        level, ratio, n_yawn = self.alert.update(worst, now - self.t0)
        if level == 'DANGER':
            self.alarm.ring()
        out = frame.copy()
        counts = Counter(t.stable[0] for t in visible)
        bar_h = draw_banner(out, worst, len(visible), level, blink=int(now * 4) % 2 == 0) if self.show_banner else 0
        for t in visible:
            draw_face(out, t, top_limit=bar_h)
        draw_legend(out, counts)
        inst = 1 / max(now - self.prev, 1e-6)
        self.fps = 0.9 * self.fps + 0.1 * inst if self.fps else inst
        self.prev = now
        draw_info(out, self.fps, self.last_ear, self.last_mar, self.ear_thr, self.mar_thr, ratio, n_yawn, self.recording)
        self.frame_idx += 1
        return out


# ----------------------------------------------------------------------------------------------------------------------
def open_source(source, width, height):
    if str(source).isdigit():
        idx = int(source)
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if platform.system() == 'Windows' else [cv2.CAP_ANY]
        for cand in [idx] + [i for i in range(4) if i != idx]:
            for be in backends:
                cap = cv2.VideoCapture(cand, be)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        if cand != idx:
                            print(f'Không mở được camera {idx}, đang dùng camera {cand}.')
                        if width and height:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        return cap, True
                cap.release()
        return None, True
    cap = cv2.VideoCapture(source)
    return (cap if cap.isOpened() else None), False


def pick_device(arg):
    if arg != 'auto':
        return arg
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def main():
    ap = argparse.ArgumentParser(description='Phát hiện ngủ gật trực tiếp bằng YOLOv8 + MediaPipe')
    ap.add_argument('--model', default='best.pt', help='tệp trọng số best.pt')
    ap.add_argument('--classes', default='', help='thứ tự lớp nếu cần')
    ap.add_argument('--source', default='0', help='số thứ tự camera (0, 1, ...), đường dẫn video hoặc URL camera IP')
    ap.add_argument('--conf', type=float, default=0.40, help='ngưỡng độ tin cậy YOLO')
    ap.add_argument('--ear', type=float, default=0.21, help='ngưỡng độ mở mắt EAR (nhỏ hơn = nhắm mắt/ngủ gật)')
    ap.add_argument('--mar', type=float, default=0.50, help='ngưỡng độ mở miệng MAR (lớn hơn = ngáp)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='auto', help='auto | cpu | 0 | mps')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--smooth', type=int, default=5, help='số khung biểu quyết nhãn ổn định')
    ap.add_argument('--clahe', action='store_true', help='bật CLAHE ngay từ đầu')
    ap.add_argument('--no-mirror', action='store_true', help='không lật gương hình camera')
    ap.add_argument('--no-alarm', action='store_true', help='tắt chuông cảnh báo')
    ap.add_argument('--no-banner', action='store_true', help='ẩn thanh trạng thái trên cùng')
    ap.add_argument('--debug', action='store_true', help='in thông số EAR/MAR ra màn hình lệnh')
    ap.add_argument('--out', default='', help='tên tệp mp4 để ghi video kết quả')
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit('Chưa cài ultralytics. Chạy:  pip install ultralytics opencv-python')
    try:
        model_path = locate_model(args.model)
        model = YOLO(model_path)
        class_map, how = resolve_classes(model.names, args.classes or None)
    except ModelError as e:
        sys.exit(f'LỖI MÔ HÌNH: {e}')
    print(f'Mô hình: {model_path}')
    print(f'Chế độ: YOLOv8 (tìm khuôn mặt) + MediaPipe FaceMesh (EAR < {args.ear}, MAR > {args.mar})')

    device = pick_device(args.device)
    cap, is_cam = open_source(args.source, args.width, args.height)
    if cap is None:
        sys.exit('Không mở được camera/video. Kiểm tra quyền camera hoặc thử --source 1.')

    det = LiveDetector(model, class_map, conf=args.conf, imgsz=args.imgsz, device=device, smooth=args.smooth,
                       use_clahe=args.clahe, alarm=not args.no_alarm, show_banner=not args.no_banner,
                       debug=args.debug, ear_thr=args.ear, mar_thr=args.mar)
    mirror = is_cam and not args.no_mirror
    writer = None
    win = 'Phat hien ngu gat - YOLOv8 + MediaPipe (q: thoat)'
    print('Đang chạy. Nhấn q hoặc ESC trên cửa sổ video để thoát.')
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8), imgsz=args.imgsz, device=device, verbose=False)

    while True:
        ok, frame = cap.read()
        if not ok:
            print('Hết video hoặc mất kết nối camera.')
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        out = det.step(frame)
        if det.recording and writer is not None:
            writer.write(out)
        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            det.use_clahe = not det.use_clahe
        elif key in (ord('+'), ord('=')):
            det.conf = min(0.95, det.conf + 0.05)
        elif key in (ord('-'), ord('_')):
            det.conf = max(0.05, det.conf - 0.05)
        elif key == ord('['):
            det.ear_thr = max(0.10, round(det.ear_thr - 0.01, 2))
        elif key == ord(']'):
            det.ear_thr = min(0.35, round(det.ear_thr + 0.01, 2))
        elif key == ord(';'):
            det.mar_thr = max(0.25, round(det.mar_thr - 0.05, 2))
        elif key == ord("'"):
            det.mar_thr = min(0.90, round(det.mar_thr + 0.05, 2))
        elif key == ord('m'):
            mirror = not mirror
        elif key == ord('b'):
            det.show_banner = not det.show_banner
        elif key == ord('d'):
            det.debug = not det.debug
        elif key == ord('s'):
            name = time.strftime('snapshot_%Y%m%d_%H%M%S.jpg')
            cv2.imwrite(name, out)
            print('Đã lưu', name)
        elif key == ord('r'):
            if not det.recording:
                path = args.out or time.strftime('record_%Y%m%d_%H%M%S.mp4')
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
                det.recording = True
                print('Bắt đầu ghi:', path)
            else:
                det.recording = False
                writer.release()
                writer = None
                print('Đã dừng ghi.')
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()