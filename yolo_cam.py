import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO


BLUR_TYPE_DEFAULT = "gaussian"
BLUR_TYPE_LABELS = {
    "mean": "흐리게",
    "gaussian": "가우시안 블러",
    "motion": "모션 블러",
}


def odd_kernel(value: int) -> int:
    value = max(5, int(value))
    return value if value % 2 == 1 else value + 1


class YOLOBlurCam:
    def __init__(
        self,
        weight_paths,
        source=0,
        width=640,
        height=480,
        imgsz=320,
        infer_interval=2,
    ):
        self.models = [YOLO(p) for p in weight_paths]
        self.source = source

        self.cap = self._open_source(source)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"카메라 오픈 실패: {source}")

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

        self.imgsz = imgsz
        self.infer_interval = infer_interval
        self.frame_id = 0
        self.last_boxes = []

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._reader_running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _open_source(self, source):
        candidates = []
        if isinstance(source, str):
            candidates = [
                lambda: cv2.VideoCapture(source, cv2.CAP_FFMPEG),
                lambda: cv2.VideoCapture(source),
            ]
        else:
            candidates = [
                lambda: cv2.VideoCapture(source, cv2.CAP_DSHOW),
                lambda: cv2.VideoCapture(source),
            ]

        for create_cap in candidates:
            cap = create_cap()
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        return None

    def _reader_loop(self):
        while self._reader_running:
            ok = self.cap.grab()
            if not ok:
                time.sleep(0.01)
                continue

            ok, frame = self.cap.retrieve()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            with self._frame_lock:
                self._latest_frame = frame

    def read_frame(self):
        for _ in range(100):
            with self._frame_lock:
                if self._latest_frame is not None:
                    return self._latest_frame.copy()
            time.sleep(0.01)
        return None

    @staticmethod
    def apply_blur(frame, blur_ksize=31, blur_type=BLUR_TYPE_DEFAULT):
        """선택한 방식으로 frame 전체 또는 ROI를 비식별화한다."""
        k = odd_kernel(blur_ksize)

        if blur_type == "mean":
            # 단순 평균 블러: 주변 픽셀 평균값으로 빠르게 흐리게 처리
            return cv2.blur(frame, (k, k))

        if blur_type == "motion":
            # 모션 블러: 가로 방향으로 움직인 것처럼 번지게 처리
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            return cv2.filter2D(frame, -1, kernel)

        # 기본값: 가우시안 블러
        return cv2.GaussianBlur(frame, (k, k), 0)

    def process(
        self,
        frame,
        conf=0.25,
        blur=True,
        blur_ksize=31,
        blur_type=BLUR_TYPE_DEFAULT,
        margin_ratio=0.10,
        imgsz=None,
        force_infer=False,
    ):
        self.frame_id += 1
        run_imgsz = imgsz or self.imgsz

        if force_infer or self.frame_id % self.infer_interval == 0:
            self.last_boxes = []
            for model in self.models:
                result = model.predict(
                    frame,
                    imgsz=run_imgsz,
                    conf=conf,
                    verbose=False,
                )[0]
                if result.boxes is None:
                    continue
                for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                    names = model.names
                    if isinstance(names, dict):
                        class_name = names.get(int(cls), "").lower()
                    else:
                        class_name = str(names[int(cls)]).lower() if int(cls) < len(names) else ""
                    if class_name in {"face", "license_plate", "plate", "number_plate"}:
                        self.last_boxes.append(box.cpu().numpy())

        if blur and self.last_boxes:
            h, w = frame.shape[:2]
            for x1, y1, x2, y2 in self.last_boxes:
                bw, bh = x2 - x1, y2 - y1
                mx, my = bw * margin_ratio, bh * margin_ratio
                nx1 = max(0, int(x1 - mx))
                ny1 = max(0, int(y1 - my))
                nx2 = min(w, int(x2 + mx))
                ny2 = min(h, int(y2 + my))
                if nx2 <= nx1 or ny2 <= ny1:
                    continue
                roi = frame[ny1:ny2, nx1:nx2]
                if roi.size > 0:
                    frame[ny1:ny2, nx1:nx2] = self.apply_blur(roi, blur_ksize, blur_type)

        return frame, len(self.last_boxes)

    def release(self):
        self._reader_running = False
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self.cap.release()
