import os
import sys
import time
import cv2
import pyzipper
import secrets
import string
import threading
from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QAction, QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from yolo_cam import YOLOBlurCam


QUALITY_DEFAULT = 70
BLUR_DEFAULT = 151
BLUR_TYPE_DEFAULT = "gaussian"
BLUR_TYPE_LABELS = {
    "mean": "흐리게",
    "gaussian": "가우시안 블러",
    "motion": "모션 블러",
}


def generate_key(length=8):
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))


def odd_kernel(value: int) -> int:
    value = max(5, int(value))
    return value if value % 2 == 1 else value + 1


APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = [
    os.path.join(APP_DIR, "best1.pt"),
    os.path.join(APP_DIR, "best2.pt"),
    os.path.join(APP_DIR, "best3.pt"),
]

DEFAULT_RTSP_URL = os.environ.get(
    "CAMERA_SOURCE",
    "0",
)

BASE_DIR = os.path.join(APP_DIR, "records")
BLUR_DIR = os.path.join(BASE_DIR, "blur")
ORG_DIR = os.path.join(BASE_DIR, "original_enc")
KEY_DIR = os.path.join(BASE_DIR, "keys")
KEY_ZIP_PASSWORD = "cabston!"
os.makedirs(BLUR_DIR, exist_ok=True)
os.makedirs(ORG_DIR, exist_ok=True)
os.makedirs(KEY_DIR, exist_ok=True)


class VideoWorker(QThread):
    frame_ready = Signal(QImage)
    stats_ready = Signal(object)
    error_text = Signal(str)
    info_text = Signal(str)

    def __init__(self, weight_paths, rtsp_url, conf=0.25, quality_percent=QUALITY_DEFAULT, blur_strength=BLUR_DEFAULT, blur_type=BLUR_TYPE_DEFAULT, parent=None):
        super().__init__(parent)
        self.weight_paths = weight_paths
        self.rtsp_url = rtsp_url
        self.conf = conf

        self._running = True
        self._failsafe = False
        self._recording = False
        self._save_blur_requested = False
        self._save_original_requested = False
        self._record_buffer = []
        self._fps_avg = 0.0
        self._last_enc_key = None
        self._quality_percent = int(quality_percent)
        self._blur_strength = odd_kernel(blur_strength)
        self._blur_type = blur_type if blur_type in BLUR_TYPE_LABELS else BLUR_TYPE_DEFAULT
        self._lock = threading.Lock()

    def run(self):
        cam = None
        prev_time = time.time()
        error_notified = False

        try:
            cam = YOLOBlurCam(self.weight_paths, source=self.rtsp_url)
            self.info_text.emit("카메라 연결 성공")
        except Exception as e:
            self.error_text.emit(f"카메라 연결 실패: {e}")
            self._emit_status(
                fps_avg=0.0,
                infer_state="연결 실패",
                status_code="🔴 ERROR",
                recording=False,
                record_frames=0,
                detections=0,
                heartbeat="중단",
                last_enc_key=self._last_enc_key,
                quality_percent=self._quality_percent,
                blur_strength=self._blur_strength,
                blur_type=self._blur_type,
                safe_fail=True,
            )
            return

        while self._running:
            frame = cam.read_frame()
            if frame is None:
                if not error_notified:
                    self.error_text.emit(
                        "프레임 수신 실패입니다. 노트북 카메라 번호 또는 RTSP 주소/계정/네트워크 상태를 확인하세요."
                    )
                    error_notified = True

                self._emit_status(
                    fps_avg=self._fps_avg,
                    infer_state="프레임 없음",
                    status_code="🔴 ERROR",
                    recording=self._recording,
                    record_frames=len(self._record_buffer),
                    detections=0,
                    heartbeat="비정상",
                    last_enc_key=self._last_enc_key,
                    quality_percent=self._quality_percent,
                    blur_strength=self._blur_strength,
                    blur_type=self._blur_type,
                    safe_fail=True,
                )
                self.msleep(120)
                continue

            error_notified = False

            now = time.time()
            dt = max(now - prev_time, 1e-6)
            prev_time = now
            fps = 1.0 / dt
            self._fps_avg = (0.9 * self._fps_avg + 0.1 * fps) if self._fps_avg > 0 else fps

            with self._lock:
                failsafe = self._failsafe
                recording = self._recording
                save_blur = self._save_blur_requested
                save_original = self._save_original_requested
                quality_percent = self._quality_percent
                blur_strength = self._blur_strength
                blur_type = self._blur_type
                if save_blur:
                    self._save_blur_requested = False
                if save_original:
                    self._save_original_requested = False

            original_frame = frame.copy()

            if recording:
                self._record_buffer.append(original_frame.copy())

            work_frame = self._apply_quality_scale(frame, quality_percent)
            dynamic_imgsz = self._quality_to_imgsz(quality_percent)

            if failsafe:
                frame_view = cam.apply_blur(work_frame, blur_strength, blur_type)
                infer_state = "수동 Safe-Fail"
                status_code = "🔴 SAFE-FAIL"
                detections = 0
                safe_fail_active = True
            else:
                processed_frame, detections = cam.process(
                    work_frame.copy(),
                    conf=self.conf,
                    blur=True,
                    blur_ksize=blur_strength,
                    blur_type=blur_type,
                    margin_ratio=0.10,
                    imgsz=dynamic_imgsz,
                )
                if detections >= 1:
                    frame_view = processed_frame
                    infer_state = "정상"
                    status_code = "🟢 NORMAL"
                    safe_fail_active = False
                else:
                    frame_view = cam.apply_blur(work_frame, blur_strength, blur_type)
                    infer_state = "탐지 없음 → Safe-Fail"
                    status_code = "🔴 SAFE-FAIL"
                    safe_fail_active = True

            if save_blur:
                self._save_blur_video(cam, blur_strength, blur_type)

            if save_original:
                self._save_original_video()

            self.frame_ready.emit(self._to_qimage(frame_view))
            self._emit_status(
                fps_avg=self._fps_avg,
                infer_state=infer_state,
                status_code=status_code,
                recording=recording,
                record_frames=len(self._record_buffer),
                detections=detections,
                heartbeat="정상",
                last_enc_key=self._last_enc_key,
                quality_percent=quality_percent,
                blur_strength=blur_strength,
                blur_type=blur_type,
                safe_fail=safe_fail_active,
            )
            self.msleep(1)

        try:
            if cam is not None:
                cam.release()
        except Exception:
            pass

    def stop(self):
        self._running = False

    def set_failsafe(self, enabled: bool):
        with self._lock:
            self._failsafe = enabled

    def set_recording(self, enabled: bool):
        with self._lock:
            self._recording = enabled

    def set_quality_percent(self, value: int):
        with self._lock:
            self._quality_percent = max(30, min(100, int(value)))

    def set_blur_strength(self, value: int):
        with self._lock:
            self._blur_strength = odd_kernel(value)

    def set_blur_type(self, blur_type: str):
        with self._lock:
            if blur_type in BLUR_TYPE_LABELS:
                self._blur_type = blur_type

    def request_save_blur(self):
        with self._lock:
            self._save_blur_requested = True

    def request_save_original(self):
        with self._lock:
            self._save_original_requested = True

    def clear_buffer(self):
        with self._lock:
            self._record_buffer.clear()
        self.info_text.emit("녹화 버퍼를 비웠습니다.")

    def _save_blur_video(self, cam, blur_strength: int, blur_type: str):
        if not self._record_buffer:
            self.info_text.emit("저장할 녹화 프레임이 없습니다.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(BLUR_DIR, f"blur_{ts}.mp4")
        h, w = self._record_buffer[0].shape[:2]

        vw = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (w, h),
        )
        for frame in self._record_buffer:
            blur_frame, detections = cam.process(
                frame.copy(),
                conf=self.conf,
                blur=True,
                blur_ksize=blur_strength,
                blur_type=blur_type,
                margin_ratio=0.10,
                imgsz=320,
                force_infer=True,
            )
            if detections < 1:
                blur_frame = cam.apply_blur(frame, blur_strength, blur_type)
            vw.write(blur_frame)
        vw.release()
        self.info_text.emit(f"비식별 영상 저장 완료: {path}")

    def _save_original_video(self):
        if not self._record_buffer:
            self.info_text.emit("저장할 녹화 프레임이 없습니다.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mp4_path = os.path.join(ORG_DIR, f"original_{ts}.mp4")

        h, w = self._record_buffer[0].shape[:2]
        vw = cv2.VideoWriter(
            mp4_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (w, h),
        )
        for frame in self._record_buffer:
            vw.write(frame)
        vw.release()

        enc_key = generate_key()
        self._last_enc_key = enc_key

        zip_path = mp4_path.replace(".mp4", ".zip")
        key_zip_path = os.path.join(
            KEY_DIR,
            os.path.basename(zip_path).replace(".zip", ".key.zip"),
        )
        key_txt_name = os.path.basename(zip_path).replace(".zip", ".key.txt")

        # 1) 원본 영상 ZIP: 매번 새로 생성되는 랜덤 암호키로 AES 암호화
        with pyzipper.AESZipFile(
            zip_path,
            "w",
            compression=pyzipper.ZIP_DEFLATED,
            encryption=pyzipper.WZ_AES,
        ) as zf:
            zf.setpassword(enc_key.encode())
            zf.write(mp4_path, arcname=os.path.basename(mp4_path))

        # 2) 암호키 파일 ZIP: records/keys에 따로 저장하고 고정 비밀번호 cabston!으로 AES 암호화
        #    평문 .key.txt 파일을 디스크에 만들지 않고, 암호화 ZIP 내부에 바로 기록한다.
        with pyzipper.AESZipFile(
            key_zip_path,
            "w",
            compression=pyzipper.ZIP_DEFLATED,
            encryption=pyzipper.WZ_AES,
        ) as zf:
            zf.setpassword(KEY_ZIP_PASSWORD.encode())
            zf.writestr(key_txt_name, f"Encryption Key: {enc_key}\n")

        os.remove(mp4_path)
        self.info_text.emit(f"원본 암호화 저장 완료: {zip_path} / 키 저장: {key_zip_path}")

    @staticmethod
    def _apply_quality_scale(frame, quality_percent: int):
        scale = max(0.3, min(1.0, quality_percent / 100.0))
        if scale >= 0.999:
            return frame
        h, w = frame.shape[:2]
        nw = max(320, int(w * scale))
        nh = max(240, int(h * scale))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _quality_to_imgsz(quality_percent: int) -> int:
        imgsz = int(round((320 * quality_percent / 100.0) / 32.0) * 32)
        return max(160, min(320, imgsz))

    def _emit_status(
        self,
        *,
        fps_avg,
        infer_state,
        status_code,
        recording,
        record_frames,
        detections,
        heartbeat,
        last_enc_key,
        quality_percent,
        blur_strength,
        blur_type=BLUR_TYPE_DEFAULT,
        safe_fail=False,
    ):
        self.stats_ready.emit(
            {
                "fps_avg": fps_avg,
                "infer_state": infer_state,
                "status_code": status_code,
                "recording": recording,
                "record_frames": record_frames,
                "detections": detections,
                "heartbeat": heartbeat,
                "last_enc_key": last_enc_key,
                "quality_percent": quality_percent,
                "blur_strength": blur_strength,
                "blur_type": blur_type,
                "blur_type_label": BLUR_TYPE_LABELS.get(blur_type, blur_type),
                "safe_fail": safe_fail,
            }
        )

    @staticmethod
    def _to_qimage(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return image.copy()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.latest_qimage = None

        self.setWindowTitle("실시간 비식별화 기반 프라이버시 보호 시스템 (PySide6)")
        self.resize(1100, 700)

        self._build_ui()
        self._apply_styles()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        top_bar = QHBoxLayout()
        self.rtsp_input = QLineEdit(DEFAULT_RTSP_URL)
        self.rtsp_input.setPlaceholderText("0 = 노트북 기본 카메라, 또는 rtsp://username:password@camera_ip:554/stream1")
        self.connect_btn = QPushButton("연결 시작")
        self.disconnect_btn = QPushButton("연결 종료")
        self.disconnect_btn.setEnabled(False)

        top_bar.addWidget(QLabel("카메라 소스"))
        top_bar.addWidget(self.rtsp_input, 1)
        top_bar.addWidget(self.connect_btn)
        top_bar.addWidget(self.disconnect_btn)
        main_layout.addLayout(top_bar)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)
        main_layout.addLayout(content_layout, 1)

        video_group = QGroupBox("실시간 영상")
        video_layout = QVBoxLayout(video_group)

        self.status_led = QLabel("상태등 : 연결 대기")
        self.status_led.setObjectName("statusLed")
        self.status_led.setAlignment(Qt.AlignCenter)

        self.video_label = QLabel("연결을 시작하면 영상이 표시됩니다.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(QSize(640, 480))
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background: #ffffff; color: #111827; border: 1px solid #d0d7de; border-radius: 12px;")

        video_layout.addWidget(self.status_led)
        video_layout.addWidget(self.video_label, 1)

        side_group = QGroupBox("제어 / 시스템 상태")
        side_layout = QVBoxLayout(side_group)

        btn_grid = QGridLayout()
        self.record_check = QCheckBox("녹화")
        self.failsafe_check = QCheckBox("Fail-Safe")
        self.save_blur_btn = QPushButton("비식별 저장")
        self.save_original_btn = QPushButton("원본 저장 (암호화)")
        self.clear_buffer_btn = QPushButton("버퍼 비우기")

        btn_grid.addWidget(self.record_check, 0, 0)
        btn_grid.addWidget(self.failsafe_check, 0, 1)
        btn_grid.addWidget(self.save_blur_btn, 1, 0, 1, 2)
        btn_grid.addWidget(self.save_original_btn, 2, 0, 1, 2)
        btn_grid.addWidget(self.clear_buffer_btn, 3, 0, 1, 2)
        side_layout.addLayout(btn_grid)

        control_group = QGroupBox("영상 조절")
        control_layout = QVBoxLayout(control_group)

        quality_row = QHBoxLayout()
        self.quality_title = QLabel("화질")
        self.quality_value_label = QLabel(f"{QUALITY_DEFAULT}%")
        quality_row.addWidget(self.quality_title)
        quality_row.addStretch(1)
        quality_row.addWidget(self.quality_value_label)

        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(30, 100)
        self.quality_slider.setValue(QUALITY_DEFAULT)
        self.quality_slider.setTickInterval(10)
        self.quality_slider.setTickPosition(QSlider.TicksBelow)

        blur_row = QHBoxLayout()
        self.blur_title = QLabel("블러 강도")
        self.blur_value_label = QLabel(str(BLUR_DEFAULT))
        blur_row.addWidget(self.blur_title)
        blur_row.addStretch(1)
        blur_row.addWidget(self.blur_value_label)

        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(5, 151)
        self.blur_slider.setSingleStep(2)
        self.blur_slider.setPageStep(4)
        self.blur_slider.setValue(BLUR_DEFAULT)
        self.blur_slider.setTickInterval(16)
        self.blur_slider.setTickPosition(QSlider.TicksBelow)

        blur_type_row = QHBoxLayout()
        self.blur_type_title = QLabel("블러 종류")
        self.blur_type_combo = QComboBox()
        for value, label in BLUR_TYPE_LABELS.items():
            self.blur_type_combo.addItem(label, value)
        default_index = self.blur_type_combo.findData(BLUR_TYPE_DEFAULT)
        if default_index >= 0:
            self.blur_type_combo.setCurrentIndex(default_index)
        blur_type_row.addWidget(self.blur_type_title)
        blur_type_row.addStretch(1)
        blur_type_row.addWidget(self.blur_type_combo)

        control_layout.addLayout(quality_row)
        control_layout.addWidget(self.quality_slider)
        control_layout.addLayout(blur_row)
        control_layout.addWidget(self.blur_slider)
        control_layout.addLayout(blur_type_row)
        side_layout.addWidget(control_group)

        stats_group = QGroupBox("시스템 상태")
        stats_form = QFormLayout(stats_group)

        self.fps_label = QLabel("0.0")
        self.infer_label = QLabel("대기")
        self.heartbeat_label = QLabel("대기")
        self.status_code_label = QLabel("연결 대기")
        self.recording_label = QLabel("OFF")
        self.record_frames_label = QLabel("0")
        self.detect_label = QLabel("0")
        self.last_key_label = QLabel("-")
        self.blur_type_status_label = QLabel(BLUR_TYPE_LABELS[BLUR_TYPE_DEFAULT])

        stats_form.addRow("FPS", self.fps_label)
        stats_form.addRow("추론 상태", self.infer_label)
        stats_form.addRow("하트비트", self.heartbeat_label)
        stats_form.addRow("상태 코드", self.status_code_label)
        stats_form.addRow("녹화 상태", self.recording_label)
        stats_form.addRow("녹화 프레임 수", self.record_frames_label)
        stats_form.addRow("탐지 수", self.detect_label)
        stats_form.addRow("블러 종류", self.blur_type_status_label)
        stats_form.addRow("최근 암호 키", self.last_key_label)

        side_layout.addWidget(stats_group)

        self.message_label = QLabel("노트북 카메라는 0, RTSP 카메라는 주소를 넣고 연결 시작을 누르세요.")
        self.message_label.setWordWrap(True)
        self.message_label.setObjectName("messageLabel")
        side_layout.addWidget(self.message_label)
        side_layout.addStretch(1)

        content_layout.addWidget(video_group, 3)
        content_layout.addWidget(side_group, 1)

        exit_action = QAction("종료", self)
        exit_action.triggered.connect(self.close)
        file_menu = self.menuBar().addMenu("파일")
        file_menu.addAction(exit_action)

        self.connect_btn.clicked.connect(self.start_stream)
        self.disconnect_btn.clicked.connect(self.stop_stream)
        self.record_check.toggled.connect(self.on_record_toggled)
        self.failsafe_check.toggled.connect(self.on_failsafe_toggled)
        self.save_blur_btn.clicked.connect(self.on_save_blur)
        self.save_original_btn.clicked.connect(self.on_save_original)
        self.clear_buffer_btn.clicked.connect(self.on_clear_buffer)
        self.quality_slider.valueChanged.connect(self.on_quality_changed)
        self.blur_slider.valueChanged.connect(self.on_blur_changed)
        self.blur_type_combo.currentIndexChanged.connect(self.on_blur_type_changed)

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                font-size: 13px;
                color: #111827;
                background: #f8fafc;
            }
            QLabel, QCheckBox, QComboBox, QGroupBox, QMenuBar, QMenu, QPushButton, QSlider {
                color: #111827;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #d0d7de;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 14px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QPushButton {
                min-height: 36px;
                border-radius: 10px;
                padding: 6px 12px;
                background: #ffffff;
                border: 1px solid #cbd5e1;
            }
            QPushButton:disabled {
                color: #6b7280;
                background: #f3f4f6;
            }
            QComboBox {
                min-height: 34px;
                border-radius: 10px;
                padding: 4px 8px;
                background: #ffffff;
                border: 1px solid #cbd5e1;
                color: #000000;
            }
            QLineEdit {
                padding: 8px;
                border-radius: 10px;
                background: #ffffff;
                border: 1px solid #cbd5e1;
                color: #000000;
                selection-background-color: #bfdbfe;
                selection-color: #000000;
            }
            QLineEdit::placeholder {
                color: #64748b;
            }
            QSlider::groove:horizontal {
                border: 1px solid #cbd5e1;
                height: 8px;
                background: #e5e7eb;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #111827;
                border: 1px solid #111827;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #94a3b8;
                border-radius: 4px;
            }
            QLabel#messageLabel {
                padding: 8px;
                border-radius: 10px;
                background: #ffffff;
                color: #111827;
                border: 1px solid #dbe3eb;
            }
            QLabel#statusLed {
                min-height: 40px;
                border-radius: 12px;
                font-weight: 700;
                background: #e2e8f0;
                color: #111827;
                padding: 6px 10px;
            }
            """
        )

    def start_stream(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "안내", "이미 스트리밍 중입니다.")
            return

        source_text = self.rtsp_input.text().strip()
        if not source_text:
            QMessageBox.warning(self, "입력 필요", "카메라 소스(예: 0 또는 RTSP URL)를 입력하세요.")
            return

        # 노트북 기본 카메라를 쉽게 쓰기 위한 변환
        # 0, 1, 2처럼 숫자만 입력하면 OpenCV 카메라 번호로 처리한다.
        if source_text.isdigit():
            camera_source = int(source_text)
        elif source_text.lower() in {"webcam", "camera", "notebook", "laptop", "노트북", "노트북카메라"}:
            camera_source = 0
        else:
            camera_source = source_text

        missing = [os.path.basename(p) for p in WEIGHTS if not os.path.exists(p)]
        if missing:
            QMessageBox.critical(
                self,
                "가중치 누락",
                f"다음 가중치 파일이 없습니다:\n- " + "\n- ".join(missing),
            )
            return

        self.worker = VideoWorker(
            WEIGHTS,
            camera_source,
            quality_percent=self.quality_slider.value(),
            blur_strength=odd_kernel(self.blur_slider.value()),
            blur_type=self.blur_type_combo.currentData(),
        )
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.stats_ready.connect(self.update_stats)
        self.worker.error_text.connect(self.show_error)
        self.worker.info_text.connect(self.show_info)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self.message_label.setText(f"카메라에 연결 중입니다: {source_text}")

    def stop_stream(self):
        if not self.worker:
            return

        self.worker.stop()
        self.worker.wait(2000)

        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.status_led.setText("상태등 : 연결 종료")
        self.status_led.setStyleSheet("background: #e2e8f0; color: #0f172a; padding: 6px 10px; border-radius: 12px;")
        self.message_label.setText("스트리밍을 종료했습니다.")
        self.video_label.setText("연결을 시작하면 영상이 표시됩니다.")
        self.video_label.setPixmap(QPixmap())
        self.latest_qimage = None
        self.worker = None

    def on_worker_finished(self):
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)

    def update_frame(self, image: QImage):
        self.latest_qimage = image
        self._render_current_frame()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_current_frame()

    def _render_current_frame(self):
        if self.latest_qimage is None:
            return

        pixmap = QPixmap.fromImage(self.latest_qimage)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.video_label.setPixmap(scaled)

    def update_stats(self, stats):
        self.fps_label.setText(f"{stats['fps_avg']:.1f}")
        self.infer_label.setText(stats["infer_state"])
        self.heartbeat_label.setText(stats["heartbeat"])
        self.status_code_label.setText(stats["status_code"])
        self.recording_label.setText("ON" if stats["recording"] else "OFF")
        self.record_frames_label.setText(str(stats["record_frames"]))
        self.detect_label.setText(str(stats["detections"]))
        self.last_key_label.setText(stats["last_enc_key"] or "-")
        self.quality_value_label.setText(f"{stats['quality_percent']}%")
        self.blur_value_label.setText(str(stats["blur_strength"]))
        self.blur_type_status_label.setText(stats.get("blur_type_label", "-"))

        if stats["detections"] >= 1 and not stats.get("safe_fail", False):
            self.status_led.setText("상태등 : 🟢 NORMAL")
            self.status_led.setStyleSheet("background: #dcfce7; color: #166534; padding: 6px 10px; border-radius: 12px;")
        else:
            self.status_led.setText("상태등 : 🔴 SAFE-FAIL")
            self.status_led.setStyleSheet("background: #fee2e2; color: #991b1b; padding: 6px 10px; border-radius: 12px;")

    def show_error(self, text):
        self.message_label.setText(text)

    def show_info(self, text):
        self.message_label.setText(text)

    def on_record_toggled(self, checked):
        if self.worker and self.worker.isRunning():
            self.worker.set_recording(checked)

    def on_failsafe_toggled(self, checked):
        if self.worker and self.worker.isRunning():
            self.worker.set_failsafe(checked)

    def on_save_blur(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_save_blur()
        else:
            QMessageBox.information(self, "안내", "먼저 카메라를 연결하세요.")

    def on_save_original(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_save_original()
        else:
            QMessageBox.information(self, "안내", "먼저 카메라를 연결하세요.")

    def on_clear_buffer(self):
        if self.worker and self.worker.isRunning():
         self.worker.clear_buffer()
         QMessageBox.information(self, "버퍼 비우기", "녹화 버퍼를 비웠습니다.")
        else:
         QMessageBox.information(self, "안내", "먼저 카메라를 연결하세요.")

    def on_quality_changed(self, value):
        self.quality_value_label.setText(f"{value}%")
        if self.worker and self.worker.isRunning():
            self.worker.set_quality_percent(value)
            self.message_label.setText(f"화질을 {value}%로 변경했습니다.")

    def on_blur_changed(self, value):
        kernel = odd_kernel(value)
        if kernel != value:
            self.blur_slider.blockSignals(True)
            self.blur_slider.setValue(kernel)
            self.blur_slider.blockSignals(False)
        self.blur_value_label.setText(str(kernel))
        if self.worker and self.worker.isRunning():
            self.worker.set_blur_strength(kernel)
            self.message_label.setText(f"블러 강도를 {kernel}로 변경했습니다.")

    def on_blur_type_changed(self, index):
        blur_type = self.blur_type_combo.itemData(index)
        blur_label = self.blur_type_combo.itemText(index)
        self.blur_type_status_label.setText(blur_label)
        if self.worker and self.worker.isRunning():
            self.worker.set_blur_type(blur_type)
            self.message_label.setText(f"블러 종류를 {blur_label}(으)로 변경했습니다.")

    def closeEvent(self, event: QCloseEvent):
        self.stop_stream()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


