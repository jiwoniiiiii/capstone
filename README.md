# YOLO 기반 실시간 영상 비식별화 및 원본 암호화 저장 시스템

## 프로젝트 소개
CCTV, 웹캠 및 RTSP 기반 IP 카메라의 실시간 영상에서 얼굴과 차량 번호판을 YOLO로 탐지하고,
OpenCV를 이용해 자동으로 비식별화하는 캡스톤디자인 프로젝트입니다.

객체가 탐지되지 않는 경우 전체 화면을 블러 처리하는 Safe-Fail 기능을 적용하고,
원본 영상은 암호화하여 별도로 저장하도록 구현했습니다.

## 주요 기능
- YOLO 기반 얼굴 및 차량 번호판 실시간 탐지
- OpenCV 기반 객체 영역 블러 처리
- 가우시안/평균/모션 블러 지원
- 웹캠 및 RTSP IP 카메라 입력
- PySide6 기반 실시간 모니터링 GUI
- 화질 및 블러 강도 조절
- 객체 미탐지 시 전체 화면 Safe-Fail
- 비식별 영상 저장
- 원본 영상 AES 암호화 ZIP 저장
- 영상별 랜덤 암호화 키 생성
- FPS, 탐지 수, 녹화 상태 등 시스템 상태 표시

## 프로젝트 구조
```text
privacy_ui/
├─ app.py
├─ yolo_cam.py
├─ best1.pt
├─ best2.pt
├─ best3.pt
├─ README.md
├─ requirements.txt
├─ .gitignore
└─ records/        
```

## 모델 파일
프로그램 실행에는 `best1.pt`, `best2.pt`, `best3.pt`가 필요합니다.
가중치 파일은 용량 문제로 GitHub 저장소에서 제외하며,
실행할 때 세 파일을 `app.py`와 같은 폴더에 배치합니다.

## 설치
```bash
pip install -r requirements.txt
```

## 암호키 ZIP 비밀번호 설정
보안을 위해 암호키 ZIP의 비밀번호를 소스코드에 저장하지 않습니다.
프로그램 실행 전에 `KEY_ZIP_PASSWORD` 환경변수를 설정합니다.

### Windows PowerShell
```powershell
$env:KEY_ZIP_PASSWORD="본인이_정한_비밀번호"
python app.py
```

### Windows 명령 프롬프트(CMD)
```cmd
set KEY_ZIP_PASSWORD=본인이_정한_비밀번호
python app.py
```



## 실행
프로그램 실행 후 카메라 소스에 다음 중 하나를 입력합니다.

- `0` : 노트북 기본 웹캠
- RTSP 주소 : IP 카메라 연결

## 처리 흐름
```text
CCTV / Webcam / IP Camera
          ↓
       영상 입력
          ↓
    YOLO 객체 탐지
          ↓
 ┌────────┴────────┐
 객체 탐지        미탐지
     ↓               ↓
객체 영역 블러    전체 화면 블러
 └────────┬────────┘
          ↓
    PySide6 GUI 출력
          ↓
   영상 저장 및 암호화
```

## 저장 데이터
- `records/blur/` : 비식별 영상
- `records/original_enc/` : 암호화된 원본 영상
- `records/keys/` : 암호키 관련 파일


## 개발 환경
- Python
- Ultralytics YOLO
- OpenCV
- PySide6
- NumPy
- pyzipper
