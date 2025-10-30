# 웨딩드레스 누끼 서비스 👰

웨딩드레스를 입은 인물 이미지에서 드레스만 자동으로 추출하는 웹 애플리케이션입니다.

## 기능

- 🎯 웨딩드레스 자동 감지 및 세그멘테이션
- 🎨 **NEW!** Gemini AI를 이용한 사람과 드레스 이미지 합성
- 🖼️ 원본 이미지와 결과 이미지 나란히 비교
- 💾 투명 배경(PNG) 결과 이미지 다운로드
- 📱 반응형 웹 디자인
- 🚀 빠른 처리 속도

## 기술 스택

- **백엔드**: FastAPI
- **AI 모델**: 
  - [SegFormer B2 Clothes Segmentation](https://huggingface.co/mattmdjaga/segformer_b2_clothes)
  - [Gemini 2.0 Flash Exp](https://ai.google.dev/) - 이미지 합성
- **프론트엔드**: HTML5, CSS3, Vanilla JavaScript
- **ML 프레임워크**: PyTorch, Transformers (Hugging Face), Google Generative AI

## 설치 방법

### 1. 필수 요구사항

- Python 3.8 이상
- pip
- Gemini API 키 (이미지 합성 기능 사용 시)

### 2. 환경 변수 설정

`.env` 파일을 생성하고 Gemini API 키를 설정하세요:

```bash
# .env 파일
GEMINI_API_KEY=your_gemini_api_key_here
```

**Gemini API 키 발급**: https://ai.google.dev/

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

**참고**: PyTorch 설치 시 시스템에 맞는 버전을 선택하세요:
- CUDA가 있는 경우: GPU 버전 PyTorch 설치 권장
- CPU만 사용하는 경우: CPU 버전 PyTorch 설치

자세한 내용은 [PyTorch 공식 사이트](https://pytorch.org/)를 참조하세요.

## 실행 방법

### uvicorn으로 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

또는 Python 스크립트로 직접 실행:

```bash
python main.py
```

### 서버 시작 옵션

- `--reload`: 개발 모드 (코드 변경 시 자동 재시작)
```bash
uvicorn main:app --reload
```

- `--workers`: 워커 프로세스 수 설정 (프로덕션 환경)
```bash
uvicorn main:app --workers 4
```

## 사용 방법

1. 웹 브라우저에서 `http://localhost:8000` 접속
2. 웨딩드레스를 입은 인물 이미지 업로드 (드래그 앤 드롭 또는 클릭)
3. 자동으로 드레스 영역 감지 및 누끼 처리
4. 원본과 결과 이미지 확인
5. "결과 다운로드" 버튼으로 PNG 파일 저장

## 프로젝트 구조

```
proj5/
├── main.py                 # FastAPI 애플리케이션
├── requirements.txt        # Python 패키지 의존성
├── README.md              # 프로젝트 문서
├── templates/
│   └── index.html         # 메인 HTML 템플릿
├── static/
│   ├── style.css          # 스타일시트
│   └── script.js          # 클라이언트 JavaScript
└── uploads/               # 임시 업로드 폴더 (자동 생성)
```

## API 엔드포인트

이 프로젝트는 다양한 세그멘테이션 기능을 제공하는 RESTful API를 포함합니다.

### 정보 조회
- `GET /`: 웹 인터페이스
- `GET /gemini-test`: **NEW!** Gemini 이미지 합성 테스트 페이지
- `GET /health`: 서버 상태 확인
- `GET /labels`: 사용 가능한 레이블 목록 조회

### 세그멘테이션
- `POST /api/segment`: 드레스 세그멘테이션 (웨딩드레스 누끼)
- `POST /api/segment-custom`: 커스텀 레이블 세그멘테이션
- `POST /api/remove-background`: 전체 배경 제거 (인물만 추출)

### Gemini 이미지 합성
- `POST /api/compose-dress`: **NEW!** 사람 이미지와 드레스 이미지를 합성

### 분석
- `POST /api/analyze`: 이미지 전체 분석 (모든 레이블 감지 및 비율 분석)

**상세 API 문서**: [`API_DOCS.md`](./API_DOCS.md) 참조

### Swagger UI

서버 실행 후 다음 URL에서 인터랙티브 API 문서를 확인할 수 있습니다:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 지원 형식

- **입력 이미지**: JPG, PNG, GIF, WEBP 등 대부분의 이미지 형식
- **출력 이미지**: PNG (투명 배경 지원)
- **최대 파일 크기**: 10MB

## 모델 정보

이 프로젝트는 [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes) 모델을 사용합니다.

### 감지 가능한 레이블
- 0: Background (배경)
- 1: Hat (모자)
- 2: Hair (머리)
- 3: Sunglasses (선글라스)
- 4: Upper-clothes (상의)
- 5: Skirt (치마)
- 6: Pants (바지)
- **7: Dress (드레스)** ← 이 프로젝트에서 사용
- 8: Belt (벨트)
- 9-10: Shoes (신발)
- 11: Face (얼굴)
- 12-15: Arms and Legs (팔과 다리)
- 16: Bag (가방)
- 17: Scarf (스카프)

## 문제 해결

### 모델 다운로드 오류
첫 실행 시 모델이 자동으로 다운로드됩니다. 인터넷 연결을 확인하세요.

### 메모리 부족
대용량 이미지 처리 시 메모리 부족이 발생할 수 있습니다. 이미지 크기를 줄이거나 시스템 메모리를 확인하세요.

### 포트 충돌
8000번 포트가 이미 사용 중인 경우:
```bash
uvicorn main:app --port 8080
```

## 라이선스

이 프로젝트에서 사용하는 SegFormer 모델의 라이선스는 [여기](https://huggingface.co/mattmdjaga/segformer_b2_clothes)에서 확인할 수 있습니다.

## 참고 자료

- [SegFormer Paper (arXiv:2105.15203)](https://arxiv.org/abs/2105.15203)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

