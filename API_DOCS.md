# 의류 세그멘테이션 API 문서

## 개요

이 API는 SegFormer B2 모델을 사용하여 이미지에서 의류 및 신체 부위를 감지하고 세그멘테이션하는 서비스를 제공합니다. 웨딩드레스를 포함한 다양한 의류 항목을 자동으로 감지하고 배경을 제거할 수 있습니다.

- **Base URL**: `http://localhost:8000`
- **버전**: 1.0.0
- **모델**: [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes)

## 목차

- [인증](#인증)
- [레이블 정보](#레이블-정보)
- [API 엔드포인트](#api-엔드포인트)
  - [정보 조회](#정보-조회)
  - [세그멘테이션](#세그멘테이션)
  - [분석](#분석)
- [사용 예제](#사용-예제)
- [오류 처리](#오류-처리)

## 인증

현재 버전은 인증이 필요하지 않습니다.

## 레이블 정보

모델이 감지할 수 있는 18개 레이블:

| ID | 레이블 이름 | 설명 |
|----|------------|------|
| 0 | Background | 배경 |
| 1 | Hat | 모자 |
| 2 | Hair | 머리카락 |
| 3 | Sunglasses | 선글라스 |
| 4 | Upper-clothes | 상의 (셔츠, 블라우스 등) |
| 5 | Skirt | 치마 |
| 6 | Pants | 바지 |
| 7 | Dress | 드레스 (웨딩드레스 포함) |
| 8 | Belt | 벨트 |
| 9 | Left-shoe | 왼쪽 신발 |
| 10 | Right-shoe | 오른쪽 신발 |
| 11 | Face | 얼굴 |
| 12 | Left-leg | 왼쪽 다리 |
| 13 | Right-leg | 오른쪽 다리 |
| 14 | Left-arm | 왼쪽 팔 |
| 15 | Right-arm | 오른쪽 팔 |
| 16 | Bag | 가방 |
| 17 | Scarf | 스카프 |

---

## API 엔드포인트

### 정보 조회

#### 1. 서버 상태 확인

서버와 모델의 상태를 확인합니다.

**엔드포인트**: `GET /health`

**응답 예제**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "mattmdjaga/segformer_b2_clothes",
  "version": "1.0.0"
}
```

---

#### 2. 레이블 목록 조회

사용 가능한 모든 레이블 목록을 조회합니다.

**엔드포인트**: `GET /labels`

**응답 예제**:
```json
{
  "labels": {
    "0": "Background",
    "1": "Hat",
    "2": "Hair",
    ...
  },
  "total_labels": 18,
  "description": "SegFormer B2 모델이 감지할 수 있는 레이블 목록"
}
```

---

### 세그멘테이션

#### 3. 드레스 세그멘테이션

웨딩드레스를 감지하고 배경을 제거합니다.

**엔드포인트**: `POST /api/segment`

**요청**:
- Content-Type: `multipart/form-data`
- Body:
  - `file` (required): 이미지 파일

**응답 예제**:
```json
{
  "success": true,
  "original_image": "data:image/png;base64,iVBORw0KG...",
  "result_image": "data:image/png;base64,iVBORw0KG...",
  "dress_detected": true,
  "dress_percentage": 15.42,
  "message": "드레스 영역: 15.42% 감지됨"
}
```

**cURL 예제**:
```bash
curl -X POST "http://localhost:8000/api/segment" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@wedding_photo.jpg"
```

**Python 예제**:
```python
import requests

url = "http://localhost:8000/api/segment"
files = {"file": open("wedding_photo.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"드레스 감지: {result['dress_detected']}")
print(f"드레스 비율: {result['dress_percentage']}%")
```

---

#### 4. 커스텀 레이블 세그멘테이션

원하는 레이블만 선택하여 세그멘테이션합니다.

**엔드포인트**: `POST /api/segment-custom`

**요청**:
- Content-Type: `multipart/form-data`
- Body:
  - `file` (required): 이미지 파일
  - `labels` (required): 추출할 레이블 ID (쉼표로 구분)

**예제 레이블 조합**:
- `labels=7`: 드레스만
- `labels=4,6`: 상의와 바지
- `labels=4,5,6,7`: 모든 의류 (상의, 치마, 바지, 드레스)
- `labels=1,2,11`: 모자, 머리, 얼굴

**응답 예제**:
```json
{
  "success": true,
  "original_image": "data:image/png;base64,iVBORw0KG...",
  "result_image": "data:image/png;base64,iVBORw0KG...",
  "requested_labels": [
    {"id": 4, "name": "Upper-clothes"},
    {"id": 6, "name": "Pants"}
  ],
  "detected_labels": [
    {"id": 4, "name": "Upper-clothes", "percentage": 8.5},
    {"id": 6, "name": "Pants", "percentage": 12.3}
  ],
  "total_percentage": 20.8,
  "message": "2개의 레이블 감지됨"
}
```

**cURL 예제**:
```bash
curl -X POST "http://localhost:8000/api/segment-custom?labels=4,6" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@person_photo.jpg"
```

**Python 예제**:
```python
import requests

url = "http://localhost:8000/api/segment-custom"
params = {"labels": "4,6"}  # 상의와 바지
files = {"file": open("person_photo.jpg", "rb")}
response = requests.post(url, params=params, files=files)
result = response.json()

for label in result['detected_labels']:
    print(f"{label['name']}: {label['percentage']}%")
```

---

#### 5. 전체 배경 제거

배경만 제거하고 인물과 의류는 모두 유지합니다.

**엔드포인트**: `POST /api/remove-background`

**요청**:
- Content-Type: `multipart/form-data`
- Body:
  - `file` (required): 이미지 파일

**응답 예제**:
```json
{
  "success": true,
  "original_image": "data:image/png;base64,iVBORw0KG...",
  "result_image": "data:image/png;base64,iVBORw0KG...",
  "foreground_percentage": 35.6,
  "message": "배경 제거 완료 (인물 영역: 35.6%)"
}
```

**cURL 예제**:
```bash
curl -X POST "http://localhost:8000/api/remove-background" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@portrait.jpg"
```

**JavaScript 예제**:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/api/remove-background', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('배경 제거 완료:', data.message);
  document.getElementById('result').src = data.result_image;
});
```

---

### 분석

#### 6. 이미지 전체 분석

이미지에서 모든 레이블을 감지하고 비율을 분석합니다. 누끼 처리 없이 분석 정보만 반환합니다.

**엔드포인트**: `POST /api/analyze`

**요청**:
- Content-Type: `multipart/form-data`
- Body:
  - `file` (required): 이미지 파일

**응답 예제**:
```json
{
  "success": true,
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "total_pixels": 2073600,
  "detected_labels": [
    {"id": 0, "name": "Background", "pixels": 1500000, "percentage": 72.35},
    {"id": 7, "name": "Dress", "pixels": 320000, "percentage": 15.43},
    {"id": 2, "name": "Hair", "pixels": 125000, "percentage": 6.03},
    {"id": 11, "name": "Face", "pixels": 85000, "percentage": 4.10},
    {"id": 14, "name": "Left-arm", "pixels": 25000, "percentage": 1.21},
    {"id": 15, "name": "Right-arm", "pixels": 18600, "percentage": 0.90}
  ],
  "total_detected": 6,
  "message": "총 6개의 레이블 감지됨"
}
```

**cURL 예제**:
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

**Python 예제**:
```python
import requests

url = "http://localhost:8000/api/analyze"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"이미지 크기: {result['image_size']['width']}x{result['image_size']['height']}")
print(f"감지된 레이블: {result['total_detected']}개\n")

for label in result['detected_labels']:
    print(f"- {label['name']}: {label['percentage']}% ({label['pixels']:,} pixels)")
```

---

## 사용 예제

### JavaScript/TypeScript (프론트엔드)

```javascript
// 파일 업로드 및 드레스 세그멘테이션
async function segmentDress(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('http://localhost:8000/api/segment', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (data.success) {
      // 원본 이미지 표시
      document.getElementById('original').src = data.original_image;
      // 결과 이미지 표시
      document.getElementById('result').src = data.result_image;
      // 메시지 표시
      console.log(data.message);
    } else {
      console.error('오류:', data.message);
    }
  } catch (error) {
    console.error('네트워크 오류:', error);
  }
}

// 파일 input 이벤트 핸들러
document.getElementById('fileInput').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    segmentDress(file);
  }
});
```

### Python

```python
import requests
import base64
from PIL import Image
from io import BytesIO

def segment_wedding_dress(image_path):
    """웨딩드레스 세그멘테이션"""
    url = "http://localhost:8000/api/segment"
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        
        if result['success']:
            # Base64 결과 이미지를 PIL Image로 변환
            image_data = result['result_image'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            
            # 이미지 저장
            image.save('result.png')
            print(f"저장 완료: {result['message']}")
            
            return image
        else:
            print(f"오류: {result['message']}")
    else:
        print(f"HTTP 오류: {response.status_code}")
    
    return None

# 사용 예제
result_image = segment_wedding_dress('wedding_photo.jpg')
```

### Node.js

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function analyzeImage(imagePath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(imagePath));

  try {
    const response = await axios.post(
      'http://localhost:8000/api/analyze',
      form,
      { headers: form.getHeaders() }
    );

    const data = response.data;
    
    console.log(`이미지 크기: ${data.image_size.width}x${data.image_size.height}`);
    console.log(`감지된 레이블: ${data.total_detected}개\n`);

    data.detected_labels.forEach(label => {
      console.log(`- ${label.name}: ${label.percentage}%`);
    });

  } catch (error) {
    console.error('오류:', error.message);
  }
}

analyzeImage('photo.jpg');
```

---

## 오류 처리

### 오류 응답 형식

```json
{
  "success": false,
  "error": "상세 에러 메시지",
  "message": "사용자 친화적 에러 메시지"
}
```

### 일반적인 오류 코드

| 상태 코드 | 설명 | 해결 방법 |
|----------|------|----------|
| 400 | 잘못된 요청 | 요청 파라미터 확인 |
| 413 | 파일 크기 초과 | 10MB 이하의 이미지 사용 |
| 415 | 지원하지 않는 파일 형식 | JPG, PNG 등 지원되는 형식 사용 |
| 500 | 서버 내부 오류 | 서버 로그 확인 또는 관리자 문의 |

### 오류 처리 예제

```python
import requests

def safe_api_call(image_path):
    try:
        url = "http://localhost:8000/api/segment"
        files = {'file': open(image_path, 'rb')}
        response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                return result
            else:
                print(f"처리 실패: {result['message']}")
        else:
            print(f"HTTP 오류: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("요청 시간 초과")
    except requests.exceptions.ConnectionError:
        print("서버 연결 실패")
    except Exception as e:
        print(f"예상치 못한 오류: {str(e)}")
    
    return None
```

---

## Swagger UI

API 문서는 서버 실행 후 다음 URL에서 인터랙티브하게 확인할 수 있습니다:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Swagger UI에서 직접 API를 테스트하고 요청/응답 예제를 확인할 수 있습니다.

---

## 성능 및 제한사항

### 성능

- **처리 시간**: 이미지당 약 1-3초 (이미지 크기 및 하드웨어에 따라 다름)
- **최대 파일 크기**: 10MB
- **지원 형식**: JPG, PNG, GIF, WEBP

### 제한사항

- CPU에서 실행 시 처리 속도가 느릴 수 있습니다
- 매우 큰 이미지는 메모리 부족이 발생할 수 있습니다
- 동시 요청 수가 많을 경우 대기 시간이 증가할 수 있습니다

### 최적화 팁

1. **이미지 크기 조정**: 처리 전 적절한 크기로 리사이즈 (예: 최대 1920x1080)
2. **GPU 사용**: CUDA 지원 GPU가 있다면 처리 속도 향상
3. **배치 처리**: 여러 이미지를 처리할 경우 비동기로 처리

---

## 라이선스

이 API는 MIT 라이선스 하에 제공됩니다.
사용된 SegFormer 모델의 라이선스는 [Hugging Face 모델 페이지](https://huggingface.co/mattmdjaga/segformer_b2_clothes)에서 확인하세요.

---

## 지원 및 문의

- GitHub: [프로젝트 저장소]
- Email: [이메일 주소]

---

## 변경 이력

### v1.0.0 (2025-10-28)
- 초기 릴리스
- 드레스 세그멘테이션 기능
- 커스텀 레이블 세그멘테이션 기능
- 배경 제거 기능
- 이미지 분석 기능


