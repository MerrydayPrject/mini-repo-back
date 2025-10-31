# 옷 합성 시스템 프로세스 문서

## 목차
1. [누끼 기능 프로세스](#누끼-기능-프로세스)
2. [합성 기능 프로세스](#합성-기능-프로세스)
3. [사용 모델 및 API](#사용-모델-및-api)

---

## 누끼 기능 프로세스

누끼 기능은 이미지에서 특정 의류 항목을 감지하고 배경을 제거하는 기능입니다. SegFormer B2 모델을 사용하여 세그멘테이션을 수행합니다.

### 1. 드레스 누끼 (`/api/segment`)

**기능**: 웨딩드레스(레이블 7)만 감지하여 배경 제거

**프로세스**:

1. **이미지 입력 받기**
   - 클라이언트로부터 이미지 파일 업로드 (JPG, PNG, GIF, WEBP 등)
   - `UploadFile` 타입으로 FastAPI가 파일 수신

2. **이미지 전처리**
   - 업로드된 파일을 `io.BytesIO`로 읽기
   - PIL Image로 변환 후 RGB 포맷으로 변환
   - 원본 이미지 크기 저장

3. **모델 추론**
   - SegFormer ImageProcessor로 이미지 전처리
   - `mattmdjaga/segformer_b2_clothes` 모델에 입력
   - PyTorch 모델 추론 수행 (torch.no_grad()로 메모리 최적화)
   - 모델 출력 logits 획득

4. **업샘플링 (Upsampling)**
   - 모델 출력 크기를 원본 이미지 크기로 조정
   - Bilinear 보간 방식 사용
   - `(height, width)` 형태로 변환

5. **세그멘테이션 마스크 생성**
   - 업샘플링된 logits에서 argmax로 각 픽셀의 예측 레이블 결정
   - 드레스 레이블(7)만 추출하여 마스크 생성
   - 마스크: 드레스 영역은 255, 배경은 0

6. **누끼 이미지 생성**
   - 원본 이미지를 numpy 배열로 변환
   - RGBA 포맷의 결과 이미지 생성
   - RGB 채널: 원본 이미지 픽셀 값
   - Alpha 채널: 생성한 드레스 마스크 (투명도 제어)

7. **결과 처리**
   - 원본 이미지와 결과 이미지를 base64로 인코딩
   - 드레스 감지 비율 계산
   - JSON 응답 반환:
     - `success`: 성공 여부
     - `original_image`: 원본 이미지 (base64)
     - `result_image`: 누끼 결과 이미지 (base64, 투명 배경)
     - `dress_detected`: 드레스 감지 여부
     - `dress_percentage`: 드레스 영역 비율

### 2. 커스텀 레이블 누끼 (`/api/segment-custom`)

**기능**: 사용자가 지정한 여러 레이블만 추출하여 배경 제거

**프로세스**:

1. **이미지 및 레이블 입력**
   - 이미지 파일 업로드
   - 쿼리 파라미터로 레이블 ID 입력 (예: `labels=4,5,6,7`)
   - 레이블 ID를 파싱하여 리스트로 변환

2. **모델 추론 및 업샘플링**
   - 드레스 누끼와 동일한 방식으로 모델 추론 수행
   - 업샘플링 수행

3. **복합 마스크 생성**
   - 지정된 모든 레이블의 마스크를 OR 연산으로 결합
   - 선택한 레이블 중 하나라도 감지된 픽셀은 포함

4. **누끼 이미지 생성**
   - 복합 마스크를 사용하여 RGBA 이미지 생성
   - 각 레이블별 감지 비율 계산

5. **결과 반환**
   - 요청한 레이블 정보
   - 감지된 레이블 목록 및 비율
   - 전체 감지 비율

**사용 가능한 레이블**:
- 0: Background
- 1: Hat
- 2: Hair
- 3: Sunglasses
- 4: Upper-clothes
- 5: Skirt
- 6: Pants
- 7: Dress
- 8: Belt
- 9-10: Left/Right-shoe
- 11: Face
- 12-13: Left/Right-leg
- 14-15: Left/Right-arm
- 16: Bag
- 17: Scarf

### 3. 전체 배경 제거 (`/api/remove-background`)

**기능**: 배경을 제거하고 인물과 의류만 추출

**프로세스**:

1. **이미지 입력 및 모델 추론**
   - 동일한 방식으로 이미지 처리
   - 모델 추론 수행

2. **배경 제거 마스크 생성**
   - 배경이 아닌 모든 레이블 (레이블 0 제외)을 포함하는 마스크 생성
   - `pred_seg != 0` 조건으로 마스크 생성

3. **누끼 이미지 생성**
   - 인물과 의류만 남기고 배경을 투명하게 처리

4. **결과 반환**
   - 배경 제거된 이미지 (투명 배경)
   - 인물 영역 비율

---

## 합성 기능 프로세스

합성 기능은 사람 이미지와 드레스 이미지를 받아서 AI가 사람이 드레스를 입은 것처럼 합성하는 기능입니다. Google Gemini API를 사용합니다.

### 드레스 합성 (`/api/compose-dress`)

**기능**: 사람 이미지에 드레스를 입혀서 합성된 이미지 생성

**프로세스**:

1. **처리 시작 및 초기화**
   - 처리 시작 시간 기록 (`time.time()`)
   - 변수 초기화 (이미지 경로, 성공 여부, 에러 메시지 등)

2. **API 키 확인**
   - 환경 변수에서 `GEMINI_API_KEY` 확인
   - 없으면 에러 응답 반환

3. **이미지 입력 받기**
   - `person_image`: 사람 이미지 파일
   - `dress_image`: 드레스 이미지 파일
   - 파일을 읽어서 PIL Image로 변환

4. **입력 이미지 저장**
   - 파일 시스템에 타임스탬프 기반 파일명으로 저장
   - `uploads/person_{timestamp}.png`
   - `uploads/dress_{timestamp}.png`
   - 저장 경로를 변수에 저장

5. **이미지 Base64 인코딩**
   - 원본 이미지들을 base64로 인코딩
   - 클라이언트 응답에 포함하기 위함

6. **Gemini Client 생성**
   - API 키로 `genai.Client` 인스턴스 생성

7. **프롬프트 생성**
   - 드레스 합성을 위한 상세 프롬프트 작성:
     ```
     IMPORTANT: You must preserve the person's identity completely.
     
     Task: Apply ONLY the dress from the first image onto the person from the second image.
     
     STRICT REQUIREMENTS:
     1. PRESERVE EXACTLY: The person's face, facial features, skin tone, hair, and body proportions
     2. PRESERVE EXACTLY: The person's pose, stance, and body position
     3. PRESERVE EXACTLY: The background and lighting from the person's image
     4. CHANGE ONLY: Replace the person's clothing with the dress from the first image
     5. The dress should fit naturally on the person's body shape
     6. Maintain realistic shadows and fabric draping on the dress
     7. Keep the person's hands, arms, legs exactly as they are in the original
     
     DO NOT change the person's appearance, face, body type, or any physical features.
     ONLY apply the dress design, color, and style onto the existing person.
     ```

8. **Gemini API 호출**
   - 모델: `gemini-2.5-flash-image`
   - 입력 순서: `[dress_img, person_img, text_input]`
   - API 응답 대기

9. **응답 처리**
   - 응답 후보(candidates) 확인
   - 없으면 에러 처리 및 DB 로그 저장

10. **결과 이미지 추출**
    - 응답에서 `inline_data`로 포함된 이미지 추출
    - 텍스트 응답도 추출 (있을 경우)

11. **성공 케이스 처리**
    - 결과 이미지를 base64로 인코딩
    - 결과 이미지를 파일 시스템에 저장 (`uploads/result_{timestamp}.png`)
    - 처리 시간 계산
    - DB에 로그 저장:
      - 모델명: `gemini-2.5-flash-image`
      - API명: `Gemini API`
      - 프롬프트
      - 입력 이미지 경로들
      - 결과 이미지 경로
      - 성공 여부: `True`
      - 처리 시간
    - JSON 응답 반환

12. **실패 케이스 처리**
    - 이미지가 생성되지 않은 경우
    - API 응답이 없는 경우
    - 예외 발생 시
    - 모든 실패 케이스에서:
      - 처리 시간 계산
      - DB에 로그 저장 (성공: `False`, 에러 메시지 포함)
      - 에러 응답 반환

13. **에러 처리**
    - try-except로 모든 예외 처리
    - traceback 정보 포함
    - DB 로그 저장 (에러 정보 포함)

**응답 형식**:
```json
{
  "success": true,
  "person_image": "data:image/png;base64,...",
  "dress_image": "data:image/png;base64,...",
  "result_image": "data:image/png;base64,...",
  "message": "이미지 합성이 완료되었습니다.",
  "gemini_response": "텍스트 응답 (있는 경우)"
}
```

---

## 사용 모델 및 API

### 누끼 기능
- **모델**: SegFormer B2 Clothes Segmentation
- **모델 ID**: `mattmdjaga/segformer_b2_clothes`
- **라이브러리**: Hugging Face Transformers
- **프로세서**: `SegformerImageProcessor`
- **모델 타입**: `AutoModelForSemanticSegmentation`
- **감지 가능 레이블**: 18개 (배경, 모자, 머리, 상의, 치마, 바지, 드레스 등)

### 합성 기능
- **API**: Google Gemini API
- **모델**: `gemini-2.5-flash-image`
- **라이브러리**: `google.genai`
- **클라이언트**: `genai.Client`
- **입력**: 이미지 2개 + 텍스트 프롬프트
- **출력**: 합성된 이미지 (PNG 형식)

---

## 데이터 흐름도

### 누끼 기능 데이터 흐름
```
이미지 파일 업로드
    ↓
이미지 전처리 (RGB 변환)
    ↓
SegFormer 모델 추론
    ↓
업샘플링 (원본 크기로 조정)
    ↓
세그멘테이션 마스크 생성
    ↓
RGBA 누끼 이미지 생성
    ↓
Base64 인코딩
    ↓
JSON 응답 반환
```

### 합성 기능 데이터 흐름
```
사람 이미지 + 드레스 이미지 업로드
    ↓
이미지 파일 시스템 저장
    ↓
Base64 인코딩 (원본용)
    ↓
Gemini Client 생성
    ↓
프롬프트 생성
    ↓
Gemini API 호출 (이미지 2개 + 프롬프트)
    ↓
결과 이미지 추출
    ↓
결과 이미지 파일 시스템 저장
    ↓
DB 로그 저장
    ↓
Base64 인코딩 (결과용)
    ↓
JSON 응답 반환
```

---

## 기술 스택

- **백엔드 프레임워크**: FastAPI
- **딥러닝 프레임워크**: PyTorch
- **이미지 처리**: PIL (Pillow)
- **수치 계산**: NumPy
- **세그멘테이션 모델**: SegFormer (Transformers)
- **이미지 생성 API**: Google Gemini 2.5 Flash Image
- **데이터베이스**: MySQL (로그 저장용)
- **파일 저장**: 로컬 파일 시스템 (`uploads/`)

---

## 참고사항

1. **누끼 기능**은 서버 시작 시 모델을 로드하여 전역 변수로 저장합니다. (`@app.on_event("startup")`)
2. **합성 기능**은 모든 요청마다 DB에 로그를 저장합니다 (성공/실패 관계없이).
3. 저장되는 이미지 파일명은 타임스탬프 기반으로 중복을 방지합니다.
4. DB 로그는 관리자 페이지(`/admin`)에서 확인할 수 있습니다.


