from fastapi import FastAPI, File, UploadFile, Request, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import io
import base64
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# FastAPI 앱 초기화
app = FastAPI(
    title="의류 세그멘테이션 API",
    description="SegFormer 모델을 사용한 고급 의류 세그멘테이션 서비스. 웨딩드레스를 포함한 다양한 의류 항목을 감지하고 배경을 제거할 수 있습니다.",
    version="1.0.0",
    contact={
        "name": "API Support",
        "url": "https://github.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # 프론트엔드 주소들
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 디렉토리 생성
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 전역 변수로 모델 저장
processor = None
model = None

# 레이블 정보
LABELS = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses",
    4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress",
    8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face",
    12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
    16: "Bag", 17: "Scarf"
}

# Pydantic 모델
class LabelInfo(BaseModel):
    """레이블 정보 모델"""
    id: int = Field(..., description="레이블 ID")
    name: str = Field(..., description="레이블 이름")
    percentage: float = Field(..., description="이미지 내 해당 레이블이 차지하는 비율 (%)")

class SegmentationResponse(BaseModel):
    """세그멘테이션 응답 모델"""
    success: bool = Field(..., description="처리 성공 여부")
    original_image: str = Field(..., description="원본 이미지 (base64)")
    result_image: str = Field(..., description="결과 이미지 (base64)")
    detected_labels: List[LabelInfo] = Field(..., description="감지된 레이블 목록")
    message: str = Field(..., description="처리 결과 메시지")

class ErrorResponse(BaseModel):
    """에러 응답 모델"""
    success: bool = Field(False, description="처리 성공 여부")
    error: str = Field(..., description="에러 메시지")
    message: str = Field(..., description="사용자 친화적 에러 메시지")

@app.on_event("startup")
async def load_model():
    """애플리케이션 시작 시 모델 로드"""
    global processor, model
    print("SegFormer 모델 로딩 중...")
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model.eval()
    print("모델 로딩 완료!")

@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def home(request: Request):
    """
    메인 웹 인터페이스
    
    웨딩드레스 누끼 서비스의 메인 페이지를 반환합니다.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/labels", tags=["정보"])
async def get_labels():
    """
    사용 가능한 모든 레이블 목록 조회
    
    SegFormer 모델이 감지할 수 있는 18개 의류/신체 부위 레이블 목록을 반환합니다.
    
    Returns:
        dict: 레이블 ID를 키로, 레이블 이름을 값으로 하는 딕셔너리
    """
    return {
        "labels": LABELS,
        "total_labels": len(LABELS),
        "description": "SegFormer B2 모델이 감지할 수 있는 레이블 목록"
    }

@app.post("/api/segment", tags=["세그멘테이션"])
async def segment_dress(file: UploadFile = File(..., description="세그멘테이션할 이미지 파일")):
    """
    드레스 세그멘테이션 (웨딩드레스 누끼)
    
    업로드된 이미지에서 드레스(레이블 7)를 감지하고 배경을 제거합니다.
    
    Args:
        file: 업로드할 이미지 파일 (JPG, PNG, GIF, WEBP 등)
    
    Returns:
        JSONResponse: 원본 이미지, 누끼 결과 이미지(투명 배경), 감지 정보
        
    Raises:
        500: 이미지 처리 중 오류 발생
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        
        # 원본 이미지를 base64로 인코딩
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        # 모델 추론
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
        
        # 업샘플링
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=original_size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        
        # 세그멘테이션 마스크 생성
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # 드레스 마스크 생성 (레이블 7: Dress)
        dress_mask = (pred_seg == 7).astype(np.uint8) * 255
        
        # 원본 이미지를 numpy 배열로 변환
        image_array = np.array(image)
        
        # 누끼 이미지 생성 (RGBA)
        result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
        result_image[:, :, :3] = image_array  # RGB 채널
        result_image[:, :, 3] = dress_mask    # 알파 채널
        
        # PIL 이미지로 변환
        result_pil = Image.fromarray(result_image, mode='RGBA')
        
        # 결과 이미지를 base64로 인코딩
        buffered_result = io.BytesIO()
        result_pil.save(buffered_result, format="PNG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
        
        # 드레스가 감지되었는지 확인
        dress_pixels = int(np.sum(pred_seg == 7))
        total_pixels = int(pred_seg.size)
        dress_percentage = float((dress_pixels / total_pixels) * 100)
        
        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "dress_detected": bool(dress_pixels > 0),
            "dress_percentage": round(dress_percentage, 2),
            "message": f"드레스 영역: {dress_percentage:.2f}% 감지됨" if dress_pixels > 0 else "드레스가 감지되지 않았습니다."
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.get("/health", tags=["정보"])
async def health_check():
    """
    서버 상태 확인
    
    서버와 모델의 로딩 상태를 확인합니다.
    
    Returns:
        dict: 서버 상태 및 모델 로딩 여부
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None and processor is not None,
        "model_name": "mattmdjaga/segformer_b2_clothes",
        "version": "1.0.0"
    }

@app.post("/api/segment-custom", tags=["세그멘테이션"])
async def segment_custom(
    file: UploadFile = File(..., description="세그멘테이션할 이미지 파일"),
    labels: str = Query(..., description="추출할 레이블 ID (쉼표로 구분, 예: 4,5,6,7)")
):
    """
    커스텀 레이블 세그멘테이션
    
    지정한 레이블들만 추출하여 배경을 제거합니다.
    
    Args:
        file: 업로드할 이미지 파일
        labels: 추출할 레이블 ID (쉼표로 구분)
                예: "7" (드레스만), "4,5,6,7" (상의, 치마, 바지, 드레스)
    
    Returns:
        JSONResponse: 원본 이미지, 선택한 레이블만 추출한 결과 이미지
        
    Example:
        - labels="7": 드레스만 추출
        - labels="4,6": 상의와 바지만 추출
        - labels="1,2,11": 모자, 머리, 얼굴만 추출
    """
    try:
        # 레이블 파싱
        label_ids = [int(l.strip()) for l in labels.split(",")]
        
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        
        # 원본 이미지를 base64로 인코딩
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        # 모델 추론
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
        
        # 업샘플링
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=original_size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        # 세그멘테이션 마스크 생성
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # 선택한 레이블들의 마스크 생성
        combined_mask = np.zeros_like(pred_seg, dtype=bool)
        for label_id in label_ids:
            combined_mask |= (pred_seg == label_id)
        
        mask = combined_mask.astype(np.uint8) * 255
        
        # 원본 이미지를 numpy 배열로 변환
        image_array = np.array(image)
        
        # 누끼 이미지 생성 (RGBA)
        result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
        result_image[:, :, :3] = image_array
        result_image[:, :, 3] = mask
        
        # PIL 이미지로 변환
        result_pil = Image.fromarray(result_image, mode='RGBA')
        
        # 결과 이미지를 base64로 인코딩
        buffered_result = io.BytesIO()
        result_pil.save(buffered_result, format="PNG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
        
        # 각 레이블의 픽셀 수 계산
        detected_labels = []
        total_pixels = int(pred_seg.size)
        for label_id in label_ids:
            pixels = int(np.sum(pred_seg == label_id))
            if pixels > 0:
                detected_labels.append({
                    "id": label_id,
                    "name": LABELS.get(label_id, "Unknown"),
                    "percentage": round((pixels / total_pixels) * 100, 2)
                })
        
        total_detected = int(np.sum(combined_mask))
        
        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "requested_labels": [{"id": lid, "name": LABELS.get(lid, "Unknown")} for lid in label_ids],
            "detected_labels": detected_labels,
            "total_percentage": round((total_detected / total_pixels) * 100, 2),
            "message": f"{len(detected_labels)}개의 레이블 감지됨" if detected_labels else "선택한 레이블이 감지되지 않았습니다."
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/analyze", tags=["분석"])
async def analyze_image(file: UploadFile = File(..., description="분석할 이미지 파일")):
    """
    이미지 전체 분석
    
    이미지에서 모든 레이블을 감지하고 각 레이블의 비율을 분석합니다.
    누끼 처리 없이 분석 정보만 반환합니다.
    
    Args:
        file: 분석할 이미지 파일
    
    Returns:
        JSONResponse: 감지된 모든 레이블과 비율 정보
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        
        # 모델 추론
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
        
        # 업샘플링
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=original_size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        # 세그멘테이션 마스크 생성
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # 각 레이블의 픽셀 수 계산
        total_pixels = int(pred_seg.size)
        detected_labels = []
        
        for label_id, label_name in LABELS.items():
            pixels = int(np.sum(pred_seg == label_id))
            percentage = round((pixels / total_pixels) * 100, 2)
            if pixels > 0:
                detected_labels.append({
                    "id": label_id,
                    "name": label_name,
                    "pixels": pixels,
                    "percentage": percentage
                })
        
        # 비율 순으로 정렬
        detected_labels.sort(key=lambda x: x["percentage"], reverse=True)
        
        return JSONResponse({
            "success": True,
            "image_size": {"width": original_size[0], "height": original_size[1]},
            "total_pixels": total_pixels,
            "detected_labels": detected_labels,
            "total_detected": len(detected_labels),
            "message": f"총 {len(detected_labels)}개의 레이블 감지됨"
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/remove-background", tags=["세그멘테이션"])
async def remove_background(file: UploadFile = File(..., description="배경을 제거할 이미지 파일")):
    """
    전체 배경 제거 (인물만 추출)
    
    배경(레이블 0)을 제거하고 인물과 의류만 남깁니다.
    
    Args:
        file: 배경을 제거할 이미지 파일
    
    Returns:
        JSONResponse: 배경이 제거된 이미지 (투명 배경)
    """
    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size
        
        # 원본 이미지를 base64로 인코딩
        buffered_original = io.BytesIO()
        image.save(buffered_original, format="PNG")
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        # 모델 추론
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
        
        # 업샘플링
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=original_size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        # 세그멘테이션 마스크 생성
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # 배경이 아닌 모든 것을 포함하는 마스크
        mask = (pred_seg != 0).astype(np.uint8) * 255
        
        # 원본 이미지를 numpy 배열로 변환
        image_array = np.array(image)
        
        # 누끼 이미지 생성 (RGBA)
        result_image = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
        result_image[:, :, :3] = image_array
        result_image[:, :, 3] = mask
        
        # PIL 이미지로 변환
        result_pil = Image.fromarray(result_image, mode='RGBA')
        
        # 결과 이미지를 base64로 인코딩
        buffered_result = io.BytesIO()
        result_pil.save(buffered_result, format="PNG")
        result_base64 = base64.b64encode(buffered_result.getvalue()).decode()
        
        # 배경이 아닌 픽셀 수 계산
        foreground_pixels = int(np.sum(pred_seg != 0))
        total_pixels = int(pred_seg.size)
        foreground_percentage = round((foreground_pixels / total_pixels) * 100, 2)
        
        return JSONResponse({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "result_image": f"data:image/png;base64,{result_base64}",
            "foreground_percentage": foreground_percentage,
            "message": f"배경 제거 완료 (인물 영역: {foreground_percentage}%)"
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"처리 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/compose-dress", tags=["Gemini 이미지 합성"])
async def compose_dress(
    person_image: UploadFile = File(..., description="사람 이미지 파일"),
    dress_image: UploadFile = File(..., description="드레스 이미지 파일")
):
    """
    Gemini API를 사용한 사람과 드레스 이미지 합성
    
    사람 이미지와 드레스 이미지를 받아서 Gemini API를 통해
    사람이 드레스를 입은 것처럼 합성된 이미지를 생성합니다.
    
    Args:
        person_image: 사람 이미지 파일
        dress_image: 드레스 이미지 파일
    
    Returns:
        JSONResponse: 합성된 이미지 (base64)
    """
    try:
        # .env에서 API 키 가져오기
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse({
                "success": False,
                "error": "API key not found",
                "message": ".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다."
            }, status_code=500)
        
        # 이미지 읽기
        person_contents = await person_image.read()
        dress_contents = await dress_image.read()
        
        person_img = Image.open(io.BytesIO(person_contents))
        dress_img = Image.open(io.BytesIO(dress_contents))
        
        # 원본 이미지들을 base64로 변환
        person_buffered = io.BytesIO()
        person_img.save(person_buffered, format="PNG")
        person_base64 = base64.b64encode(person_buffered.getvalue()).decode()
        
        dress_buffered = io.BytesIO()
        dress_img.save(dress_buffered, format="PNG")
        dress_base64 = base64.b64encode(dress_buffered.getvalue()).decode()
        
        # Gemini Client 생성 (공식 문서와 동일한 방식)
        client = genai.Client(api_key=api_key)
        
        # 프롬프트 생성 (얼굴과 체형 유지 강조)
        text_input = """IMPORTANT: You must preserve the person's identity completely.

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
ONLY apply the dress design, color, and style onto the existing person."""
        
        # Gemini API 호출 (공식 문서 방식: dress, model, text 순서)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[dress_img, person_img, text_input]
        )
        
        # 응답 확인
        if not response.candidates or len(response.candidates) == 0:
            return JSONResponse({
                "success": False,
                "error": "No response from Gemini",
                "message": "Gemini API가 응답을 생성하지 못했습니다. 이미지가 안전 정책에 위배되거나 모델이 이미지를 생성할 수 없습니다."
            }, status_code=500)
        
        # 응답에서 이미지 추출 (예시 코드와 동일한 방식)
        image_parts = [
            part.inline_data.data
            for part in response.candidates[0].content.parts
            if hasattr(part, 'inline_data') and part.inline_data
        ]
        
        # 텍스트 응답도 추출
        result_text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                result_text += part.text
        
        if image_parts:
            # 첫 번째 이미지를 base64로 인코딩
            result_image_base64 = base64.b64encode(image_parts[0]).decode()
            
            return JSONResponse({
                "success": True,
                "person_image": f"data:image/png;base64,{person_base64}",
                "dress_image": f"data:image/png;base64,{dress_base64}",
                "result_image": f"data:image/png;base64,{result_image_base64}",
                "message": "이미지 합성이 완료되었습니다.",
                "gemini_response": result_text
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "No image generated",
                "message": "Gemini API가 이미지를 생성하지 못했습니다. 응답: " + result_text,
                "gemini_response": result_text
            }, status_code=500)
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "error_detail": error_detail,
            "message": f"이미지 합성 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.get("/gemini-test", response_class=HTMLResponse, tags=["Web Interface"])
async def gemini_test_page(request: Request):
    """
    Gemini 이미지 합성 테스트 페이지
    
    사람 이미지와 드레스 이미지를 업로드하여 합성 결과를 테스트할 수 있는 페이지
    """
    return templates.TemplateResponse("gemini_test.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

