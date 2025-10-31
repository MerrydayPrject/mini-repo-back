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
import time
import pymysql
from datetime import datetime
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
app.mount("/images", StaticFiles(directory="images"), name="images")
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

# ===================== DB 연결 함수 =====================

def get_db_connection():
    """MySQL 데이터베이스 연결 반환"""
    try:
        connection = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER", "devuser"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "marryday"),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        print(f"DB 연결 오류: {e}")
        return None

def init_database():
    """데이터베이스 테이블 생성"""
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 테이블 생성 건너뜀")
        return
    
    try:
        with connection.cursor() as cursor:
            # composition_logs 테이블 생성
            create_table_query = """
            CREATE TABLE IF NOT EXISTS composition_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name VARCHAR(100) NOT NULL,
                api_name VARCHAR(50) NOT NULL,
                prompt TEXT NOT NULL,
                person_image_path VARCHAR(255) NOT NULL,
                dress_image_path VARCHAR(255) NOT NULL,
                result_image_path VARCHAR(255),
                success BOOLEAN NOT NULL,
                processing_time FLOAT,
                error_message TEXT,
                INDEX idx_created_at (created_at),
                INDEX idx_success (success)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            cursor.execute(create_table_query)
            connection.commit()
            print("DB 테이블 생성 완료: composition_logs")
            
            # dress_info 테이블 생성
            create_dress_info_table = """
            CREATE TABLE IF NOT EXISTS dress_info (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_name VARCHAR(255) NOT NULL,
                style VARCHAR(100) NOT NULL,
                INDEX idx_image_name (image_name),
                INDEX idx_style (style)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            cursor.execute(create_dress_info_table)
            connection.commit()
            print("DB 테이블 생성 완료: dress_info")
    except Exception as e:
        print(f"테이블 생성 오류: {e}")
    finally:
        connection.close()

def save_composition_log(
    model_name: str,
    api_name: str,
    prompt: str,
    person_image_path: str,
    dress_image_path: str,
    result_image_path: Optional[str],
    success: bool,
    processing_time: float,
    error_message: Optional[str] = None
) -> Optional[int]:
    """합성 로그를 DB에 저장"""
    connection = get_db_connection()
    if not connection:
        print("DB 연결 실패 - 로그 저장 건너뜀")
        return None
    
    try:
        with connection.cursor() as cursor:
            insert_query = """
            INSERT INTO composition_logs 
            (model_name, api_name, prompt, person_image_path, dress_image_path, 
             result_image_path, success, processing_time, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                model_name, api_name, prompt, person_image_path, dress_image_path,
                result_image_path, success, processing_time, error_message
            ))
            connection.commit()
            log_id = cursor.lastrowid
            print(f"DB 로그 저장 완료: ID={log_id}")
            return log_id
    except Exception as e:
        print(f"로그 저장 오류: {e}")
        return None
    finally:
        connection.close()

def save_uploaded_image(image: Image.Image, prefix: str) -> str:
    """이미지를 파일 시스템에 저장"""
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}.png"
    filepath = Path("uploads") / filename
    image.save(filepath)
    return str(filepath)

def detect_style_from_filename(filename: str) -> Optional[str]:
    """
    이미지 파일명에서 스타일을 감지
    
    Args:
        filename: 이미지 파일명 (예: "Adress1.jpg", "Mini_dress.png")
    
    Returns:
        감지된 스타일 문자열 또는 None (감지 실패 시)
        - "A라인": "A"로 시작
        - "미니드레스": "Mini" 포함 (대소문자 구분 없음)
        - "벨라인": "B"로 시작
        - "프린세스": "P"로 시작
        - None: 위 조건에 해당하지 않으면 삽입 불가
    """
    filename_upper = filename.upper()
    
    # 1. "A"로 시작하는지 확인
    if filename_upper.startswith("A"):
        return "A라인"
    
    # 2. "Mini" 포함 여부 확인 (대소문자 구분 없음)
    if "MINI" in filename_upper:
        return "미니드레스"
    
    # 3. "B"로 시작하는지 확인
    if filename_upper.startswith("B"):
        return "벨라인"
    
    # 4. "P"로 시작하는지 확인
    if filename_upper.startswith("P"):
        return "프린세스"
    
    # 5. 위 조건에 해당하지 않으면 None 반환 (삽입 불가)
    return None

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
    """애플리케이션 시작 시 모델 로드 및 DB 초기화"""
    global processor, model
    print("SegFormer 모델 로딩 중...")
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model.eval()
    print("모델 로딩 완료!")
    
    # DB 초기화
    print("데이터베이스 초기화 중...")
    init_database()

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
    # 처리 시작 시간
    start_time = time.time()
    person_image_path = None
    dress_image_path = None
    result_image_path = None
    success = False
    error_message = None
    
    try:
        # .env에서 API 키 가져오기
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            error_message = "API key not found"
            return JSONResponse({
                "success": False,
                "error": error_message,
                "message": ".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다."
            }, status_code=500)
        
        # 이미지 읽기
        person_contents = await person_image.read()
        dress_contents = await dress_image.read()
        
        person_img = Image.open(io.BytesIO(person_contents))
        dress_img = Image.open(io.BytesIO(dress_contents))
        
        # 입력 이미지들을 파일 시스템에 저장
        person_image_path = save_uploaded_image(person_img, "person")
        dress_image_path = save_uploaded_image(dress_img, "dress")
        
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
            error_message = "No response from Gemini"
            processing_time = time.time() - start_time
            # DB 로그 저장
            save_composition_log(
                model_name="gemini-2.5-flash-image",
                api_name="Gemini API",
                prompt=text_input,
                person_image_path=person_image_path or "",
                dress_image_path=dress_image_path or "",
                result_image_path=None,
                success=False,
                processing_time=processing_time,
                error_message=error_message
            )
            return JSONResponse({
                "success": False,
                "error": error_message,
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
            
            # 결과 이미지를 파일 시스템에 저장
            result_img = Image.open(io.BytesIO(image_parts[0]))
            result_image_path = save_uploaded_image(result_img, "result")
            success = True
            processing_time = time.time() - start_time
            
            # DB 로그 저장
            save_composition_log(
                model_name="gemini-2.5-flash-image",
                api_name="Gemini API",
                prompt=text_input,
                person_image_path=person_image_path or "",
                dress_image_path=dress_image_path or "",
                result_image_path=result_image_path,
                success=True,
                processing_time=processing_time,
                error_message=None
            )
            
            return JSONResponse({
                "success": True,
                "person_image": f"data:image/png;base64,{person_base64}",
                "dress_image": f"data:image/png;base64,{dress_base64}",
                "result_image": f"data:image/png;base64,{result_image_base64}",
                "message": "이미지 합성이 완료되었습니다.",
                "gemini_response": result_text
            })
        else:
            error_message = f"No image generated. Response: {result_text}"
            processing_time = time.time() - start_time
            # DB 로그 저장
            save_composition_log(
                model_name="gemini-2.5-flash-image",
                api_name="Gemini API",
                prompt=text_input,
                person_image_path=person_image_path or "",
                dress_image_path=dress_image_path or "",
                result_image_path=None,
                success=False,
                processing_time=processing_time,
                error_message=error_message
            )
            return JSONResponse({
                "success": False,
                "error": "No image generated",
                "message": "Gemini API가 이미지를 생성하지 못했습니다. 응답: " + result_text,
                "gemini_response": result_text
            }, status_code=500)
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        error_message = str(e)
        processing_time = time.time() - start_time
        
        # DB 로그 저장 (에러 발생 시에도)
        save_composition_log(
            model_name="gemini-2.5-flash-image",
            api_name="Gemini API",
            prompt="N/A",
            person_image_path=person_image_path or "",
            dress_image_path=dress_image_path or "",
            result_image_path=None,
            success=False,
            processing_time=processing_time,
            error_message=error_message
        )
        
        return JSONResponse({
            "success": False,
            "error": error_message,
            "error_detail": error_detail,
            "message": f"이미지 합성 중 오류 발생: {error_message}"
        }, status_code=500)

@app.get("/gemini-test", response_class=HTMLResponse, tags=["Web Interface"])
async def gemini_test_page(request: Request):
    """
    Gemini 이미지 합성 테스트 페이지
    
    사람 이미지와 드레스 이미지를 업로드하여 합성 결과를 테스트할 수 있는 페이지
    """
    return templates.TemplateResponse("gemini_test.html", {"request": request})

# ===================== 관리자 API =====================

@app.get("/api/admin/logs", tags=["관리자"])
async def get_composition_logs(
    page: int = Query(1, ge=1, description="페이지 번호"),
    limit: int = Query(20, ge=1, le=100, description="페이지당 항목 수")
):
    """
    로그 목록 조회
    
    Args:
        page: 페이지 번호 (1부터 시작)
        limit: 페이지당 항목 수
    
    Returns:
        JSONResponse: 로그 목록, 총 개수, 페이지 정보
    """
    connection = get_db_connection()
    if not connection:
        return JSONResponse({
            "success": False,
            "error": "Database connection failed",
            "message": "데이터베이스 연결에 실패했습니다."
        }, status_code=500)
    
    try:
        with connection.cursor() as cursor:
            # 전체 개수 조회
            cursor.execute("SELECT COUNT(*) as total FROM composition_logs")
            total = cursor.fetchone()["total"]
            
            # 페이지네이션 적용하여 로그 조회
            offset = (page - 1) * limit
            cursor.execute("""
                SELECT id, created_at, model_name, api_name, success, processing_time
                FROM composition_logs
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
            logs = cursor.fetchall()
            
            return JSONResponse({
                "success": True,
                "data": logs,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total,
                    "total_pages": (total + limit - 1) // limit
                }
            })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"로그 조회 중 오류 발생: {str(e)}"
        }, status_code=500)
    finally:
        connection.close()

@app.get("/api/admin/logs/{log_id}", tags=["관리자"])
async def get_composition_log(log_id: int):
    """
    특정 로그 상세 조회
    
    Args:
        log_id: 로그 ID
    
    Returns:
        JSONResponse: 로그의 모든 정보
    """
    connection = get_db_connection()
    if not connection:
        return JSONResponse({
            "success": False,
            "error": "Database connection failed",
            "message": "데이터베이스 연결에 실패했습니다."
        }, status_code=500)
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM composition_logs WHERE id = %s
            """, (log_id,))
            log = cursor.fetchone()
            
            if not log:
                return JSONResponse({
                    "success": False,
                    "error": "Not found",
                    "message": f"ID {log_id}의 로그를 찾을 수 없습니다."
                }, status_code=404)
            
            return JSONResponse({
                "success": True,
                "data": log
            })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"로그 조회 중 오류 발생: {str(e)}"
        }, status_code=500)
    finally:
        connection.close()

@app.get("/api/admin/stats", tags=["관리자"])
async def get_statistics():
    """
    통계 정보 조회
    
    Returns:
        JSONResponse: 통계 정보
    """
    connection = get_db_connection()
    if not connection:
        return JSONResponse({
            "success": False,
            "error": "Database connection failed",
            "message": "데이터베이스 연결에 실패했습니다."
        }, status_code=500)
    
    try:
        with connection.cursor() as cursor:
            # 전체 합성 횟수
            cursor.execute("SELECT COUNT(*) as total FROM composition_logs")
            total = cursor.fetchone()["total"]
            
            # 성공/실패 건수
            cursor.execute("SELECT success, COUNT(*) as count FROM composition_logs GROUP BY success")
            success_stats = {row["success"]: row["count"] for row in cursor.fetchall()}
            success_count = success_stats.get(True, 0)
            fail_count = success_stats.get(False, 0)
            
            # 평균 처리 시간
            cursor.execute("SELECT AVG(processing_time) as avg_time FROM composition_logs WHERE processing_time IS NOT NULL")
            avg_time = cursor.fetchone()["avg_time"]
            
            # 오늘 합성 횟수
            cursor.execute("""
                SELECT COUNT(*) as today_count FROM composition_logs 
                WHERE DATE(created_at) = CURDATE()
            """)
            today_count = cursor.fetchone()["today_count"]
            
            # 이번 주 합성 횟수
            cursor.execute("""
                SELECT COUNT(*) as week_count FROM composition_logs 
                WHERE YEARWEEK(created_at, 1) = YEARWEEK(CURDATE(), 1)
            """)
            week_count = cursor.fetchone()["week_count"]
            
            # 이번 달 합성 횟수
            cursor.execute("""
                SELECT COUNT(*) as month_count FROM composition_logs 
                WHERE YEAR(created_at) = YEAR(CURDATE()) AND MONTH(created_at) = MONTH(CURDATE())
            """)
            month_count = cursor.fetchone()["month_count"]
            
            return JSONResponse({
                "success": True,
                "data": {
                    "total": total,
                    "success": success_count,
                    "failed": fail_count,
                    "success_rate": round((success_count / total * 100) if total > 0 else 0, 2),
                    "average_processing_time": round(float(avg_time) if avg_time else 0, 3),
                    "today": today_count,
                    "this_week": week_count,
                    "this_month": month_count
                }
            })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"통계 조회 중 오류 발생: {str(e)}"
        }, status_code=500)
    finally:
        connection.close()

# ===================== 드레스 관리 API =====================

@app.get("/api/admin/dresses/scan", tags=["드레스 관리"])
async def scan_dress_images():
    """
    이미지 폴더를 스캔하여 드레스 이미지 목록 조회
    
    Returns:
        JSONResponse: 스캔된 이미지 목록, 감지된 스타일, DB 존재 여부
    """
    try:
        # 프론트엔드의 Image 폴더 경로 (상대 경로)
        image_folder = Path("../mini-repo-front/public/Image")
        
        if not image_folder.exists():
            return JSONResponse({
                "success": False,
                "error": "Image folder not found",
                "message": f"이미지 폴더를 찾을 수 없습니다: {image_folder.absolute()}"
            }, status_code=404)
        
        # 이미지 파일 확장자
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        
        # DB에 존재하는 이미지 목록 조회
        connection = get_db_connection()
        existing_images = set()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT image_name FROM dress_info")
                    existing_images = {row["image_name"] for row in cursor.fetchall()}
            except Exception as e:
                print(f"DB 조회 오류: {e}")
            finally:
                connection.close()
        
        # 이미지 파일 스캔
        scanned_images = []
        for file_path in image_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                filename = file_path.name
                style = detect_style_from_filename(filename)
                exists_in_db = filename in existing_images
                
                scanned_images.append({
                    "image_name": filename,
                    "style": style,
                    "style_detected": style is not None,
                    "exists_in_db": exists_in_db,
                    "image_url": f"/Image/{filename}"
                })
        
        # 스타일이 감지된 것부터 정렬
        scanned_images.sort(key=lambda x: (x["style_detected"], x["image_name"]), reverse=True)
        
        return JSONResponse({
            "success": True,
            "data": scanned_images,
            "total": len(scanned_images),
            "detected": sum(1 for img in scanned_images if img["style_detected"]),
            "undetected": sum(1 for img in scanned_images if not img["style_detected"])
        })
        
    except Exception as e:
        import traceback
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"이미지 스캔 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.post("/api/admin/dresses/bulk-insert", tags=["드레스 관리"])
async def bulk_insert_dresses(request: Request):
    """
    여러 드레스 이미지를 일괄 삽입
    
    Request Body:
        {
            "images": [
                {"image_name": "Adress1.jpg", "style": "A라인"},
                ...
            ]
        }
    
    Returns:
        JSONResponse: 삽입 결과
    """
    try:
        body = await request.json()
        images = body.get("images", [])
        
        if not images:
            return JSONResponse({
                "success": False,
                "error": "No images provided",
                "message": "삽입할 이미지가 없습니다."
            }, status_code=400)
        
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)
        
        inserted_count = 0
        skipped_count = 0
        error_count = 0
        errors = []
        
        try:
            with connection.cursor() as cursor:
                for image_data in images:
                    image_name = image_data.get("image_name")
                    style = image_data.get("style")
                    
                    # 스타일이 없는 경우 건너뛰기
                    if not style or not image_name:
                        skipped_count += 1
                        continue
                    
                    try:
                        # 중복 체크
                        cursor.execute("SELECT id FROM dress_info WHERE image_name = %s", (image_name,))
                        if cursor.fetchone():
                            skipped_count += 1
                            continue
                        
                        # 삽입
                        cursor.execute(
                            "INSERT INTO dress_info (image_name, style) VALUES (%s, %s)",
                            (image_name, style)
                        )
                        inserted_count += 1
                    except Exception as e:
                        error_count += 1
                        errors.append({
                            "image_name": image_name,
                            "error": str(e)
                        })
                
                connection.commit()
        except Exception as e:
            connection.rollback()
            return JSONResponse({
                "success": False,
                "error": str(e),
                "message": f"삽입 중 오류 발생: {str(e)}"
            }, status_code=500)
        finally:
            connection.close()
        
        return JSONResponse({
            "success": True,
            "inserted": inserted_count,
            "skipped": skipped_count,
            "errors": error_count,
            "error_details": errors,
            "message": f"{inserted_count}개 이미지 삽입 완료, {skipped_count}개 건너뜀, {error_count}개 오류"
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"일괄 삽입 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.get("/api/admin/dresses", tags=["드레스 관리"])
async def get_dresses():
    """
    DB에 저장된 드레스 목록 조회
    
    Returns:
        JSONResponse: 드레스 목록
    """
    connection = get_db_connection()
    if not connection:
        return JSONResponse({
            "success": False,
            "error": "Database connection failed",
            "message": "데이터베이스 연결에 실패했습니다."
        }, status_code=500)
    
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, image_name, style FROM dress_info ORDER BY id DESC")
            dresses = cursor.fetchall()
            
            return JSONResponse({
                "success": True,
                "data": dresses,
                "total": len(dresses)
            })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"드레스 목록 조회 중 오류 발생: {str(e)}"
        }, status_code=500)
    finally:
        connection.close()

@app.post("/api/admin/dresses", tags=["드레스 관리"])
async def add_dress(request: Request):
    """
    드레스 정보 단일 추가
    
    Request Body:
        {
            "image_name": "Adress1.jpg",
            "style": "A라인"
        }
    
    Returns:
        JSONResponse: 삽입 결과
    """
    try:
        body = await request.json()
        image_name = body.get("image_name")
        style = body.get("style")
        
        if not image_name or not style:
            return JSONResponse({
                "success": False,
                "error": "Missing required fields",
                "message": "image_name과 style은 필수 입력 항목입니다."
            }, status_code=400)
        
        # 파일명으로부터 스타일 감지하여 검증
        detected_style = detect_style_from_filename(image_name)
        if not detected_style:
            return JSONResponse({
                "success": False,
                "error": "Invalid image name",
                "message": f"이미지 파일명 '{image_name}'에서 스타일을 감지할 수 없습니다. 파일명은 A, Mini, B, P로 시작해야 합니다."
            }, status_code=400)
        
        # 감지된 스타일과 제공된 스타일이 일치하는지 확인
        if detected_style != style:
            return JSONResponse({
                "success": False,
                "error": "Style mismatch",
                "message": f"파일명에서 감지된 스타일({detected_style})과 제공된 스타일({style})이 일치하지 않습니다."
            }, status_code=400)
        
        connection = get_db_connection()
        if not connection:
            return JSONResponse({
                "success": False,
                "error": "Database connection failed",
                "message": "데이터베이스 연결에 실패했습니다."
            }, status_code=500)
        
        try:
            with connection.cursor() as cursor:
                # 중복 체크
                cursor.execute("SELECT id FROM dress_info WHERE image_name = %s", (image_name,))
                if cursor.fetchone():
                    return JSONResponse({
                        "success": False,
                        "error": "Duplicate image name",
                        "message": f"이미지명 '{image_name}'이 이미 존재합니다."
                    }, status_code=400)
                
                # 삽입
                cursor.execute(
                    "INSERT INTO dress_info (image_name, style) VALUES (%s, %s)",
                    (image_name, style)
                )
                connection.commit()
                dress_id = cursor.lastrowid
                
                return JSONResponse({
                    "success": True,
                    "data": {
                        "id": dress_id,
                        "image_name": image_name,
                        "style": style
                    },
                    "message": "드레스 정보가 성공적으로 추가되었습니다."
                })
        except Exception as e:
            connection.rollback()
            return JSONResponse({
                "success": False,
                "error": str(e),
                "message": f"드레스 추가 중 오류 발생: {str(e)}"
            }, status_code=500)
        finally:
            connection.close()
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": f"드레스 추가 중 오류 발생: {str(e)}"
        }, status_code=500)

@app.get("/admin/dress-insert", response_class=HTMLResponse, tags=["관리자"])
async def dress_insert_page(request: Request):
    """
    드레스 이미지 삽입 관리자 페이지
    """
    return templates.TemplateResponse("dress_insert.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse, tags=["관리자"])
async def admin_page(request: Request):
    """
    관리자 페이지
    
    로그 목록과 통계를 확인할 수 있는 관리자 페이지
    """
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/admin/dress-manage", response_class=HTMLResponse, tags=["관리자"])
async def dress_manage_page(request: Request):
    """
    드레스 관리자 페이지
    
    드레스 정보 목록 조회 및 추가가 가능한 관리자 페이지
    """
    return templates.TemplateResponse("dress_manage.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

