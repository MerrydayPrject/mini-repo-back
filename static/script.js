// DOM 요소
const uploadArea = document.getElementById('uploadArea');
const uploadButton = document.getElementById('uploadButton');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.querySelector('.upload-section');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');
const originalImage = document.getElementById('originalImage');
const resultImage = document.getElementById('resultImage');
const downloadButton = document.getElementById('downloadButton');
const resetButton = document.getElementById('resetButton');
const errorResetButton = document.getElementById('errorResetButton');
const errorText = document.getElementById('errorText');
const resultInfo = document.getElementById('resultInfo');

let currentResultImageData = null;

// 업로드 버튼 클릭 이벤트
uploadButton.addEventListener('click', () => {
    fileInput.click();
});

// 업로드 영역 클릭 이벤트
uploadArea.addEventListener('click', (e) => {
    if (e.target !== uploadButton) {
        fileInput.click();
    }
});

// 파일 선택 이벤트
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// 드래그 앤 드롭 이벤트
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    } else {
        showError('이미지 파일만 업로드할 수 있습니다.');
    }
});

// 파일 처리 함수
async function handleFile(file) {
    // 이미지 파일인지 확인
    if (!file.type.startsWith('image/')) {
        showError('이미지 파일만 업로드할 수 있습니다.');
        return;
    }

    // 파일 크기 확인 (10MB 제한)
    if (file.size > 10 * 1024 * 1024) {
        showError('파일 크기는 10MB 이하여야 합니다.');
        return;
    }

    // UI 업데이트
    uploadSection.style.display = 'none';
    loadingSection.style.display = 'block';
    errorSection.style.display = 'none';
    resultSection.style.display = 'none';

    try {
        // FormData 생성
        const formData = new FormData();
        formData.append('file', file);

        // API 호출
        const response = await fetch('/api/segment', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            showResult(data);
        } else {
            showError(data.message || '이미지 처리 중 오류가 발생했습니다.');
        }
    } catch (error) {
        showError('서버와 통신 중 오류가 발생했습니다: ' + error.message);
    }
}

// 결과 표시 함수
function showResult(data) {
    loadingSection.style.display = 'none';
    resultSection.style.display = 'block';

    // 이미지 설정
    originalImage.src = data.original_image;
    resultImage.src = data.result_image;
    currentResultImageData = data.result_image;

    // 결과 정보 표시
    if (data.dress_detected) {
        resultInfo.innerHTML = `✅ ${data.message}`;
        resultInfo.style.background = '#e8f5e9';
        resultInfo.style.color = '#2e7d32';
    } else {
        resultInfo.innerHTML = `⚠️ ${data.message}`;
        resultInfo.style.background = '#fff3e0';
        resultInfo.style.color = '#e65100';
    }
}

// 에러 표시 함수
function showError(message) {
    loadingSection.style.display = 'none';
    uploadSection.style.display = 'none';
    resultSection.style.display = 'none';
    errorSection.style.display = 'block';
    errorText.textContent = message;
}

// 다운로드 버튼 클릭 이벤트
downloadButton.addEventListener('click', () => {
    if (currentResultImageData) {
        // base64 이미지를 다운로드
        const link = document.createElement('a');
        link.href = currentResultImageData;
        link.download = `wedding_dress_${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
});

// 리셋 버튼 클릭 이벤트
resetButton.addEventListener('click', resetApp);
errorResetButton.addEventListener('click', resetApp);

function resetApp() {
    uploadSection.style.display = 'block';
    loadingSection.style.display = 'none';
    resultSection.style.display = 'none';
    errorSection.style.display = 'none';
    fileInput.value = '';
    currentResultImageData = null;
}

// 초기 로드 시 헬스 체크
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log('서버 상태:', data);
        
        if (!data.model_loaded) {
            console.warn('모델이 아직 로드 중입니다. 잠시 후 다시 시도해주세요.');
        }
    } catch (error) {
        console.error('서버 연결 실패:', error);
    }
}

// 페이지 로드 시 헬스 체크 실행
checkHealth();


