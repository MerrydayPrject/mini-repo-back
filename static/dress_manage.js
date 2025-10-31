// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    loadDresses();
    setupEventListeners();
});

// 이벤트 리스너 설정
function setupEventListeners() {
    const imageNameInput = document.getElementById('image-name-input');
    const addDressBtn = document.getElementById('add-dress-btn');
    const clearFormBtn = document.getElementById('clear-form-btn');
    const refreshBtn = document.getElementById('refresh-btn');

    // 이미지명 입력 시 스타일 자동 감지
    imageNameInput.addEventListener('input', handleImageNameChange);

    // 드레스 추가 버튼
    addDressBtn.addEventListener('click', handleAddDress);

    // 폼 초기화 버튼
    clearFormBtn.addEventListener('click', clearForm);

    // 새로고침 버튼
    refreshBtn.addEventListener('click', () => {
        loadDresses();
    });
}

// 이미지명 입력 시 스타일 감지
function handleImageNameChange(e) {
    const imageName = e.target.value.trim();
    const styleDisplay = document.getElementById('style-display');
    const addDressBtn = document.getElementById('add-dress-btn');

    if (!imageName) {
        styleDisplay.value = '';
        styleDisplay.classList.remove('valid', 'invalid');
        addDressBtn.disabled = true;
        return;
    }

    const detectedStyle = detectStyleFromFilename(imageName);

    if (detectedStyle) {
        styleDisplay.value = detectedStyle;
        styleDisplay.classList.add('valid');
        styleDisplay.classList.remove('invalid');
        addDressBtn.disabled = false;
    } else {
        styleDisplay.value = '스타일을 감지할 수 없습니다';
        styleDisplay.classList.add('invalid');
        styleDisplay.classList.remove('valid');
        addDressBtn.disabled = true;
    }
}

// 파일명에서 스타일 감지 (서버의 detect_style_from_filename 함수와 동일한 로직)
function detectStyleFromFilename(filename) {
    const filenameUpper = filename.toUpperCase();

    // 1. "A"로 시작하는지 확인
    if (filenameUpper.startsWith('A')) {
        return 'A라인';
    }

    // 2. "Mini" 포함 여부 확인 (대소문자 구분 없음)
    if (filenameUpper.includes('MINI')) {
        return '미니드레스';
    }

    // 3. "B"로 시작하는지 확인
    if (filenameUpper.startsWith('B')) {
        return '벨라인';
    }

    // 4. "P"로 시작하는지 확인
    if (filenameUpper.startsWith('P')) {
        return '프린세스';
    }

    // 5. 위 조건에 해당하지 않으면 null 반환 (삽입 불가)
    return null;
}

// 드레스 목록 로드
async function loadDresses() {
    const tbody = document.getElementById('dresses-tbody');
    const totalCount = document.getElementById('total-count');

    tbody.innerHTML = '<tr><td colspan="4" class="loading">데이터를 불러오는 중...</td></tr>';

    try {
        const response = await fetch('/api/admin/dresses');
        const data = await response.json();

        if (data.success) {
            renderDresses(data.data);
            totalCount.textContent = `총 ${data.total}개`;
        } else {
            tbody.innerHTML = `<tr><td colspan="4" class="loading" style="color: #ef4444;">${data.message || '드레스 목록을 불러오는 중 오류가 발생했습니다.'}</td></tr>`;
        }
    } catch (error) {
        console.error('드레스 목록 로드 오류:', error);
        tbody.innerHTML = '<tr><td colspan="4" class="loading" style="color: #ef4444;">드레스 목록을 불러오는 중 오류가 발생했습니다.</td></tr>';
    }
}

// 드레스 목록 렌더링
function renderDresses(dresses) {
    const tbody = document.getElementById('dresses-tbody');

    if (dresses.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="loading">등록된 드레스가 없습니다.</td></tr>';
        return;
    }

    tbody.innerHTML = dresses.map(dress => {
        const imageUrl = `/images/${dress.image_name}`;
        const styleClass = getStyleClass(dress.style);

        return `
            <tr>
                <td>${dress.id}</td>
                <td class="image-name-cell">${escapeHtml(dress.image_name)}</td>
                <td><span class="style-badge ${styleClass}">${escapeHtml(dress.style)}</span></td>
                <td class="image-preview-cell">
                    <img 
                        src="${imageUrl}" 
                        alt="${escapeHtml(dress.image_name)}"
                        class="image-preview"
                        onerror="this.onerror=null; this.parentElement.innerHTML='<div class=\\'image-preview error\\'>이미지 없음</div>';"
                        loading="lazy"
                    >
                </td>
            </tr>
        `;
    }).join('');
}

// 스타일에 따른 CSS 클래스 반환
function getStyleClass(style) {
    if (style === 'A라인') return 'aline';
    if (style === '미니드레스') return 'mini';
    if (style === '벨라인') return 'bell';
    if (style === '프린세스') return 'princess';
    return '';
}

// 드레스 추가
async function handleAddDress() {
    const imageNameInput = document.getElementById('image-name-input');
    const styleDisplay = document.getElementById('style-display');
    const addDressBtn = document.getElementById('add-dress-btn');
    const messageBar = document.getElementById('add-message');

    const imageName = imageNameInput.value.trim();
    const style = styleDisplay.value;

    if (!imageName || !style) {
        showMessage('이미지명과 스타일을 모두 입력해주세요.', 'error');
        return;
    }

    // 스타일 검증 (다시 한번 확인)
    const detectedStyle = detectStyleFromFilename(imageName);
    if (!detectedStyle || detectedStyle !== style) {
        showMessage('파일명에서 스타일을 정확히 감지할 수 없습니다.', 'error');
        return;
    }

    // 버튼 비활성화
    addDressBtn.disabled = true;
    addDressBtn.textContent = '추가 중...';

    try {
        const response = await fetch('/api/admin/dresses', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_name: imageName,
                style: style
            })
        });

        const data = await response.json();

        if (data.success) {
            showMessage(data.message || '드레스가 성공적으로 추가되었습니다.', 'success');
            clearForm();
            // 목록 새로고침
            setTimeout(() => {
                loadDresses();
            }, 500);
        } else {
            showMessage(data.message || '드레스 추가 중 오류가 발생했습니다.', 'error');
        }
    } catch (error) {
        console.error('드레스 추가 오류:', error);
        showMessage('드레스 추가 중 오류가 발생했습니다.', 'error');
    } finally {
        addDressBtn.disabled = false;
        addDressBtn.textContent = '추가';
    }
}

// 폼 초기화
function clearForm() {
    const imageNameInput = document.getElementById('image-name-input');
    const styleDisplay = document.getElementById('style-display');
    const addDressBtn = document.getElementById('add-dress-btn');
    const messageBar = document.getElementById('add-message');

    imageNameInput.value = '';
    styleDisplay.value = '';
    styleDisplay.classList.remove('valid', 'invalid');
    addDressBtn.disabled = true;
    hideMessage();
}

// 메시지 표시
function showMessage(message, type) {
    const messageBar = document.getElementById('add-message');
    messageBar.textContent = message;
    messageBar.className = `message-bar ${type} show`;
}

// 메시지 숨기기
function hideMessage() {
    const messageBar = document.getElementById('add-message');
    messageBar.classList.remove('show');
}

// HTML 이스케이프
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

