// 전역 변수
let currentPage = 1;
const itemsPerPage = 20;

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    loadLogs(currentPage);
});

// 통계 로드
async function loadStats() {
    try {
        const response = await fetch('/api/admin/stats');
        const data = await response.json();
        
        if (data.success) {
            const stats = data.data;
            document.getElementById('stat-total').textContent = stats.total;
            document.getElementById('stat-success').textContent = stats.success;
            document.getElementById('stat-failed').textContent = stats.failed;
            document.getElementById('stat-success-rate').textContent = stats.success_rate + '%';
            document.getElementById('stat-avg-time').textContent = stats.average_processing_time + '초';
            document.getElementById('stat-today').textContent = stats.today;
        }
    } catch (error) {
        console.error('통계 로드 오류:', error);
    }
}

// 로그 목록 로드
async function loadLogs(page) {
    try {
        const response = await fetch(`/api/admin/logs?page=${page}&limit=${itemsPerPage}`);
        const data = await response.json();
        
        if (data.success) {
            renderLogs(data.data);
            renderPagination(data.pagination);
            currentPage = page;
        } else {
            showError('로그를 불러오는 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('로그 로드 오류:', error);
        document.getElementById('logs-tbody').innerHTML = 
            '<tr><td colspan="7" class="loading">로그를 불러오는 중 오류가 발생했습니다.</td></tr>';
    }
}

// 로그 테이블 렌더링
function renderLogs(logs) {
    const tbody = document.getElementById('logs-tbody');
    
    if (logs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="loading">로그가 없습니다.</td></tr>';
        return;
    }
    
    tbody.innerHTML = logs.map(log => `
        <tr>
            <td>${log.id}</td>
            <td>${formatDateTime(log.created_at)}</td>
            <td>${log.model_name}</td>
            <td>${log.api_name}</td>
            <td>${renderStatusBadge(log.success)}</td>
            <td>${log.processing_time ? log.processing_time.toFixed(2) + '초' : '-'}</td>
            <td>
                <button class="btn-detail" onclick="showDetail(${log.id})">상세</button>
            </td>
        </tr>
    `).join('');
}

// 상태 배지 렌더링
function renderStatusBadge(success) {
    if (success) {
        return '<span class="status-badge status-success">성공</span>';
    } else {
        return '<span class="status-badge status-failed">실패</span>';
    }
}

// 페이지네이션 렌더링
function renderPagination(pagination) {
    const paginationDiv = document.getElementById('pagination');
    
    if (pagination.total_pages === 0) {
        paginationDiv.innerHTML = '';
        return;
    }
    
    let html = `
        <button onclick="loadLogs(1)" ${pagination.page === 1 ? 'disabled' : ''}>처음</button>
    `;
    
    // 이전 페이지
    if (pagination.page > 1) {
        html += `<button onclick="loadLogs(${pagination.page - 1})">이전</button>`;
    }
    
    // 페이지 번호들
    const startPage = Math.max(1, pagination.page - 2);
    const endPage = Math.min(pagination.total_pages, pagination.page + 2);
    
    if (startPage > 1) {
        html += '<button disabled>...</button>';
    }
    
    for (let i = startPage; i <= endPage; i++) {
        html += `<button onclick="loadLogs(${i})" class="${i === pagination.page ? 'active' : ''}">${i}</button>`;
    }
    
    if (endPage < pagination.total_pages) {
        html += '<button disabled>...</button>';
    }
    
    // 다음 페이지
    if (pagination.page < pagination.total_pages) {
        html += `<button onclick="loadLogs(${pagination.page + 1})">다음</button>`;
    }
    
    html += `
        <button onclick="loadLogs(${pagination.total_pages})" ${pagination.page === pagination.total_pages ? 'disabled' : ''}>마지막</button>
    `;
    
    html += `<span class="pagination-info">총 ${pagination.total}개 항목 (${pagination.page}/${pagination.total_pages} 페이지)</span>`;
    
    paginationDiv.innerHTML = html;
}

// 로그 상세 보기
async function showDetail(logId) {
    try {
        const response = await fetch(`/api/admin/logs/${logId}`);
        const data = await response.json();
        
        if (data.success) {
            renderDetailModal(data.data);
            openModal();
        } else {
            alert('로그를 불러오는 중 오류가 발생했습니다.');
        }
    } catch (error) {
        console.error('상세 로그 로드 오류:', error);
        alert('로그를 불러오는 중 오류가 발생했습니다.');
    }
}

// 상세 모달 렌더링
function renderDetailModal(log) {
    const modalBody = document.getElementById('modal-body');
    
    const promptHtml = log.prompt ? `
        <div class="detail-item">
            <div class="detail-label">사용한 프롬프트</div>
            <div class="detail-prompt">${escapeHtml(log.prompt)}</div>
        </div>
    ` : '';
    
    const resultImageHtml = log.result_image_path ? `
        <div class="detail-item">
            <div class="detail-label">결과 이미지</div>
            <div class="image-preview">
                <img src="/${log.result_image_path}" alt="Result" loading="lazy">
            </div>
        </div>
    ` : '';
    
    const errorMessageHtml = log.error_message ? `
        <div class="detail-item">
            <div class="detail-label">에러 메시지</div>
            <div class="detail-value" style="color: #ef4444;">${escapeHtml(log.error_message)}</div>
        </div>
    ` : '';
    
    modalBody.innerHTML = `
        <div class="detail-grid">
            <div class="detail-item">
                <div class="detail-label">ID</div>
                <div class="detail-value">${log.id}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">매칭 일시</div>
                <div class="detail-value">${formatDateTime(log.created_at)}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">모델명</div>
                <div class="detail-value">${escapeHtml(log.model_name)}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">사용한 API</div>
                <div class="detail-value">${escapeHtml(log.api_name)}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">상태</div>
                <div class="detail-value">${renderStatusBadge(log.success)}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">처리 시간</div>
                <div class="detail-value">${log.processing_time ? log.processing_time.toFixed(3) + '초' : '-'}</div>
            </div>
            ${promptHtml}
            <div class="detail-item">
                <div class="detail-label">입력 이미지</div>
                <div class="image-preview">
                    <img src="/${log.person_image_path}" alt="Person" loading="lazy">
                    <img src="/${log.dress_image_path}" alt="Dress" loading="lazy">
                </div>
            </div>
            ${resultImageHtml}
            ${errorMessageHtml}
        </div>
    `;
}

// 모달 열기
function openModal() {
    document.getElementById('detail-modal').classList.add('show');
}

// 모달 닫기
function closeModal() {
    document.getElementById('detail-modal').classList.remove('show');
}

// 모달 외부 클릭 시 닫기
document.addEventListener('click', (e) => {
    const modal = document.getElementById('detail-modal');
    if (e.target === modal) {
        closeModal();
    }
});

// ESC 키로 모달 닫기
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal();
    }
});

// 유틸리티 함수들
function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString('ko-KR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showError(message) {
    // 간단한 에러 표시 (필요시 토스트 메시지 등으로 변경 가능)
    alert(message);
}


