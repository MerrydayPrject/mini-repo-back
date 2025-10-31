# 데이터베이스 설정 가이드

이 문서는 프로젝트에 필요한 데이터베이스와 테이블을 설정하는 방법을 설명합니다.

## 목차

1. [데이터베이스 요구사항](#데이터베이스-요구사항)
2. [MySQL 설치 및 데이터베이스 생성](#mysql-설치-및-데이터베이스-생성)
3. [환경 변수 설정](#환경-변수-설정)
4. [자동 테이블 생성](#자동-테이블-생성)
5. [수동 테이블 생성](#수동-테이블-생성)
6. [테이블 구조](#테이블-구조)

---

## 데이터베이스 요구사항

- **데이터베이스 엔진**: MySQL 5.7 이상 (또는 MariaDB 10.2 이상)
- **필수 테이블**: 
  - `composition_logs` - 이미지 합성 로그 저장
  - `dress_info` - 드레스 정보 저장

---

## MySQL 설치 및 데이터베이스 생성

### 1. MySQL 설치

#### Windows
1. [MySQL 공식 사이트](https://dev.mysql.com/downloads/mysql/)에서 MySQL Community Server 다운로드
2. 설치 마법사를 따라 설치 완료
3. MySQL 서비스 시작

#### macOS (Homebrew)
```bash
brew install mysql
brew services start mysql
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install mysql-server
sudo systemctl start mysql
sudo systemctl enable mysql
```

### 2. 데이터베이스 생성

MySQL에 접속하여 데이터베이스를 생성합니다:

```bash
mysql -u root -p
```

MySQL 프롬프트에서 다음 명령 실행:

```sql
-- 데이터베이스 생성
CREATE DATABASE IF NOT EXISTS marryday 
    CHARACTER SET utf8mb4 
    COLLATE utf8mb4_unicode_ci;

-- 사용자 생성 및 권한 부여 (선택사항)
CREATE USER IF NOT EXISTS 'devuser'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON marryday.* TO 'devuser'@'localhost';
FLUSH PRIVILEGES;

-- 확인
SHOW DATABASES;
USE marryday;
```

---

## 환경 변수 설정

프로젝트 루트 디렉토리(`mini-repo-back/`)에 `.env` 파일을 생성하고 다음과 같이 설정합니다:

```env
# MySQL 데이터베이스 설정
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=devuser
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=marryday
```

### 환경 변수 설명

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `MYSQL_HOST` | MySQL 서버 호스트 주소 | `localhost` |
| `MYSQL_PORT` | MySQL 서버 포트 번호 | `3306` |
| `MYSQL_USER` | MySQL 사용자 이름 | `devuser` |
| `MYSQL_PASSWORD` | MySQL 비밀번호 | (빈 문자열) |
| `MYSQL_DATABASE` | 사용할 데이터베이스 이름 | `marryday` |

**보안 주의사항**: 
- `.env` 파일은 절대 Git에 커밋하지 마세요
- `.gitignore`에 `.env` 파일이 포함되어 있는지 확인하세요

---

## 자동 테이블 생성

애플리케이션을 실행하면 자동으로 필요한 테이블이 생성됩니다:

```bash
cd mini-repo-back
python main.py
```

서버 시작 시 `init_database()` 함수가 호출되어 다음 메시지가 출력됩니다:

```
DB 테이블 생성 완료: composition_logs
DB 테이블 생성 완료: dress_info
```

### 자동 생성이 작동하지 않는 경우

1. 데이터베이스 연결 확인:
   - `.env` 파일의 설정이 올바른지 확인
   - MySQL 서비스가 실행 중인지 확인
   - 사용자 권한이 올바른지 확인

2. 수동 생성 방법 사용 (아래 참조)

---

## 수동 테이블 생성

자동 생성이 작동하지 않는 경우, SQL 스크립트로 직접 테이블을 생성할 수 있습니다.

### MySQL에 접속

```bash
mysql -u devuser -p marryday
```

또는 root로 접속:

```bash
mysql -u root -p
USE marryday;
```

### 테이블 생성 SQL 실행

#### 1. composition_logs 테이블

```sql
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
```

#### 2. dress_info 테이블

```sql
CREATE TABLE IF NOT EXISTS dress_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,
    style VARCHAR(100) NOT NULL,
    INDEX idx_image_name (image_name),
    INDEX idx_style (style)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 테이블 생성 확인

```sql
-- 테이블 목록 확인
SHOW TABLES;

-- 테이블 구조 확인
DESCRIBE composition_logs;
DESCRIBE dress_info;

-- 테이블 데이터 확인
SELECT * FROM composition_logs LIMIT 5;
SELECT * FROM dress_info LIMIT 5;
```

---

## 테이블 구조

### composition_logs 테이블

이미지 합성 작업의 로그를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `id` | INT | 자동 증가 기본 키 |
| `created_at` | DATETIME | 로그 생성 일시 (자동 설정) |
| `model_name` | VARCHAR(100) | 사용한 모델 이름 |
| `api_name` | VARCHAR(50) | 사용한 API 이름 |
| `prompt` | TEXT | 사용한 프롬프트 |
| `person_image_path` | VARCHAR(255) | 사람 이미지 경로 |
| `dress_image_path` | VARCHAR(255) | 드레스 이미지 경로 |
| `result_image_path` | VARCHAR(255) | 결과 이미지 경로 (NULL 가능) |
| `success` | BOOLEAN | 작업 성공 여부 |
| `processing_time` | FLOAT | 처리 시간 (초) |
| `error_message` | TEXT | 에러 메시지 (NULL 가능) |

**인덱스**:
- `idx_created_at`: 생성 일시로 빠른 검색
- `idx_success`: 성공 여부로 필터링

### dress_info 테이블

드레스 정보를 저장하는 테이블입니다.

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `id` | INT | 자동 증가 기본 키 |
| `image_name` | VARCHAR(255) | 이미지 파일명 (예: Adress1.png) |
| `style` | VARCHAR(100) | 드레스 스타일 (A라인, 미니드레스, 벨라인, 프린세스) |

**인덱스**:
- `idx_image_name`: 이미지명으로 빠른 검색 및 중복 체크
- `idx_style`: 스타일로 필터링

**스타일 규칙**:
- 이미지명이 **A**로 시작 → `A라인`
- 이미지명에 **Mini** 포함 → `미니드레스`
- 이미지명이 **B**로 시작 → `벨라인`
- 이미지명이 **P**로 시작 → `프린세스`
- 그 외 → 삽입 불가

---

## 테스트

### 데이터베이스 연결 테스트

Python 스크립트로 연결 테스트:

```python
import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

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
    print("✅ 데이터베이스 연결 성공!")
    
    with connection.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"테이블 개수: {len(tables)}")
        for table in tables:
            print(f"  - {list(table.values())[0]}")
    
    connection.close()
except Exception as e:
    print(f"❌ 데이터베이스 연결 실패: {e}")
```

### 샘플 데이터 삽입 테스트

```sql
-- dress_info 테이블에 샘플 데이터 삽입
INSERT INTO dress_info (image_name, style) VALUES
    ('Adress1.png', 'A라인'),
    ('Minidress1.png', '미니드레스'),
    ('Bdress1.png', '벨라인'),
    ('Pdress1.png', '프린세스');

-- 데이터 확인
SELECT * FROM dress_info;
```

---

## 문제 해결

### 1. "Access denied" 오류

- 사용자 이름과 비밀번호가 올바른지 확인
- 사용자에게 데이터베이스 접근 권한이 있는지 확인

```sql
-- 권한 확인
SHOW GRANTS FOR 'devuser'@'localhost';
```

### 2. "Table doesn't exist" 오류

- 데이터베이스가 올바르게 선택되었는지 확인
- 테이블이 생성되었는지 확인

```sql
USE marryday;
SHOW TABLES;
```

### 3. "Can't connect to MySQL server" 오류

- MySQL 서비스가 실행 중인지 확인

```bash
# Windows
net start MySQL

# macOS/Linux
sudo systemctl status mysql
# 또는
brew services list | grep mysql
```

### 4. 한글 인코딩 문제

- 데이터베이스와 테이블이 `utf8mb4` 문자셋을 사용하는지 확인

```sql
-- 데이터베이스 문자셋 확인
SHOW CREATE DATABASE marryday;

-- 테이블 문자셋 확인
SHOW CREATE TABLE dress_info;
```

---

## 참고 자료

- [MySQL 공식 문서](https://dev.mysql.com/doc/)
- [PyMySQL 문서](https://pymysql.readthedocs.io/)
- [FastAPI 데이터베이스 가이드](https://fastapi.tiangolo.com/tutorial/sql-databases/)

---

## 요약

1. **MySQL 설치** 및 데이터베이스 `marryday` 생성
2. **`.env` 파일** 생성 및 데이터베이스 연결 정보 설정
3. **애플리케이션 실행** 시 자동으로 테이블 생성됨
   - 또는 SQL 스크립트로 수동 생성 가능
4. **테스트**로 연결 및 데이터 확인

서버를 실행하면 자동으로 테이블이 생성되므로, 대부분의 경우 데이터베이스만 미리 생성하면 됩니다! 🚀

