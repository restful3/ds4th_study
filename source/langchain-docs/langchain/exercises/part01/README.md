# 📚 LangChain Part 1 퀴즈

Part 1(AI Agent의 이해)을 테스트할 수 있는 간단한 퀴즈입니다.

## ✨ 특징

- 🌐 **설치 불필요**: 브라우저만 있으면 OK
- 📝 **10개 객관식 문제**: 한 번에 쭉 풀기
- 📊 **즉시 채점**: 정답, 오답, 해설 한 번에 확인
- 💾 **간단함**: 복잡한 설정 없음
- 📱 **반응형**: PC, 태블릿, 모바일 모두 지원

## 🚀 사용 방법

### 1. 파일 열기 (가장 간단)

```bash
# Mac
open index.html

# Windows
start index.html

# Linux
xdg-open index.html
```

**또는** 파일 탐색기에서 `index.html`을 더블클릭!

### 2. 퀴즈 풀기

1. **"퀴즈 시작하기"** 버튼 클릭
2. 10개 문제 쭉 풀기 (스크롤하며 답 선택)
3. **"제출하기"** 버튼 클릭
4. 결과 확인 (정답률 + 문제별 해설)

## 📊 퀴즈 내용

총 **10개의 객관식 문제**:

| 주제 | 문제 수 |
|-----|--------|
| LangChain 소개 | 2 |
| LangChain 철학 | 1 |
| AI Agent 개념 | 2 |
| LangChain vs LangGraph | 2 |
| 환경 설정 | 2 |

**범위**: LangChain 기본 개념, Agent 개념, 환경 설정 등

## 📁 파일 구조

```
learning_tools/
├── index.html       # 메인 퀴즈 페이지 ⭐
├── style.css        # 스타일
├── script.js        # 퀴즈 로직 (데이터 포함)
└── README.md        # 이 파일
```

## 🎯 평가 기준

- **80% 이상**: 훌륭! Part 2로 진행 가능
- **60-79%**: 잘했어요! 틀린 부분 복습 권장
- **60% 미만**: 교안 복습 후 다시 도전

## 💡 학습 팁

### 1. 교안 먼저 읽기
```
docs/part01_introduction.md
```

### 2. 예제 코드 실행
```bash
cd src/part01_introduction
python 01_hello_langchain.py
python 02_environment_check.py
```

### 3. 퀴즈로 복습
- 여러 번 풀어보기
- 틀린 문제 해설 꼭 읽기

## 🌐 GitHub Pages (선택사항)

GitHub Pages로 배포하면 URL로 접속 가능:

1. GitHub 저장소 → **Settings** → **Pages**
2. Source: **main** 브랜치 선택
3. 저장 후 URL 확인:
   ```
   https://[username].github.io/langchain/learning_tools/
   ```

## 🔧 문제 해결

### 퀴즈가 로드되지 않음

**해결**: 최신 브라우저 사용 (Chrome, Firefox, Safari, Edge)

### 모바일에서 레이아웃 깨짐

**해결**: 화면을 세로로 보거나 최신 브라우저 사용

## 📄 라이선스

이 퀴즈는 LangChain AI Agent 교안의 일부입니다.

---

**🎓 즐거운 학습 되세요!**

문의사항은 이슈로 등록해주세요.
