# Part 9: 프로덕션 - 프론트엔드 예제

> React + FastAPI를 사용한 실시간 스트리밍 Agent 채팅 애플리케이션

---

## 📋 개요

이 프론트엔드 예제는 LangChain Agent를 React로 구현한 실시간 채팅 인터페이스입니다.

### 주요 기능
- ✅ 실시간 토큰 스트리밍
- ✅ 도구 호출 시각화
- ✅ 마크다운 렌더링
- ✅ 에러 처리
- ✅ 스트리밍 중단 기능

---

## 🚀 빠른 시작

### 1. 백엔드 서버 실행

```bash
# 의존성 설치 (처음 한 번만)
pip install fastapi uvicorn langchain langchain-openai python-dotenv

# .env 파일 설정
echo "OPENAI_API_KEY=your-key-here" > .env

# 서버 실행
python backend_server.py
```

서버가 실행되면: http://localhost:8000

### 2. 프론트엔드 실행

```bash
# 의존성 설치 (처음 한 번만)
npm install

# 개발 서버 실행
npm run dev
```

브라우저에서: http://localhost:5173

---

## 📁 파일 구조

```
frontend/
├── package.json           # Node.js 의존성
├── react_stream.tsx       # React 메인 컴포넌트
├── backend_server.py      # FastAPI 백엔드
└── README.md             # 이 파일
```

---

## 🎨 주요 컴포넌트

### AgentChat (메인)

```tsx
export default function AgentChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  // SSE 스트리밍 처리
  const streamAgentResponse = async (userMessage: string) => {
    // ...
  };

  return (
    <div className="flex flex-col h-screen">
      {/* 헤더 */}
      {/* 메시지 영역 */}
      {/* 입력 영역 */}
    </div>
  );
}
```

### MessageBubble (메시지 버블)

```tsx
function MessageBubble({ message }: { message: Message }) {
  return (
    <div className="flex gap-3">
      {/* 아바타 */}
      {/* 메시지 내용 (마크다운 렌더링) */}
    </div>
  );
}
```

---

## 🔧 API 엔드포인트

### POST /stream

스트리밍 채팅 요청

**요청:**
```json
{
  "message": "서울 날씨 알려줘",
  "stream": true
}
```

**응답 (SSE):**
```
data: {"type": "token", "content": "서울"}

data: {"type": "token", "content": "의"}

data: {"type": "tool_call", "tool": "get_weather"}

data: [DONE]
```

### POST /chat

일반 채팅 요청 (비스트리밍)

**요청:**
```json
{
  "message": "안녕하세요"
}
```

**응답:**
```json
{
  "response": "안녕하세요! 무엇을 도와드릴까요?"
}
```

---

## 💡 핵심 구현

### 1. Server-Sent Events (SSE)

```tsx
const response = await fetch('http://localhost:8000/stream', {
  method: 'POST',
  body: JSON.stringify({ message: userMessage }),
});

const reader = response.body?.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  // 청크 처리...
}
```

### 2. 상태 관리

```tsx
const [messages, setMessages] = useState<Message[]>([]);
const [currentStreamContent, setCurrentStreamContent] = useState('');
const [isStreaming, setIsStreaming] = useState(false);
```

### 3. 스트리밍 중단

```tsx
const abortControllerRef = useRef<AbortController | null>(null);

const handleStop = () => {
  if (abortControllerRef.current) {
    abortControllerRef.current.abort();
  }
};
```

---

## 🎨 스타일링

이 예제는 Tailwind CSS를 사용합니다.

### 설치 (선택사항)

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### tailwind.config.js

```js
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

---

## 🐛 문제 해결

### CORS 오류

```
Access to fetch at 'http://localhost:8000/stream' from origin
'http://localhost:5173' has been blocked by CORS policy
```

**해결:**
- `backend_server.py`의 CORS 설정 확인
- `allow_origins`에 프론트엔드 주소 추가

### 스트리밍이 작동하지 않음

**확인 사항:**
1. 백엔드 서버가 실행 중인지
2. OPENAI_API_KEY가 설정되어 있는지
3. 브라우저 콘솔에 오류가 있는지

---

## 🚀 프로덕션 배포

### 프론트엔드 빌드

```bash
npm run build
```

빌드된 파일: `dist/` 폴더

### 백엔드 배포

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "backend_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🎓 학습 포인트

### 배울 수 있는 것

1. **React Hooks**
   - useState: 상태 관리
   - useRef: DOM 참조, AbortController
   - useEffect: 사이드 이펙트

2. **비동기 처리**
   - fetch API
   - ReadableStream
   - async/await

3. **SSE (Server-Sent Events)**
   - 단방향 실시간 통신
   - EventSource 대안

4. **UX/UI**
   - 실시간 피드백
   - 로딩 상태 표시
   - 에러 핸들링

---

## 🔗 참고 자료

- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [React 문서](https://react.dev/)
- [LangChain Streaming](https://python.langchain.com/docs/modules/model_io/streaming/)
- [Server-Sent Events (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

---

## 💡 추가 개선 아이디어

### 기본 개선
- [ ] 대화 이력 저장 (localStorage)
- [ ] 다크 모드
- [ ] 음성 입력/출력
- [ ] 파일 업로드

### 고급 개선
- [ ] WebSocket으로 양방향 통신
- [ ] 사용자 인증 (JWT)
- [ ] 멀티 세션 관리
- [ ] Agent 설정 UI
- [ ] 대화 내보내기

---

**Good Luck! 🚀**

*Part 10 (배포와 관측성)으로 진행하세요!*
