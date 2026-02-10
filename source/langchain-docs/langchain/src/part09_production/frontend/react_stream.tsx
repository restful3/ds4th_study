/**
 * ============================================================================
 * LangChain AI Agent 마스터 교안
 * Part 9: 프로덕션 준비 - React 스트리밍 예제
 * ============================================================================
 *
 * 파일명: react_stream.tsx
 * 난이도: ⭐⭐⭐⭐☆ (고급)
 * 예상 시간: 1시간
 *
 * 📚 학습 목표:
 *   - React에서 Agent 스트리밍 구현
 *   - SSE (Server-Sent Events) 사용
 *   - 실시간 UI 업데이트
 *   - 에러 처리 및 로딩 상태
 *
 * 📖 공식 문서:
 *   • Streaming: /official/11-streaming-overview.md
 *   • Frontend: /official/12-streaming-frontend.md
 *
 * 🚀 실행 방법:
 *   1. 백엔드 서버 실행: python backend_server.py
 *   2. 프론트엔드 실행: npm run dev
 *
 * ============================================================================
 */

import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, AlertCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

// ============================================================================
// 타입 정의
// ============================================================================

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface StreamEvent {
  type: 'token' | 'tool_call' | 'error' | 'end';
  content?: string;
  tool?: string;
  error?: string;
}

// ============================================================================
// 메인 컴포넌트
// ============================================================================

export default function AgentChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStreamContent, setCurrentStreamContent] = useState('');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // 메시지 추가 시 자동 스크롤
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentStreamContent]);

  // ============================================================================
  // 스트리밍 처리
  // ============================================================================

  const streamAgentResponse = async (userMessage: string) => {
    setIsStreaming(true);
    setError(null);
    setCurrentStreamContent('');

    // 사용자 메시지 추가
    const newUserMessage: Message = {
      role: 'user',
      content: userMessage,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, newUserMessage]);

    // AbortController로 취소 가능하게
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch('http://localhost:8000/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          stream: true,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();

          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);

              if (data === '[DONE]') {
                // 스트리밍 완료
                const assistantMessage: Message = {
                  role: 'assistant',
                  content: accumulatedContent,
                  timestamp: new Date(),
                };
                setMessages(prev => [...prev, assistantMessage]);
                setCurrentStreamContent('');
                break;
              }

              try {
                const event: StreamEvent = JSON.parse(data);

                if (event.type === 'token' && event.content) {
                  accumulatedContent += event.content;
                  setCurrentStreamContent(accumulatedContent);
                } else if (event.type === 'tool_call' && event.tool) {
                  // 도구 호출 표시
                  accumulatedContent += `\n\n🔧 도구 사용: ${event.tool}\n\n`;
                  setCurrentStreamContent(accumulatedContent);
                } else if (event.type === 'error') {
                  throw new Error(event.error || '알 수 없는 오류');
                }
              } catch (parseError) {
                console.error('파싱 오류:', parseError);
              }
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setError('요청이 취소되었습니다.');
      } else {
        setError(err.message || '오류가 발생했습니다.');
      }
      setCurrentStreamContent('');
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  };

  // ============================================================================
  // 이벤트 핸들러
  // ============================================================================

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim() || isStreaming) return;

    const userMessage = input.trim();
    setInput('');

    await streamAgentResponse(userMessage);
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const handleClear = () => {
    setMessages([]);
    setError(null);
    setCurrentStreamContent('');
  };

  // ============================================================================
  // 렌더링
  // ============================================================================

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* 헤더 */}
      <header className="bg-white shadow-sm p-4 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Bot className="w-6 h-6 text-blue-600" />
          <h1 className="text-xl font-semibold">LangChain Agent</h1>
        </div>
        <button
          onClick={handleClear}
          className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900"
        >
          대화 초기화
        </button>
      </header>

      {/* 메시지 영역 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && !currentStreamContent && (
          <div className="text-center text-gray-500 mt-8">
            <Bot className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <p>안녕하세요! 무엇을 도와드릴까요?</p>
            <p className="text-sm mt-2">예: "서울 날씨 알려줘", "Python 코드 작성해줘"</p>
          </div>
        )}

        {messages.map((message, index) => (
          <MessageBubble key={index} message={message} />
        ))}

        {/* 스트리밍 중인 메시지 */}
        {currentStreamContent && (
          <div className="flex gap-3">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
            </div>
            <div className="flex-1 bg-white rounded-lg p-4 shadow-sm">
              <ReactMarkdown className="prose prose-sm max-w-none">
                {currentStreamContent}
              </ReactMarkdown>
              <Loader2 className="w-4 h-4 animate-spin text-blue-500 mt-2" />
            </div>
          </div>
        )}

        {/* 에러 표시 */}
        {error && (
          <div className="flex items-center gap-2 p-4 bg-red-50 text-red-700 rounded-lg">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* 입력 영역 */}
      <form onSubmit={handleSubmit} className="p-4 bg-white border-t">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="메시지를 입력하세요..."
            className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isStreaming}
          />

          {isStreaming ? (
            <button
              type="button"
              onClick={handleStop}
              className="px-6 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition"
            >
              중지
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Send className="w-4 h-4" />
              전송
            </button>
          )}
        </div>
      </form>
    </div>
  );
}

// ============================================================================
// 메시지 버블 컴포넌트
// ============================================================================

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* 아바타 */}
      <div className="flex-shrink-0">
        <div
          className={`w-8 h-8 rounded-full flex items-center justify-center ${
            isUser ? 'bg-gray-700' : 'bg-blue-500'
          }`}
        >
          {isUser ? (
            <User className="w-5 h-5 text-white" />
          ) : (
            <Bot className="w-5 h-5 text-white" />
          )}
        </div>
      </div>

      {/* 메시지 내용 */}
      <div
        className={`flex-1 max-w-[70%] rounded-lg p-4 shadow-sm ${
          isUser ? 'bg-gray-700 text-white' : 'bg-white'
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <ReactMarkdown className="prose prose-sm max-w-none">
            {message.content}
          </ReactMarkdown>
        )}

        <span className="text-xs opacity-70 mt-2 block">
          {message.timestamp.toLocaleTimeString('ko-KR', {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>
    </div>
  );
}

// ============================================================================
// 📚 학습 포인트
// ============================================================================
//
// 1. Server-Sent Events (SSE):
//    - 단방향 서버 → 클라이언트 스트리밍
//    - ReadableStream으로 데이터 읽기
//    - data: 프리픽스로 이벤트 파싱
//
// 2. React State 관리:
//    - messages: 완료된 메시지 목록
//    - currentStreamContent: 스트리밍 중인 내용
//    - isStreaming: 로딩 상태
//
// 3. AbortController:
//    - 스트리밍 중단 기능
//    - 메모리 누수 방지
//
// 4. UX 개선:
//    - 실시간 토큰 표시
//    - 도구 호출 시각화
//    - 에러 핸들링
//    - 자동 스크롤
//
// ============================================================================
// 🎓 추가 개선 아이디어
// ============================================================================
//
// - WebSocket으로 양방향 통신
// - 대화 이력 저장 (localStorage)
// - 음성 입력 (Web Speech API)
// - 파일 업로드 (이미지, 문서)
// - 마크다운 렌더링 개선
// - 다크 모드
//
// ============================================================================
