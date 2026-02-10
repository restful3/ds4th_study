"""
================================================================================
LangChain AI Agent 마스터 교안
Part 9: Production - 실습 과제 3 해답
================================================================================

과제: API 통합 Agent (Structured Output)
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. 구조화된 응답 생성 (JSON, Pydantic)
2. API 스펙 준수
3. 타입 안전성 확보

학습 목표:
- Structured Output
- Pydantic 모델 사용
- API 통합 패턴

================================================================================
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from datetime import datetime
import json

# ============================================================================
# Pydantic 모델 정의
# ============================================================================

class User(BaseModel):
    """사용자 정보"""
    id: int = Field(description="사용자 ID")
    name: str = Field(description="이름")
    email: str = Field(description="이메일 주소")
    role: str = Field(description="역할 (admin, user, guest)")
    
class Task(BaseModel):
    """작업 정보"""
    id: int = Field(description="작업 ID")
    title: str = Field(description="작업 제목")
    description: str = Field(description="작업 설명")
    status: str = Field(description="상태 (pending, in_progress, completed)")
    priority: str = Field(description="우선순위 (low, medium, high)")
    assigned_to: Optional[str] = Field(default=None, description="담당자")
    due_date: Optional[str] = Field(default=None, description="마감일 (YYYY-MM-DD)")

class TaskAnalysis(BaseModel):
    """작업 분석 결과"""
    total_tasks: int = Field(description="총 작업 수")
    completed_tasks: int = Field(description="완료된 작업 수")
    pending_tasks: int = Field(description="대기 중인 작업 수")
    high_priority_tasks: int = Field(description="높은 우선순위 작업 수")
    overdue_tasks: List[str] = Field(description="지연된 작업 목록")
    recommendations: List[str] = Field(description="추천 사항")

class EmailDraft(BaseModel):
    """이메일 초안"""
    to: List[str] = Field(description="수신자 목록")
    cc: Optional[List[str]] = Field(default=None, description="참조")
    subject: str = Field(description="제목")
    body: str = Field(description="본문")
    priority: str = Field(description="우선순위 (low, normal, high)")

# ============================================================================
# Structured Output Agent
# ============================================================================

def create_structured_output_agent():
    """구조화된 출력 Agent 생성"""
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # with_structured_output으로 Pydantic 모델 바인딩
    return {
        "user_extractor": model.with_structured_output(User),
        "task_creator": model.with_structured_output(Task),
        "task_analyzer": model.with_structured_output(TaskAnalysis),
        "email_drafter": model.with_structured_output(EmailDraft),
    }

# ============================================================================
# 사용 예제
# ============================================================================

def test_user_extraction():
    """사용자 정보 추출"""
    print("=" * 70)
    print("👤 사용자 정보 추출 (Structured Output)")
    print("=" * 70)
    
    agents = create_structured_output_agent()
    user_extractor = agents["user_extractor"]
    
    text = """
    안녕하세요, 저는 홍길동입니다.
    이메일은 hong@example.com이고,
    관리자 역할을 맡고 있습니다.
    사용자 ID는 1001입니다.
    """
    
    print(f"\n입력 텍스트:\n{text}")
    
    result = user_extractor.invoke([
        HumanMessage(content=f"다음 텍스트에서 사용자 정보를 추출하세요:\n{text}")
    ])
    
    print(f"\n✅ 추출된 사용자 정보:")
    print(f"  ID: {result.id}")
    print(f"  이름: {result.name}")
    print(f"  이메일: {result.email}")
    print(f"  역할: {result.role}")
    
    # JSON 변환
    print(f"\n📄 JSON 형식:")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

def test_task_creation():
    """작업 생성"""
    print("\n" + "=" * 70)
    print("📋 작업 생성 (Structured Output)")
    print("=" * 70)
    
    agents = create_structured_output_agent()
    task_creator = agents["task_creator"]
    
    instruction = """
    새로운 기능 개발: 사용자 프로필 페이지
    - 사용자 정보 편집 기능
    - 프로필 이미지 업로드
    - 비밀번호 변경
    
    담당자: 김개발
    우선순위: 높음
    마감일: 2024-02-15
    """
    
    print(f"\n입력 지시사항:\n{instruction}")
    
    result = task_creator.invoke([
        HumanMessage(content=f"다음 지시사항을 작업으로 변환하세요:\n{instruction}")
    ])
    
    print(f"\n✅ 생성된 작업:")
    print(f"  제목: {result.title}")
    print(f"  설명: {result.description}")
    print(f"  상태: {result.status}")
    print(f"  우선순위: {result.priority}")
    print(f"  담당자: {result.assigned_to}")
    print(f"  마감일: {result.due_date}")
    
    # JSON 변환
    print(f"\n📄 JSON 형식:")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

def test_task_analysis():
    """작업 분석"""
    print("\n" + "=" * 70)
    print("📊 작업 분석 (Structured Output)")
    print("=" * 70)
    
    agents = create_structured_output_agent()
    task_analyzer = agents["task_analyzer"]
    
    tasks_data = """
    [작업 1] 로그인 버그 수정 - 완료
    [작업 2] API 문서 작성 - 진행중 (높은 우선순위)
    [작업 3] 테스트 코드 추가 - 대기중 (마감: 어제)
    [작업 4] 데이터베이스 최적화 - 대기중 (높은 우선순위, 마감: 모레)
    [작업 5] UI 개선 - 완료
    """
    
    print(f"\n작업 데이터:\n{tasks_data}")
    
    result = task_analyzer.invoke([
        HumanMessage(content=f"다음 작업 목록을 분석하세요:\n{tasks_data}")
    ])
    
    print(f"\n✅ 분석 결과:")
    print(f"  총 작업: {result.total_tasks}")
    print(f"  완료: {result.completed_tasks}")
    print(f"  대기: {result.pending_tasks}")
    print(f"  높은 우선순위: {result.high_priority_tasks}")
    print(f"  지연된 작업: {', '.join(result.overdue_tasks)}")
    
    print(f"\n💡 추천 사항:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    
    # JSON 변환
    print(f"\n📄 JSON 형식:")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

def test_email_drafting():
    """이메일 초안 작성"""
    print("\n" + "=" * 70)
    print("✉️  이메일 초안 작성 (Structured Output)")
    print("=" * 70)
    
    agents = create_structured_output_agent()
    email_drafter = agents["email_drafter"]
    
    context = """
    팀 미팅 공지
    - 일시: 2024-01-25 오후 2시
    - 장소: 회의실 A
    - 안건: Q1 계획 논의
    - 참석자: 개발팀 전체
    """
    
    print(f"\n컨텍스트:\n{context}")
    
    result = email_drafter.invoke([
        HumanMessage(content=f"다음 내용으로 팀 미팅 공지 이메일을 작성하세요:\n{context}")
    ])
    
    print(f"\n✅ 이메일 초안:")
    print(f"  받는 사람: {', '.join(result.to)}")
    if result.cc:
        print(f"  참조: {', '.join(result.cc)}")
    print(f"  제목: {result.subject}")
    print(f"  우선순위: {result.priority}")
    print(f"\n  본문:\n{result.body}")
    
    # JSON 변환
    print(f"\n📄 JSON 형식:")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

# ============================================================================
# API 통합 예제
# ============================================================================

def api_integration_example():
    """API 통합 예제"""
    print("\n" + "=" * 70)
    print("🔌 API 통합 예제")
    print("=" * 70)
    
    print("""
Structured Output을 API와 통합하는 방법:

1. FastAPI 엔드포인트:
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.post("/api/extract-user", response_model=User)
async def extract_user(text: str):
    result = user_extractor.invoke([HumanMessage(content=text)])
    return result
```

2. 타입 안전성:
- Pydantic 모델로 입출력 타입 보장
- 자동 검증 (validation)
- API 문서 자동 생성

3. 장점:
- 명확한 계약 (contract)
- 에러 감소
- 개발자 경험 향상
- 테스트 용이
    """)

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🔧 Part 9: API 통합 Agent - 실습 과제 3 해답")
    print("=" * 70)
    
    try:
        test_user_extraction()
        test_task_creation()
        test_task_analysis()
        test_email_drafting()
        api_integration_example()
        
        print("\n💡 학습 포인트:")
        print("  1. Structured Output (Pydantic)")
        print("  2. 타입 안전성 확보")
        print("  3. JSON 스키마 준수")
        print("  4. API 통합 패턴")
    except Exception as e:
        print(f"⚠️ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
