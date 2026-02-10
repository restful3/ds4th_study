"""
================================================================================
LangChain AI Agent 마스터 교안
Part 7: Multi-Agent Systems
================================================================================

파일명: 05_skills_pattern.py
난이도: ⭐⭐⭐⭐ (고급)
예상 시간: 30분

📚 학습 목표:
  - Skills 패턴의 개념 이해
  - 동적 스킬 로딩 구현
  - On-demand tool activation
  - 스킬 라이브러리 구축
  - 실전: 플러그인 시스템

📖 공식 문서:
  • Skills: /official/25-skills.md

📄 교안 문서:
  • Part 7 Skills: /docs/part07_multi_agent.md (Section 4)

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🚀 실행 방법:
  python 05_skills_pattern.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import Callable
from functools import lru_cache

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 스킬 레지스트리
SKILLS_REGISTRY: dict[str, Callable] = {}

# ============================================================================
# 예제 1: Skills 패턴 개념
# ============================================================================

def example_1_skills_concept():
    """Skills 패턴의 기본 개념"""
    print("=" * 70)
    print("📌 예제 1: Skills 패턴 개념")
    print("=" * 70)

    print("""
💡 Skills 패턴이란?
   - 재사용 가능한 Agent 컴포넌트
   - 필요할 때만 동적으로 로딩
   - 플러그인처럼 추가/제거 가능
   - 여러 프로젝트에서 재사용

🔄 기존 방식 vs Skills 패턴:

   기존 (모든 도구 항상 로드):
   Agent + [Tool1, Tool2, ..., Tool50]
   → 느림, 비효율적

   Skills (필요 시에만 로드):
   Agent + load_skill("translation") when needed
   → 빠름, 효율적
    """)

    # 스킬 레지스트리 데모
    skills = {
        "translation": "번역 스킬",
        "summarization": "요약 스킬",
        "coding": "코딩 스킬",
        "analysis": "분석 스킬"
    }

    print("\n📦 사용 가능한 스킬:")
    print("-" * 70)
    for name, desc in skills.items():
        print(f"  • {name}: {desc}")

    print("\n💡 장점:")
    print("-" * 70)
    print("  1. 메모리 효율: 필요한 스킬만 로드")
    print("  2. 빠른 시작: 초기 로딩 시간 단축")
    print("  3. 유연성: 런타임에 스킬 추가/제거")
    print("  4. 재사용성: 다른 프로젝트에서도 사용")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 2: 동적 스킬 로딩
# ============================================================================

def example_2_dynamic_loading():
    """스킬을 동적으로 로딩하는 방법"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 동적 스킬 로딩")
    print("=" * 70)

    print("""
💡 스킬 레지스트리 패턴:
   - 스킬 생성 함수를 레지스트리에 등록
   - 필요할 때 레지스트리에서 스킬 로드
   - 데코레이터로 간편하게 등록
    """)

    # 스킬 레지스트리
    registry = {}

    def register_skill(name: str):
        """스킬 등록 데코레이터"""
        def decorator(func):
            registry[name] = func
            return func
        return decorator

    # 스킬 정의
    @register_skill("translator")
    def create_translator():
        """번역 스킬 생성"""
        @tool
        def translate(text: str, target: str = "영어") -> str:
            f"""텍스트를 {target}로 번역"""
            prompt = f"다음을 {target}로 번역: {text}"
            response = llm.invoke(prompt)
            return response.content
        return translate

    @register_skill("summarizer")
    def create_summarizer():
        """요약 스킬 생성"""
        @tool
        def summarize(text: str) -> str:
            """텍스트를 요약"""
            prompt = f"다음을 3문장으로 요약: {text}"
            response = llm.invoke(prompt)
            return response.content
        return summarize

    # 동적 로딩
    def load_skill(skill_name: str):
        """스킬 동적 로딩"""
        if skill_name not in registry:
            raise ValueError(f"스킬 '{skill_name}'을 찾을 수 없습니다")
        return registry[skill_name]()

    print("\n📦 등록된 스킬:")
    print("-" * 70)
    for name in registry.keys():
        print(f"  • {name}")

    print("\n🧪 동적 로딩 테스트:")
    print("-" * 70)

    skill_name = input("로딩할 스킬 (translator/summarizer): ").strip() or "translator"

    try:
        print(f"\n'{skill_name}' 스킬 로딩 중...")
        skill = load_skill(skill_name)
        print(f"✅ '{skill.name}' 로딩 완료!")

        if skill_name == "translator":
            result = skill.invoke({"text": "안녕하세요", "target": "영어"})
            print(f"\n결과: {result}")
        else:
            text = "인공지능은 빠르게 발전하고 있습니다. 많은 기업이 AI를 도입하고 있습니다. 미래는 밝습니다."
            result = skill.invoke({"text": text})
            print(f"\n결과: {result}")

    except ValueError as e:
        print(f"❌ {e}")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 3: On-demand Tool Activation
# ============================================================================

def example_3_ondemand_activation():
    """필요할 때만 도구를 활성화"""
    print("\n" + "=" * 70)
    print("📌 예제 3: On-demand Tool Activation")
    print("=" * 70)

    print("""
💡 On-demand 활성화:
   - Agent가 필요 판단 시 스킬 로드
   - 사용하지 않는 스킬은 메모리에서 제외
   - 성능과 리소스 최적화
    """)

    # 스킬 생성 함수들
    def create_math_skill():
        """수학 스킬"""
        print("  [로딩] 수학 스킬")
        @tool
        def calculate(expression: str) -> str:
            """수식 계산"""
            try:
                result = eval(expression)
                return f"결과: {result}"
            except:
                return "계산 오류"
        return calculate

    def create_text_skill():
        """텍스트 스킬"""
        print("  [로딩] 텍스트 스킬")
        @tool
        def analyze_text(text: str) -> str:
            """텍스트 분석"""
            return f"길이: {len(text)}, 단어: {len(text.split())}"
        return analyze_text

    def create_data_skill():
        """데이터 스킬"""
        print("  [로딩] 데이터 스킬")
        @tool
        def process_data(data: str) -> str:
            """데이터 처리"""
            return f"데이터 처리 완료: {data[:20]}..."
        return process_data

    # 스킬 매니저
    class SkillManager:
        def __init__(self):
            self.loaded_skills = {}
            self.skill_creators = {
                "math": create_math_skill,
                "text": create_text_skill,
                "data": create_data_skill
            }

        def activate(self, skill_name: str):
            """스킬 활성화"""
            if skill_name not in self.loaded_skills:
                print(f"\n🔄 '{skill_name}' 스킬 활성화 중...")
                creator = self.skill_creators.get(skill_name)
                if creator:
                    self.loaded_skills[skill_name] = creator()
                else:
                    raise ValueError(f"스킬 '{skill_name}' 없음")
            return self.loaded_skills[skill_name]

        def list_loaded(self):
            """로드된 스킬 목록"""
            return list(self.loaded_skills.keys())

    # 테스트
    manager = SkillManager()

    print("\n🧪 On-demand 활성화 테스트:")
    print("=" * 70)

    tasks = [
        ("math", "2 + 3 * 4"),
        ("text", "Hello World"),
        ("data", "sample,data,here")
    ]

    for skill_name, task_input in tasks:
        print(f"\n작업: {skill_name} - {task_input}")
        skill = manager.activate(skill_name)
        result = skill.invoke({list(skill.args.keys())[0]: task_input})
        print(f"결과: {result}")
        print(f"현재 로드된 스킬: {manager.list_loaded()}")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 4: 스킬 라이브러리 구축
# ============================================================================

def example_4_skill_library():
    """재사용 가능한 스킬 라이브러리"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 스킬 라이브러리 구축")
    print("=" * 70)

    print("""
💡 스킬 라이브러리:
   - 표준화된 스킬 인터페이스
   - 메타데이터 포함 (버전, 설명, 의존성)
   - 스킬 검색 및 필터링
    """)

    # 스킬 메타데이터
    class SkillMetadata:
        def __init__(self, name: str, version: str, description: str, category: str):
            self.name = name
            self.version = version
            self.description = description
            self.category = category

        def __repr__(self):
            return f"{self.name} v{self.version} [{self.category}]"

    # 스킬 라이브러리
    class SkillLibrary:
        def __init__(self):
            self.skills = {}

        def register(self, metadata: SkillMetadata, creator: Callable):
            """스킬 등록"""
            self.skills[metadata.name] = {
                "metadata": metadata,
                "creator": creator
            }

        def search(self, category: str = None):
            """카테고리로 스킬 검색"""
            results = []
            for name, skill_info in self.skills.items():
                metadata = skill_info["metadata"]
                if category is None or metadata.category == category:
                    results.append(metadata)
            return results

        def load(self, name: str):
            """스킬 로드"""
            if name not in self.skills:
                raise ValueError(f"스킬 '{name}' 없음")
            return self.skills[name]["creator"]()

    # 라이브러리 생성
    library = SkillLibrary()

    # 스킬 등록
    library.register(
        SkillMetadata("translator", "1.0", "다국어 번역", "language"),
        lambda: tool(lambda text, lang: f"{text} → {lang}")(lambda t, l: f"번역: {t}")
    )

    library.register(
        SkillMetadata("summarizer", "1.0", "텍스트 요약", "language"),
        lambda: tool(lambda text: f"요약: {text[:20]}...")(lambda t: f"요약: {t}")
    )

    library.register(
        SkillMetadata("calculator", "1.0", "수학 계산", "math"),
        lambda: tool(lambda expr: f"결과: {eval(expr)}")(lambda e: str(eval(e)))
    )

    library.register(
        SkillMetadata("analyzer", "1.0", "데이터 분석", "data"),
        lambda: tool(lambda data: f"분석: {len(data)} items")(lambda d: f"분석: {d}")
    )

    # 스킬 탐색
    print("\n📚 스킬 라이브러리:")
    print("=" * 70)

    print("\n모든 스킬:")
    for skill in library.search():
        print(f"  • {skill} - {skill.description}")

    print("\n언어 스킬:")
    for skill in library.search("language"):
        print(f"  • {skill} - {skill.description}")

    print("\n수학 스킬:")
    for skill in library.search("math"):
        print(f"  • {skill} - {skill.description}")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 5: 실전 - 플러그인 시스템
# ============================================================================

def example_5_plugin_system():
    """실전: 동적 플러그인 시스템"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 - 플러그인 시스템")
    print("=" * 70)

    print("""
🎯 실전 시나리오: AI 어시스턴트 플러그인 시스템

특징:
   - 사용자 요청에 따라 필요한 플러그인만 로드
   - 플러그인 추가/제거 가능
   - 캐싱으로 성능 최적화
    """)

    # 플러그인 레지스트리
    PLUGINS = {}

    def plugin(name: str, description: str, category: str):
        """플러그인 등록 데코레이터"""
        def decorator(func):
            PLUGINS[name] = {
                "creator": func,
                "description": description,
                "category": category
            }
            return func
        return decorator

    # 플러그인 정의
    @plugin("weather", "날씨 정보 제공", "info")
    def create_weather_plugin():
        @tool
        def get_weather(city: str) -> str:
            """날씨 조회"""
            return f"{city}의 날씨: 맑음, 20°C"
        return get_weather

    @plugin("translator", "텍스트 번역", "language")
    def create_translator_plugin():
        @tool
        def translate(text: str, lang: str = "영어") -> str:
            """번역"""
            prompt = f"{text}를 {lang}로 번역"
            response = llm.invoke(prompt)
            return response.content
        return translate

    @plugin("calculator", "계산기", "math")
    def create_calculator_plugin():
        @tool
        def calculate(expression: str) -> str:
            """계산"""
            try:
                return f"결과: {eval(expression)}"
            except:
                return "계산 오류"
        return calculate

    @plugin("memo", "메모 관리", "productivity")
    def create_memo_plugin():
        memos = []
        @tool
        def add_memo(content: str) -> str:
            """메모 추가"""
            memos.append(content)
            return f"메모 저장됨. 총 {len(memos)}개"
        return add_memo

    # 플러그인 매니저 (캐싱 포함)
    class PluginManager:
        def __init__(self):
            self.loaded = {}

        @lru_cache(maxsize=10)
        def load_cached(self, name: str):
            """캐싱된 플러그인 로드"""
            if name not in PLUGINS:
                raise ValueError(f"플러그인 '{name}' 없음")

            print(f"  📦 '{name}' 플러그인 로딩...")
            return PLUGINS[name]["creator"]()

        def load(self, name: str):
            """플러그인 로드"""
            if name not in self.loaded:
                self.loaded[name] = self.load_cached(name)
            return self.loaded[name]

        def list_available(self):
            """사용 가능한 플러그인"""
            return list(PLUGINS.keys())

        def list_loaded(self):
            """로드된 플러그인"""
            return list(self.loaded.keys())

        def search(self, query: str):
            """플러그인 검색"""
            results = []
            for name, info in PLUGINS.items():
                if query.lower() in name.lower() or query.lower() in info["description"].lower():
                    results.append((name, info["description"]))
            return results

    # AI 어시스턴트
    def ai_assistant():
        """플러그인 기반 AI 어시스턴트"""
        manager = PluginManager()

        print("\n🤖 AI 어시스턴트 시작")
        print("=" * 70)
        print(f"사용 가능한 플러그인: {', '.join(manager.list_available())}")

        while True:
            print("\n" + "-" * 70)
            print("명령어:")
            print("  /plugins - 플러그인 목록")
            print("  /search <query> - 플러그인 검색")
            print("  /load <name> - 플러그인 로드")
            print("  /use <name> <args> - 플러그인 사용")
            print("  /quit - 종료")

            cmd = input("\n입력: ").strip()

            if cmd == "/quit":
                break

            elif cmd == "/plugins":
                print("\n사용 가능한 플러그인:")
                for name, info in PLUGINS.items():
                    print(f"  • {name} [{info['category']}]: {info['description']}")

            elif cmd.startswith("/search "):
                query = cmd.split(" ", 1)[1]
                results = manager.search(query)
                print(f"\n'{query}' 검색 결과:")
                for name, desc in results:
                    print(f"  • {name}: {desc}")

            elif cmd.startswith("/load "):
                name = cmd.split(" ", 1)[1]
                try:
                    manager.load(name)
                    print(f"✅ '{name}' 로드 완료")
                    print(f"로드된 플러그인: {manager.list_loaded()}")
                except ValueError as e:
                    print(f"❌ {e}")

            elif cmd.startswith("/use "):
                parts = cmd.split(" ", 2)
                if len(parts) < 3:
                    print("❌ 사용법: /use <name> <args>")
                    continue

                name, args = parts[1], parts[2]
                try:
                    plugin = manager.load(name)
                    arg_name = list(plugin.args.keys())[0]
                    result = plugin.invoke({arg_name: args})
                    print(f"\n결과: {result}")
                except Exception as e:
                    print(f"❌ 오류: {e}")

    # 실행
    print("\n💡 사용 예시:")
    print("  /search 날씨")
    print("  /load weather")
    print("  /use weather 서울")

    choice = input("\nAI 어시스턴트를 시작하시겠습니까? (y/n): ").strip()
    if choice.lower() == "y":
        ai_assistant()

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("=" * 70)
    print("Part 7: Multi-Agent Systems")
    print("05. Skills Pattern (스킬 패턴)")
    print("=" * 70)

    while True:
        print("\n")
        print("📚 실행할 예제를 선택하세요:")
        print("-" * 70)
        print("1. Skills 패턴 개념")
        print("2. 동적 스킬 로딩")
        print("3. On-demand Tool Activation")
        print("4. 스킬 라이브러리 구축")
        print("5. 실전: 플러그인 시스템")
        print("0. 종료")
        print("-" * 70)

        choice = input("\n선택 (0-5): ").strip()

        if choice == "1":
            example_1_skills_concept()
        elif choice == "2":
            example_2_dynamic_loading()
        elif choice == "3":
            example_3_ondemand_activation()
        elif choice == "4":
            example_4_skill_library()
        elif choice == "5":
            example_5_plugin_system()
        elif choice == "0":
            print("\n👋 프로그램을 종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다.")

    print("\n" + "=" * 70)
    print("📚 학습 완료!")
    print("=" * 70)
    print("""
✅ 배운 내용:
   - Skills 패턴의 개념과 장점
   - 스킬 레지스트리와 동적 로딩
   - On-demand 도구 활성화
   - 재사용 가능한 스킬 라이브러리
   - 실전 플러그인 시스템

💡 핵심 요약:
   ┌─────────────────────────────────────────────────────────────────┐
   │ Skills는 재사용 가능한 Agent 컴포넌트를 동적으로 로딩          │
   │                                                                   │
   │ 주요 특징:                                                       │
   │ • 필요할 때만 로딩 (메모리 효율)                                │
   │ • 플러그인처럼 추가/제거 가능                                   │
   │ • 여러 프로젝트에서 재사용                                      │
   │ • 캐싱으로 성능 최적화                                          │
   │                                                                   │
   │ 사용 시점:                                                       │
   │ • 많은 기능을 가진 어시스턴트                                   │
   │ • 런타임에 기능 추가 필요                                       │
   │ • 리소스 최적화가 중요한 경우                                   │
   └─────────────────────────────────────────────────────────────────┘
    """)

if __name__ == "__main__":
    main()
