# LangGraph 완벽 가이드

> AI 에이전트와 워크플로우를 구축하기 위한 한국어 LangGraph 튜토리얼

<p align="center">
  <img src="https://img.shields.io/badge/LangGraph-0.2.0+-blue" alt="LangGraph Version">
  <img src="https://img.shields.io/badge/Python-3.10+-green" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

## 📖 소개

이 교재는 LangGraph를 처음 접하는 개발자부터 프로덕션 배포까지 다루는 종합 가이드입니다. 공식 문서를 기반으로 체계적인 학습 경로와 실습 코드를 제공합니다.

### 이 교재의 특징

- **체계적인 학습 경로**: 기초부터 고급까지 5개 Part로 구성
- **실행 가능한 코드**: 모든 예제 코드가 독립적으로 실행 가능
- **한국어 설명**: 모든 문서와 주석이 한국어로 작성
- **연습 문제**: 각 Part별 연습 문제와 해답 제공

## 🎯 학습 목표

이 교재를 완료하면 다음을 할 수 있습니다:

- ✅ LangGraph의 핵심 개념(State, Node, Edge) 이해
- ✅ 다양한 워크플로우 패턴 구현 (라우팅, 병렬 처리, 분기)
- ✅ ReAct Agent 및 Multi-Agent 시스템 구축
- ✅ 프로덕션 기능 활용 (메모리, 스트리밍, Human-in-the-Loop)
- ✅ 안정적인 배포를 위한 Durable Execution 구현

## 📚 목차

### Part 1: Foundation (기초)
- [Chapter 1: LangGraph 소개](docs/Part1-Foundation/01-introduction.md)
- [Chapter 2: 핵심 개념](docs/Part1-Foundation/02-core-concepts.md)
- [Chapter 3: 첫 번째 그래프](docs/Part1-Foundation/03-first-graph.md)
- [Chapter 4: State 관리](docs/Part1-Foundation/04-state-management.md)

### Part 2: Workflows (워크플로우)
- [Chapter 5: 워크플로우 패턴](docs/Part2-Workflows/05-workflow-patterns.md)
- [Chapter 6: 조건부 라우팅](docs/Part2-Workflows/06-conditional-routing.md)
- [Chapter 7: 병렬 실행](docs/Part2-Workflows/07-parallel-execution.md)
- [Chapter 8: Orchestrator-Worker](docs/Part2-Workflows/08-orchestrator-worker.md)

### Part 3: Agent (에이전트)
- [Chapter 9: 도구와 에이전트](docs/Part3-Agent/09-tools-and-agents.md)
- [Chapter 10: ReAct Agent](docs/Part3-Agent/10-react-agent.md)
- [Chapter 11: Multi-Agent 시스템](docs/Part3-Agent/11-multi-agent.md)
- [Chapter 12: 서브그래프](docs/Part3-Agent/12-subgraphs.md)

### Part 4: Production (프로덕션)
- [Chapter 13: Persistence](docs/Part4-Production/13-persistence.md)
- [Chapter 14: 메모리 시스템](docs/Part4-Production/14-memory.md)
- [Chapter 15: 스트리밍](docs/Part4-Production/15-streaming.md)
- [Chapter 16: Human-in-the-Loop](docs/Part4-Production/16-human-in-the-loop.md)
- [Chapter 17: Time Travel](docs/Part4-Production/17-time-travel.md)

### Part 5: Advanced (고급)
- [Chapter 18: Functional API](docs/Part5-Advanced/18-functional-api.md)
- [Chapter 19: Durable Execution](docs/Part5-Advanced/19-durable-execution.md)
- [Chapter 20: 배포 준비](docs/Part5-Advanced/20-deployment.md)

### Appendix (부록)
- [Appendix A: API 레퍼런스](docs/Appendix/A-api-reference.md)
- [Appendix B: 문제 해결](docs/Appendix/B-troubleshooting.md)
- [Appendix C: 모범 사례](docs/Appendix/C-best-practices.md)

## 🚀 시작하기

### 요구 사항

- Python 3.10 이상
- Anthropic API 키 또는 OpenAI API 키

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/langgraph-tutorial.git
cd langgraph-tutorial

# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 API 키 입력
```

### 첫 번째 예제 실행

```bash
# Part 1 기초 예제
python -m src.part1_foundation.01_hello_langgraph

# 또는 특정 챕터 실행
python -m src.part2_workflows.06_routing
```

## 📁 폴더 구조

```
langgraph-tutorial/
├── docs/                    # 📖 마크다운 문서
│   ├── Part1-Foundation/
│   ├── Part2-Workflows/
│   ├── Part3-Agent/
│   ├── Part4-Production/
│   ├── Part5-Advanced/
│   └── Appendix/
├── src/                     # 💻 실습 코드
│   ├── part1_foundation/
│   ├── part2_workflows/
│   ├── part3_agent/
│   ├── part4_production/
│   └── part5_advanced/
├── exercises/               # 💪 연습 문제
│   ├── part1_exercises.md
│   └── solutions/
├── examples/                # 🎯 완성 프로젝트
├── official_docs/           # 📚 공식 문서 참조
└── tests/                   # 🧪 테스트
```

## 📝 학습 방법

1. **마크다운 문서 읽기** - 개념 이해
2. **src/ 코드 실행** - 직접 실행하며 학습
3. **exercises/ 도전** - 연습 문제 풀기
4. **examples/ 참고** - 실전 응용

## 🔧 필수 환경 변수

```env
# .env.example
ANTHROPIC_API_KEY=your-api-key-here
OPENAI_API_KEY=your-openai-key-here  # 선택
LANGCHAIN_TRACING_V2=true            # LangSmith 트레이싱 (선택)
LANGCHAIN_API_KEY=your-langsmith-key # LangSmith API 키 (선택)
```

## 📊 공식 문서 참조

이 교재는 다음 공식 문서를 참조하여 작성되었습니다:

- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangChain 공식 문서](https://python.langchain.com/)
- [Anthropic API 문서](https://docs.anthropic.com/)

## 🤝 기여하기

버그 리포트, 기능 제안, 문서 개선 등 모든 기여를 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- [LangChain](https://github.com/langchain-ai/langchain) 팀
- [LangGraph](https://github.com/langchain-ai/langgraph) 팀
- [Anthropic](https://www.anthropic.com/)

---

**Happy Learning! 🚀**
