# 변경 기록

Python 패키지에 대한 업데이트 및 개선 사항 로그입니다.

> 구독: 우리의 변경 기록에는 Slack, 이메일, Readybot 또는 RSS Feeds to Discord Bot과 같은 Discord 봇 및 기타 구독 도구와 통합할 수 있는 RSS 피드가 포함되어 있습니다.

---

## 2025년 12월 15일

**langchain**, **integrations**

### **langchain v1.2.0**

*   **`create_agent`**: Tool에 새로운 `extras` 속성을 통한 공급자별 Tool 매개변수 및 정의에 대한 단순화된 지원. 예:
    *   Anthropic의 [프로그래밍 방식 Tool 호출](/oss/python/integrations/chat/anthropic#programmatic-tool-calling) 및 [Tool 검색](/oss/python/integrations/chat/anthropic#tool-search)과 같은 공급자별 구성.
    *   [Anthropic](/oss/python/integrations/chat/anthropic#built-in-tools), [OpenAI](/oss/python/integrations/chat/openai#responses-api) 및 기타 공급자에서 지원하는 클라이언트 측 실행 기본 제공 Tool.
*   Agent `response_format`에서 엄격한 스키마 준수 지원 ([ProviderStrategy](/oss/python/langchain/structured-output#provider-strategy) 문서 참조).

---

## 2025년 12월 8일

**langchain**, **integrations**

#### langchain-google-genai v4.0.0

Google GenAI 통합을 Google의 통합 Generative AI SDK를 사용하도록 다시 작성했습니다. 이는 Gemini API와 Vertex AI Platform에 대한 액세스를 동일한 인터페이스 아래에서 제공합니다. 여기에는 최소한의 주요 변경 사항과 `langchain-google-vertexai`의 사용 중단된 패키지가 포함됩니다.

자세한 내용은 전체 [릴리스 노트 및 마이그레이션 가이드](https://github.com/langchain-ai/langchain-google/discussions/1422)를 참조합니다.

---

## 2025년 11월 25일

**langchain**

#### langchain v1.1.0

*   **[모델 프로필](/oss/python/langchain/models#model-profiles)**: Chat 모델은 이제 `.profile` 속성을 통해 지원되는 기능과 성능을 노출합니다. 이 데이터는 모델 기능 데이터를 제공하는 오픈소스 프로젝트인 [models.dev](https://models.dev)에서 파생됩니다.
*   **요약 Middleware**: 문맥 인식 요약을 위해 모델 프로필을 사용하는 유연한 트리거 지점을 지원하도록 업데이트되었습니다.
*   **[구조화된 출력](/oss/python/langchain/structured-output)**: `ProviderStrategy` 지원(기본 구조화된 출력)은 이제 모델 프로필에서 추론될 수 있습니다.
*   **[create_agent에 대한 SystemMessage](/oss/python/langchain/middleware/custom#working-with-system-messages)**: `SystemMessage` 인스턴스를 `create_agent`의 `system_prompt` 매개변수로 직접 전달하는 지원으로 캐시 제어 및 구조화된 콘텐츠 블록과 같은 고급 기능을 사용할 수 있습니다.
*   **[모델 재시도 Middleware](/oss/python/langchain/middleware/built-in#model-retry)**: 구성 가능한 지수 백오프를 사용하여 실패한 모델 호출을 자동으로 재시도하기 위한 새로운 Middleware.
*   **[콘텐츠 중재 Middleware](/oss/python/langchain/middleware/built-in#content-moderation)**: Agent 상호작용에서 안전하지 않은 콘텐츠를 감지하고 처리하기 위한 OpenAI 콘텐츠 중재 Middleware. 사용자 입력, 모델 출력 및 Tool 결과 확인을 지원합니다.

---

## 2025년 10월 20일

**langchain**, **langgraph**

#### v1.0.0

##### langchain
*   [릴리스 노트](/oss/python/releases/langchain-v1)
*   [마이그레이션 가이드](/oss/python/migrate/langchain-v1)

##### langgraph
*   [릴리스 노트](/oss/python/releases/langgraph-v1)
*   [마이그레이션 가이드](/oss/python/migrate/langgraph-v1)

> 문제가 발생하거나 피드백이 있으면 [이슈를 열어](https://github.com/langchain-ai/docs/issues/new/choose) 주시기 바랍니다. v0.x 문서를 보려면 [아카이브된 콘텐츠](https://github.com/langchain-ai/langchain/tree/v0.3/docs/docs) 및 [API 참조](https://reference.langchain.com/v0.3/python/)로 이동합니다.
