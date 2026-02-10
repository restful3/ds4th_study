# LangSmith 배포

LangChain Agent를 프로덕션에 배포할 준비가 되면 LangSmith는 Agent 워크로드를 위해 설계된 관리형 호스팅 플랫폼을 제공합니다. 기존 호스팅 플랫폼은 상태 비저장, 단기 실행 웹 애플리케이션을 위해 구축되었지만 LangGraph는 **지속적인 상태와 백그라운드 실행이 필요한 상태 저장, 장기 실행 Agent를 위해 특별히 구축되었습니다**. LangSmith는 인프라, 스케일링 및 운영 문제를 처리하므로 저장소에서 직접 배포할 수 있습니다.

## 필수 조건

시작하기 전에 다음을 확인하세요:

- [**GitHub 계정**](https://github.com/)
- [**LangSmith 계정**](https://smith.langchain.com/) (가입 무료)

## Agent 배포

### 1. GitHub에 저장소 생성

애플리케이션 코드는 LangSmith에 배포하기 위해 GitHub 저장소에 있어야 합니다. 공개 및 비공개 저장소 모두 지원됩니다. 이 빠른 시작을 위해 먼저 [**로컬 서버 설정 가이드**](https://docs.langchain.com/oss/python/langchain/studio)를 따라 앱이 LangGraph 호환 가능한지 확인하세요. 그런 다음 코드를 저장소로 푸시합니다.

### 2. LangSmith에 배포

1. **LangSmith 배포 페이지로 이동**

   [LangSmith](https://smith.langchain.com/)에 로그인합니다. 왼쪽 사이드바에서 **배포**를 선택합니다.

2. **새 배포 생성**

   **+ 새 배포** 버튼을 클릭합니다. 필수 필드를 채울 수 있는 창이 열립니다.

3. **저장소 연결**

   처음 사용자이거나 이전에 연결되지 않은 비공개 저장소를 추가하는 경우 **새 계정 추가** 버튼을 클릭하고 지침을 따라 GitHub 계정을 연결합니다.

4. **저장소 배포**

   애플리케이션의 저장소를 선택합니다. **제출**을 클릭하여 배포합니다. 이는 완료하는 데 약 15분이 소요될 수 있습니다. **배포 세부정보** 보기에서 상태를 확인할 수 있습니다.

### 3. Studio에서 애플리케이션 테스트

애플리케이션이 배포되면:

1. 방금 생성한 배포를 선택하여 더 자세한 내용을 봅니다.
2. 오른쪽 상단의 **Studio** 버튼을 클릭합니다. Studio가 열려 그래프를 표시합니다.

### 4. 배포의 API URL 가져오기

1. LangGraph의 **배포 세부정보** 보기에서 **API URL**을 클릭하여 클립보드에 복사합니다.
2. `URL`을 클릭하여 클립보드에 복사합니다.

### 5. API 테스트

이제 API를 테스트할 수 있습니다:

#### Python

1. LangGraph Python을 설치합니다:

```bash
pip install langgraph-sdk
```

2. Agent에 메시지를 보냅니다:

```python
from langgraph_sdk import get_sync_client  # 또는 비동기의 경우 get_client

client = get_sync_client(url="your-deployment-url", api_key="your-langsmith-api-key")

for chunk in client.runs.stream(
    None,       # Threadless run
    "agent",    # langgraph.json에 정의된 Agent 이름.
    input={
        "messages": [{
            "role": "human",
            "content": "What is LangGraph?",
        }],
    },
    stream_mode="updates",
):
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)
    print("\n\n")
```

#### Rest API

```bash
curl -s --request POST \
  --url <DEPLOYMENT_URL>/runs/stream \
  --header 'Content-Type: application/json' \
  --header "X-Api-Key: <LANGSMITH API KEY>" \
  --data "{
    \"assistant_id\": \"agent\",
    \"input\": {
      \"messages\": [
        {
          \"role\": \"human\",
          \"content\": \"What is LangGraph?\"
        }
      ]
    },
    \"stream_mode\": \"updates\"
  }"
```

> [!TIP]
> LangSmith는 자체 호스팅 및 하이브리드를 포함한 추가 호스팅 옵션을 제공합니다. 자세한 내용은 [플랫폼 설정 개요](https://docs.smith.langchain.com/self_hosting)를 참조합니다.
