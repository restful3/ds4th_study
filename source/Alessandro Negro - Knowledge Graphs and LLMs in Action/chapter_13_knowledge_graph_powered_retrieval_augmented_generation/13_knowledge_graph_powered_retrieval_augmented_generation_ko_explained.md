---
lang: ko
format:
  html:
    toc: true
    embed-resources: true
    theme: cosmo
---

# 지식 그래프 기반 검색 증강 생성(Graph RAG) — 쉬운 해설판

> 이 문서는 원서 13장 "Knowledge graph–powered retrieval-augmented generation"을 한국어로 풀어 설명하는 해설판입니다. 원문의 모든 문단·절·그림·표·코드·수식을 빠짐없이 다루되, 번역을 넘어 "왜 그런지"까지 함께 이야기하듯 설명합니다. 어려운 용어는 처음 나올 때 영어 원어와 한 줄 정의를 붙이고, 이후에는 한국어로만 부릅니다. 이 장의 큰 주제는 **지식 그래프(KG)** 를 LLM의 검증된 사실 대장으로 삼아, 그럴듯하지만 틀린 답(환각)을 막는 **그래프 RAG** 시스템을 만드는 것입니다.

---

## 4부 소개 — 지식 그래프와 LLM으로 완성하는 정보 검색

책의 이 마지막 파트에서 **지식 그래프(Knowledge Graph, KG)**, 즉 개념과 개념 사이의 관계를 그래프 구조로 정리해 둔 검증된 사실 저장소와 **거대 언어 모델(Large Language Model, LLM)**, 즉 방대한 텍스트로 학습해 사람처럼 말하는 인공지능 모델의 결합이 실무적인 정점에 도달합니다. 여기서 우리는 이 두 기술을 함께 써서 정확하고 신뢰할 수 있는 정보 검색을 어떻게 해내는지 살펴봅니다. 초점은 하나입니다. KG를 "정답에 해당하는 근거(ground truth)"로 삼아 LLM의 능력을 끌어올리면서도, 동시에 LLM이 사실을 지어내는 **환각(hallucination)** 을 막는 시스템을 실제로 구현하는 것입니다.

13장은 KG와 LLM을 **검색 증강 생성(Retrieval-Augmented Generation, RAG)**, 즉 외부에서 관련 정보를 찾아와 그 맥락 위에서 답을 생성하는 기법으로 결합하는 방법을 다룹니다. 그래프 RAG 시스템이 구조화된 데이터와 언어 이해 능력을 함께 활용해 더 정확하고 투명한 답을 어떻게 제공하는지 보여 줍니다.

14장은 도메인 전문가의 추론 방식을 흉내 내는 정교한 질의응답 시스템을 만드는 법을 설명합니다. 사용자의 의도를 파악하는 **의도 탐지(intent detection)**, 스키마를 번역하는 과정, 전문가 지식을 시스템에 심는 접근을 체계적으로 다룹니다.

15장은 이 모든 개념을 하나로 모아, **LangGraph** 와 **Streamlit** 을 이용한 완전하고 실제로 동작하는 구현을 보여 줍니다. 여러 전문화된 에이전트가 질문 처리의 서로 다른 측면을 나누어 맡으면서도, 시스템의 관찰 가능성과 확장성을 유지하는 모듈형 파이프라인을 어떻게 만드는지 시연합니다.

이 세 장을 통해 여러분은 KG와 LLM의 장점을 결합한 **운영 수준(production-ready)** 시스템을 실제로 구현할 수 있는 실용 지식을 얻게 됩니다. 신뢰할 수 있고, 지식에 근거한 AI 솔루션을 배포하려는 조직에게 이 파트는 하나의 종합 안내서가 될 것입니다.

---

## 지식 그래프 기반 검색 증강 생성(Graph RAG) — 이 장의 문 열기

### 이 장에서 다루는 것 — 한눈에 보기

이 장에서 우리가 함께 살펴볼 내용은 세 가지입니다.

- LLM을 **AI 에이전트(AI agent)** 로서 쓸모 있게 만드는 법
- 검색 증강 생성(RAG)을 이용해 맥락으로 LLM을 근거에 묶어 두는 법
- KG 기반 RAG 시스템을 실제로 만드는 법

2023년은 AI가 세상을 뒤집어 놓은 해였습니다. **자연어 처리(Natural Language Processing, NLP)**, 즉 컴퓨터가 사람의 언어를 다루게 하는 분야에서 일하던 데이터 과학자와 머신러닝 엔지니어들은 단순히 새로운 장난감을 하나 얻은 게 아니었습니다. 그들의 업무 방식 자체가 완전히 뒤바뀌었습니다. 2022년 말 OpenAI가 GPT-3.5 모델을 공개한 것이 이 전환의 신호탄이었습니다. 갑자기, 다운스트림 작업 하나하나마다 몇 달, 몇 년에 걸쳐 맞춤형 학습 데이터셋과 모델을 만들 필요가 사라졌습니다. 약간의 영리한 **프롬프트 엔지니어링(prompt engineering)**, 즉 모델에게 던지는 지시문을 잘 설계하는 기술만 있으면 거의 누구나 NLP 애플리케이션을 만들 수 있게 되었습니다.

하지만 LLM이 강력하다고 해서 모든 문제를 푸는 마법 같은 해법인 것은 결코 아닙니다. 실제 기업 환경의 운영 수준 시나리오에서 쓸모 있게 만들려면 아직 해야 할 일이 많이 남아 있습니다. 바로 이 깨달음에서 AI 에이전트라는 개념, 그리고 그것을 구현하는 여러 라이브러리와 **LLMOps(LLM operations)**, 즉 LLM을 운영·관리하는 기술이 등장했습니다.

이 장에서 다룰 주제들은 그림 13.1의 멘탈 모델(개념 지도)에 담겨 있습니다. 책의 2부와 3부에서는 비공개 데이터를 KG로 변환하는 방법을 보여 주었습니다. 이제 우리는 그렇게 만든 KG를 입력으로 삼아 동작하는 챗봇을 어떻게 만드는지 탐구합니다.

![그림 13.1 KG 기반 질의응답을 위한 AI 에이전트 설계. 에이전트는 여러 도구를 갖추고 있으며, 이 도구들은 벡터 데이터베이스와 KG 같은 외부 데이터 소스를 활용해 사용자 질문에 필요한 맥락을 제공합니다.](images/ko/figure-13-1-ko.png)

그림 13.1은 KG 기반 질의응답을 위한 AI 에이전트 설계를 나타냅니다. 에이전트는 여러 도구를 손에 쥐고 있고, 이 도구들은 벡터 데이터베이스와 KG 같은 외부 데이터 소스를 활용해 사용자의 질문에 필요한 맥락을 채워 줍니다.

---

### 13.1 AI 에이전트 — 스스로 판단하는 자율 시스템

**AI 에이전트(AI agent)** [1]는 현대 지능형 시스템의 능력이 크게 진화한 형태입니다. 그 본질을 보면, AI 에이전트는 자신의 환경과 상호작용하면서 구체적이고 복잡한 작업을 수행하도록 설계된 자율적 주체입니다. 미리 정해진 명령 집합을 그대로 따르는 전통적인 소프트웨어 프로그램과 달리, AI 에이전트는 어느 정도의 자율성, 적응성, 그리고 "지능"을 갖추고 있습니다. 덕분에 스스로 의사결정을 내리고, 경험에서 배우며, 변하는 상황에 실시간으로 반응할 수 있습니다.

완전히 동작하는 챗봇 하나를 떠올려 봅시다. 아주 단순한 질문에서 시작할 수 있습니다. "프랑스의 수도는 어디인가요?" 현재 모델 리더보드 상위권에 있는 어떤 LLM이든 즉시, 정확하게 "프랑스의 수도는 파리입니다"라고 답할 것입니다. 훌륭합니다! 잘 작동하네요! 그런데 이런 질문이 과연 수천억 개의 파라미터를 가진 모델을 만들고 운영하는 데 드는 막대한 비용을 정당화할 만큼의 투자 대비 수익(ROI)을 만들어 낼까요? 아마 아닐 것입니다. 정말로 값어치 있는 질문과 작업은 훨씬 더 복잡하며, 대개 다음 중 최소한 하나를 요구합니다.

- **고급 다단계 추론 능력** — 수학 퍼즐을 푸는 것 같은 연역 과제나, 중간 단계에서 나온 출력에 따라 원래의 실행 계획을 스스로 조정해야 하는 적응형 시스템을 생각해 보세요.
- **개념들 사이의 깊은 관계 패턴에 대한 이해** — 특정 토론 주제의 인플루언서를 찾아내고 싶은 소셜 네트워크 분석이나, 병목 지점을 짚어내고 해소해야 하는 공급망 문제를 떠올려 보세요.
- **최신의, 종종 외부이고 비공개인, 모델이 학습 중에 본 적 없는 데이터에 대한 접근** — 이는 흔히 LLM의 **지식 컷오프(knowledge cutoff)**, 즉 학습에 쓰인 데이터의 시점 한계라고 불립니다. 이 모델들은 너무 커서 자주 재학습할 수 없기 때문에 "내일 날씨 예보는 어때?" 같은 질문에 답할 수 없고, 하물며 민감한 내부 데이터와 관련된 질문은 더욱 그렇습니다. 특히 서로 다른 접근 권한을 가진 여러 사용자(페르소나)를 다뤄야 할 때는 더욱 까다롭습니다.

콘텐츠 작성을 돕는 어시스턴트를 상상해 봅시다. 이 어시스턴트는 사용자가 기사, 블로그 글, 심지어 소셜 미디어 게시물을 쓰는 것을 도와줍니다. 사람이라면 이 일을 어떻게 접근할까요? 먼저 요청받은 주제를 조사할 것입니다. 관련 기사와 블로그, 책을 읽고, 위키피디아 같은 지식 베이스에서 기본적인 사실을 끌어옵니다. 그런 다음 초안을 쓰고, 친구나 동료에게 검토를 부탁하고, 발행 전에 최종적으로 다듬습니다. 이 과정이 그대로 지능형 시스템의 설계에 반영됩니다. 이 시스템을 위해 우리는 여러 AI 에이전트를 만들 것입니다. 조사를 맡는 **리서처(Researcher)**, 무엇을 쓰느냐에 따라 여러 명의 **라이터(Writer)**, 그리고 검토를 맡는 **리뷰어(Reviewer)** 입니다. 여러 에이전트로 구성된 시스템은 일종의 롤플레잉 게임으로 생각할 수 있습니다. 서로 다른 에이전트(플레이어)들이 각각 전문화된 역할을 맡고, 하나의 목표를 이루기 위해 서로 소통하는 것입니다.

---

### 13.2 LLM과 대화하기 — 기억을 가진 챗봇 만들기

AI 에이전트라는 개념을 이해했으니, 이제 실제 예제를 살펴볼 차례입니다. 가능한 가장 단순한 시나리오를 생각해 봅시다. 사용자가 LLM과 자유롭게 대화하는 챗봇 에이전트입니다. 처음 질문을 던진 뒤, 사용자는 대화하듯 자연스럽게 후속 질문을 이어갈 수 있습니다. 이런 에이전트를 만들려면 딱 두 가지만 있으면 됩니다. 사전 학습된 LLM에 대한 접근, 그리고 지금까지 오간 모든 질문과 답을 기억해 대화 같은 경험을 만들어 주는 **메모리(memory)** 입니다. 이런 에이전트는 다음 리스트처럼 간단한 클래스 하나로 작성할 수 있습니다.

#### 리스트 13.1 기억을 갖춘 대화형 AI 에이전트

아래 코드는 원서의 OCR 상태를 그대로 보존한 것입니다. 핵심은 `self.messages` 라는 인스턴스 변수가 에이전트의 "메모리" 역할을 하며, 전체 메시지 이력을 담아 둔다는 점입니다. 질문을 던지면 그 질문이 메모리에 추가되고, 답이 나오면 그 답도 사용자에게 돌려주기 전에 메모리에 추가됩니다.

```python
class Agent:
    def __init__(self, model: str = "gpt-4o-mini", system: str = None):
        self.model = model
        self.system = system                # 메모리 역할을 하는 인스턴스 변수:
        self.messages = list()              # 전체 메시지 이력을 담는다.
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        if self.system is None or len(self.system) == 0:
            self.system = "You are an AI assistant providing straightforward concise answers."
        self.messages.append({"role": "system", "content": self.system})

    def __call__(self, message: str) -> str:    # 에이전트에게 질문하면,
        self.messages.append({"role": "user",   # 그 질문이 메모리에 추가된다.
                              "content": message})
        answer = self.execute()
        self.messages.append({"role": "assistant",   # 답도 사용자에게 반환되기 전에
                              "content": answer})     # 메모리에 추가된다.
        return answer

    def execute(self) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=self.messages)
        return completion.choices[0].message.content


if __name__ == "__main__":
    agent = Agent()                                          # 첫 번째 질문
    q = "Who are the top influencers of cyclotron funding?"  # 후속 질문:
    print(f"> Question: {q}\n> Answer: {agent(q)}")          # 에이전트는 앞선
                                                             # 질문-답변 쌍을 알고 있다.
    q = "And in the context of the 1930s, related to the Rockefeller Foundation?"
    print(f"> Question: {q}\n> Answer: {agent(q)}")
```

에이전트는 모델 버전으로 초기화되고, 시스템 메시지가 그 범위를 규정하며, OpenAI API 키는 `.env` 파일을 통해 환경 변수로 제공됩니다. 인스턴스 변수 `messages` 는 에이전트의 메모리를 나타내며 전체 메시지 이력을 담습니다. 더 상호작용적인 경험을 위해 이 에이전트를 주피터(Jupyter)에서 돌릴 수도 있고, 이상적으로는 UI 계층을 얹어 진짜 대화처럼 만들 수도 있습니다. 하지만 여기서는 단순함을 위해 평범한 파이썬 스크립트에서 실행합니다. 우리는 처음 질문을 하나 던지고, 이어서 맥락을 좁히는 후속 질문을 던집니다. 그 출력은 다음과 같습니다.

#### 리스트 13.2 리스트 13.1 대화의 출력

아래는 사이클로트론(cyclotron, 입자를 가속하는 장치) 연구 자금 지원의 주요 인플루언서가 누구인지 묻고, 이어서 1930년대 록펠러 재단 맥락으로 좁혀 물었을 때의 출력입니다.

```text
> Question: Who are the top influencers of cyclotron funding?
> Answer: The top influencers of cyclotron funding typically include
government agencies, research institutions, private foundations, and
industry stakeholders. Key players may include:

1. **National Institutes of Health (NIH)** - Provides significant funding for medical research involving cyclotrons.
2. **Department of Energy (DOE)** - Supports research related to nuclear science and particle accelerators.
3. **National Science Foundation (NSF)** - Funds various scientific research projects, including those using cyclotrons.
4. **Private Foundations** - Organizations like the Wellcome Trust and the Gates Foundation may fund related research.
5. **Universities and Research Institutions** - Institutions with cyclotron facilities often seek grants and partnerships for funding.
6. **Pharmaceutical Companies** - They may invest in cyclotron technology for drug development, particularly in radiopharmaceuticals.

These entities influence funding through grants, partnerships, and collaborative research initiatives.

> Answer: In the context of the 1930s, the Rockefeller Foundation played a
significant role in funding scientific research, including advancements in
nuclear physics and cyclotron development. Key influencers related to
cyclotron funding during this period included:

1. **John D. Rockefeller Jr.** - As a prominent philanthropist, he supported various scientific initiatives through the foundation.
2. **Vannevar Bush** - An influential engineer and science administrator who advocated for government and private funding in scientific research.
3. **Ernest O. Lawrence** - The inventor of the cyclotron, whose work received support from the Rockefeller Foundation and other entities, helping to advance particle accelerator technology.

The Rockefeller Foundation's funding helped establish research programs and
facilities that contributed to the development of cyclotrons and related
scientific fields during the 1930s.
```

첫 번째 질문에 대한 답은 정확하지만 매우 두루뭉술합니다. 반면 맥락을 명확히 해 준 후속 질문에는 더 나은 답이 돌아옵니다. 여기서 우리는 익숙한 두 이름을 발견합니다(5장과 6장에서 만든 KG를 떠올려 보세요). 바로 어니스트 O. 로런스(Ernest O. Lawrence)와 배니바 부시(Vannevar Bush)입니다. 놀랄 일은 아닙니다. 생짜(out-of-the-box) 모델은 우리가 답의 근거로 삼기를 기대하는 그 데이터를 직접 보지 않는 한, 구체적인 답을 내놓을 수 없습니다. 이 점은 곧 더 자세히 배우게 됩니다.

이런 식으로 복잡도를 하나씩 더해 가며 처음부터 에이전트를 만들 수도 있습니다. 하지만 다행히 그럴 필요는 없습니다. 현재의 AI 붐과 손을 맞잡고, 그 주위에 거대한 엔지니어링 생태계가 함께 발전했기 때문입니다.

---

### 13.3 운영 환경의 도전 과제 — 실무 배포의 벽

실제로 쓸모 있는 에이전트를 개발하는 일은 복잡하며, LLM과 관련된 여러 도전 과제와 우려 사항을 반드시 고려해야 합니다.

- **환각(hallucination), 흔히 말하는 "지어내기"** — LLM은 시퀀스에서 가장 그럴듯한 다음 토큰을 예측하도록 학습됩니다. 이 방식은 본질적으로, 그럴듯하게 들리지만 실은 지어낸 사실을 만들어 내기 쉽게 만듭니다. 특히 학습 데이터에 없던 주제에 대해 질문받을 때 이런 일이 두드러집니다. 지식이 없더라도 모델은 자신이 학습한 작업을 그대로 수행합니다. 즉, 수십억 개의 파라미터에 인코딩된 경험을 바탕으로 가장 그럴듯한 출력을 생성합니다. 그 답은 일관되고 완전히 설득력 있게 들리지만, 부분적으로 혹은 완전히 부정확할 수 있습니다. LLM이 박학다식하지만 가끔 말을 지어내는 달변가라는 비유가 딱 들어맞는 대목입니다.
- **최신성(freshness), 흔히 말하는 "지식 컷오프"** — LLM은 방대한 양의 데이터로 학습되며, 이것이 그들을 강력하게 만들지만 학습 비용을 극도로 비싸게 만듭니다. 그래서 재학습은 1년에 한두 번 정도만 이루어지고, 결과적으로 LLM은 (그다지 최근도 아닌) 최근 동향에 대해서도 정확한 답을 줄 수 없습니다.
- **투명성(transparency)** — 우리는 질문에 대해 일관된 답을 얻지만, 그 답이 어떻게 생성되었는지에 대한 통찰은 전혀 얻지 못합니다. 정보의 출처와 신뢰도, 추론 과정, 확신 수준 같은 것들은 기업용 솔루션을 개발할 때 반드시 신경 써야 하는 문제입니다.
- **데이터 프라이버시(data privacy)** — 개인적이고 사적이며 민감할 수 있는 데이터를 유출 없이 모델 학습에 쓰는 것은 많은 애플리케이션에서 우려 사항입니다. 그리고 많은 조직에서는 사람 집단마다 서로 다른 데이터 접근 권한을 가지고 있습니다.
- **비용(cost)** — 현재 최고 성능의 AI 모델을 학습·배포·유지하는 데는 금전적으로나 환경적으로나 상당한 비용이 듭니다. 큰 모델을 학습하는 데 필요한 연산 능력은 어마어마하며, 이는 높은 에너지 소비와 상당한 탄소 발자국으로 이어집니다. 그래서 이런 모델은 주로 자금이 넉넉한 기업만 접근할 수 있습니다. 더 작고 전문화된 LLM이 확산되면서 비용은 낮아지고 있지만, 여전히 중요하게 고려해야 할 사항입니다.
- **윤리적 우려와 편향(ethical concerns and biases)** — 모델은 편견이나 유해한 내용을 담고 있을 수 있는 데이터셋으로부터 학습하기 때문에, 고정관념, 잘못된 정보, 차별적 관점을 뜻하지 않게 재생산하거나 증폭시킬 수 있습니다. 이는 모델의 배포와 사회적 영향에 대한 심각한 윤리적 문제를 제기합니다.

이 문제들을 해결하려면 "질문 넣으면 답 나온다"는 단순한 시나리오를 넘어서야 하며, 이는 곧 더 복잡한 에이전트를 만들어야 함을 뜻합니다. 예를 들어 최신성 문제를 다루기 위해, 우리는 대화형 에이전트에게 외부 소스에서 정보를 가져와 활용할 수 있는 도구를 쥐여 주고 싶습니다. 최신 날씨 예보 데이터를 내려받거나, 뉴스 기사를 가져오거나, 지식 그래프처럼 항상 최신 상태로 유지되는 지식 베이스에서 콘텐츠를 끌어오는 식입니다. 이것이 바로 이 장의 나머지에서 다룰 주제입니다.

---

### 13.4 비공개 데이터에 대해 AI와 대화하기 — 우리 도메인 전문가로 만들기

LLM 모델은 전문화된 도메인에 대해서는 지식이 제한적입니다. 이런 경우 우리는, 모델의 뛰어난 언어 이해 능력과 일반 지식은 그대로 유지하면서도, 그 모델을 우리 도메인과 우리의 (종종 민감하거나 비밀인) 비공개 데이터에 대한 전문가로 만들 방법이 필요합니다.

5장과 6장에서 논의한 록펠러 아카이브 센터(Rockefeller Archive Center) 사례를 떠올려 봅시다. 우리는 1930년대에 록펠러 재단에서 이루어진 연구비 지원 과정을 추적하는 KG를 만들었습니다. 이 KG는 지원된 연구비를, 지원 금액·연구 주제·대학·연구자 같은 관련 정보와 함께 담고 있습니다. 나아가 연구비 승인에 앞서 재단 관계자들과 지원 신청자들 사이에 오간 물밑 대화(즉, 누가 누구와 무엇에 대해 이야기했는지)까지 포착하고 있습니다. 이렇게 해서 우리는, 이전에 전체가 공개된 적 없는 독점 데이터로부터 구축한 **영향력 네트워크(influence network)** 를 손에 넣게 되었습니다. 이 네트워크 덕분에 "사이클로트론 연구 자금 지원의 인플루언서는 누구였는가?" 같은 질문에 정확히 답할 수 있습니다.

5장과 6장에서 우리는 잘 튜닝된 그래프 시각화와 대시보드를 통해 이런 사례를 위한 전통적인 KG 기반 시스템을 설계하는 법을 보았습니다. 이제 흥미로운 질문이 남습니다. 사용자에게 Cypher 쿼리 언어를 배우게 하거나, 차트와 표를 읽고 해석하게 하거나, 그래프 데이터 구조를 직접 조작하고 탐색하게 하지 않고도, 폭넓은 사용자에게 똑같은 가치를 전달하는 AI 인터페이스를 만들 수 있을까요? 이것이 바로, 사전 학습된 LLM과 여러 개의 맥락 검색 도구를 사용하는 AI 에이전트가 맡을 일입니다. 이 과정을 **검색 증강 생성(RAG)** 이라고 부릅니다.

#### 13.4.1 검색 증강 생성(RAG) — 근거로 붙잡아 두기

**RAG** [2]는 사전 학습된 생성 모델의 한계, 즉 앞서 언급한 환각·최신성·투명성·데이터 프라이버시 같은 문제를 다루기 위해 개발된 기법입니다. RAG는 사전 학습된 LLM의 지식과 언어 이해 능력을, 질문과 관련해 외부 데이터 소스에서 검색해 온 추가 맥락과 결합합니다. 여기서 외부 데이터 소스란 구조화된 데이터베이스일 수도 있고, 텍스트나 이미지 같은 비구조화 데이터셋일 수도 있습니다.

실제로 RAG 에이전트는 LLM, 에이전트의 단계를 안내하는 프롬프트, 그리고 하나 이상의 도구를 조합해서 코딩됩니다. 여기서 도구란 본질적으로 질문과 관련된 외부 정보를 검색해 오는 함수입니다. 그런 다음 모델은 사용자의 질문과, 제공된 외부 정보를 맥락으로 함께 받아 답을 생성하도록 요청받습니다. 내일 날씨 예보를 묻는 상황을 생각해 보세요. 생짜 모델은 날씨 예보 API를 호출하는 도구를 쓰게 해 주지 않는 한 이 질문에 답할 수 없습니다. 도구를 쓰게 해 주면, 이제 AI는 자신의 언어 이해 능력과 외부의 정확하고 최신인 정보를 결합해 정확한 답을 생성할 수 있습니다.

이런 의미에서 RAG는 **근거화(grounding) 기법**입니다. 모델이 제멋대로 날뛰도록(환각) 놔두는 대신, 답 생성의 범위를 제공된 맥락으로 제한하는 것입니다. 이렇게 하면 모델이 사실을 지어낼 가능성이 크게 줄어듭니다. KG가 검증된 사실 대장이라면, RAG는 그 대장을 펼쳐 놓고 그 안에서만 답하게 만드는 규칙인 셈입니다.

> **참고** 우리는 이 모델들이 확률적이라는 사실을 결코 완전히 피해 갈 수 없습니다. 이들은 시퀀스에서 가장 확률이 높은 다음 토큰을 예측하도록 학습되었기에, RAG 같은 기법을 쓰더라도 여전히 엉뚱한 방향으로 갈 수 있습니다. 이 점은 지능형 시스템을 설계할 때 반드시 명심해야 합니다. 시스템은 사람을 대체하는 것이 아니라 사람을 보강해야 합니다. 우리가 만드는 어떤 제품이든, 피드백 검증이나 감독 메커니즘을 통해 **사람을 루프 안에 두는 것(human in the loop)** 이 필수적이라고 우리는 믿습니다.

이제 예제를 하나 살펴봅시다. RAG 초창기에는 맥락이 거의 전적으로 텍스트 문서 데이터베이스에서 나왔습니다. 이 과정이 그림 13.2에 나와 있습니다. 문서는 더 작은 조각(예를 들어 문단)으로 쪼개졌고, 그 조각들은 의미를 담아내는 고정 길이 벡터인 **임베딩(embedding)** 으로 변환되었습니다. 이 임베딩들은 **벡터 데이터베이스(vector database)** 에 저장되고 색인되었습니다. 록펠러 아카이브 센터 사례에서 임베딩을 생성하고 저장하는 과정이 다음 리스트에 나와 있습니다.

![그림 13.2 벡터 검색 기반 검색 증강 생성. 문서는 조밀한(dense) 벡터 표현으로 임베딩되어 벡터 데이터베이스에 색인됩니다. 사용자가 질문을 하면 그 질문 역시 임베딩되고, 데이터베이스에서 가장 유사한 문서들이 검색됩니다. 그런 다음 에이전트가 최종 답을 생성합니다.](images/ko/figure-13-2-ko.png)

그림 13.2는 벡터 검색 기반 RAG의 흐름을 보여 줍니다. 문서는 조밀한 벡터로 임베딩되어 벡터 데이터베이스에 색인되고, 사용자 질문도 같은 방식으로 임베딩되어 가장 유사한 문서들이 검색된 뒤 에이전트가 최종 답을 만들어 냅니다.

#### 리스트 13.3 문서를 임베딩으로 변환하기

아래 코드(원문 OCR 보존)는 **LangChain** 라이브러리를 사용합니다. LangChain은 다양한 모델·데이터베이스·정보 검색 도구 등을 지원하며 AI 기반 시스템을 만들 수 있게 해 주는 라이브러리입니다.

```python
import os
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

_ = load_dotenv()                                   # 지정한 노드, 속성,
                                                    # 그리고 AI 모델을 이용해
if __name__ == "__main__":                          # 임베딩과 벡터 인덱스를 생성한다.
    vector_index = Neo4jVector.from_existing_graph(
        embedding=OpenAIEmbeddings(),
        url=os.environ['NEO4J_URL'],
        username=os.environ['NEO4J_USER'],
        password=os.environ['NEO4J_PWD'],
        database=os.environ['NEO4J_DB'],
```

```python
        index_name='embeddings',
        node_label="Page",
        text_node_properties=['text'],
        embedding_node_property='embedding')

    # 의미 유사도 검색:
    # 주어진 질문에 대해 상위 두 문서를 반환한다.
    q = "What is known about cyclotron research?"
    resp = vector_index.similarity_search_with_score(q, k=2)
    for r in resp:
        print(f"------\nScore: {r[1]}")
        print(r[0].page_content)
```

이 코드는 `Page` 노드에서 텍스트를 가져와, 지정한 임베딩 모델로 벡터화하고, 그 벡터를 다시 Neo4j 데이터베이스에 저장합니다. 질문이 들어오면 같은 모델로 그 질문을 임베딩하고, 가장 유사한 텍스트 조각들을 검색합니다. 축약된 출력은 다음과 같습니다.

#### 리스트 13.4 질문과 가장 유사한 상위 2개 텍스트 청크

```text
Score: 0.9180829524993896
text: Tuesday, January 31, 1939 (Cont'd)
The reproduction cost of the large cyclotron would involve about $30,000
for the magnet, $15,000 for the power supply, and $30,000 for the cyclotron
chamber, accessories, controls, etc. Lawrence is already thinking about the
next step. He believes that his present "large" machine will be duplicated
several other places: but if it is wholly successful he wants to go on to
build a hundred million volt machine

Score: 0.9157187938690186
text: Dr. R. J. Van de Graaff, Massachusetts Institute of Physics. Van de
Graaff gives WW the whole history and philosophy of his scientific career
to date. The physical universe being composed of particles, G. decided to
study particles
Although he speaks with complete modesty and with complete generosity
relative to the cyclotron development, he is inclined to think that his
type of generator promises to offer relative advantage ..
```

첫 번째 문서는 질문과 매우 관련이 높습니다. 두 번째로 높은 순위의 문서 역시 사이클로트론을 언급하긴 하지만, 판 데 그라프(Van de Graaff)가 자신의 (지금은 유명해진) 발전기를 옹호하는 일종의 세일즈 피치에 가깝습니다.

#### 13.4.2 벡터 기반 RAG의 한계 — 임베딩만으로는 부족하다

우리는 생짜 사전 학습 LLM을, 벡터 검색 기반 RAG 에이전트를 이용해 비공개 데이터에 대한 질문에 답할 수 있는 쓸모 있는 대화형 어시스턴트로 바꾸는 법을 살펴보았습니다. 그렇다면 다음은 무엇일까요? 이게 우리가 할 수 있는 최선일까요?

벡터 검색 형태의 RAG가 유용하긴 하지만, 몇 가지 도전 과제와 단점이 있습니다.

- **맥락 파편화로 인한 제한된 추론** — 일치하는 문서 조각들의 목록을 제공하는 방식은, 문서들을 서로 독립적으로 다루려는 경향을 이 과정에 암묵적으로 심어 넣습니다. 그 결과 문서들 사이, 그리고 문서 안에 언급된 개체(entity)들 사이의 더 복잡한 **다중 홉 관계(multihop relationship)**, 즉 여러 단계를 건너 이어지는 관계를 놓칠 수 있습니다. 이는 또한 우리의 청킹(chunking, 문서 쪼개기) 전략의 한계를 드러냅니다. 우리가 필요한 정보가 이전 조각에서 시작해 현재 조각으로 이어진다면 어떻게 될까요? 두 조각이 모두, 올바른 순서로, 높은 순위로 검색될까요? 단순한 청킹은 아마 잘 동작하지 않을 것입니다. 최소한 일부 문제라도 완화하려면 그 설계에 훨씬 더 많은 고민을 담아야 합니다.
- **확장성(scalability)** — 이 접근은 말뭉치가 커질수록 연산 비용이 커지며, 종종 우리를 더 효율적이지만 덜 정확한 **근사 검색(approximate search)** 알고리즘으로 내몰게 됩니다.
- **임베딩의 한계** — 문서의 의미를 단 하나의 조밀한 벡터로 인코딩하려는 시도는 과도한 단순화를 낳습니다. 중요한 세밀한 의미와 도메인 특유의 뉘앙스를 포착하지 못하는 것입니다. 또 다른 한계는 임베딩 모델의 학습 데이터셋에 존재할 수 있는 희소성(sparsity)입니다. 즉 특정 용어가 과소 대표되는 경우입니다. 이는 부정확한 임베딩 표현과 낮은 검색 정확도로 이어집니다. 마지막으로, 정적인(미리 계산된) 벡터에 의존하기 때문에 RAG는 새롭게 등장하거나 진화하는 지식과 명명법에 덜 유연하게 됩니다.
- **검색 결과의 잡음(noise)** — 벡터 검색은 느슨하게 관련되었거나 완전히 무관한 문서를 반환할 수 있으며, 이는 **주의 분산(distraction)** [3]으로 이어집니다. 특히 긴 맥락에서 잡음이 너무 많으면 모델을 혼란스럽게 만들어 출력 품질을 떨어뜨릴 수 있습니다. 핵심은 관련 정보의 밀도가 가능한 한 높은 맥락을 제공하는 것입니다.
- **검색 누락(misses in retrieval)** — 임베딩의 한계와 근사 벡터 유사도 검색의 사용이 잡음을 유발할 수 있는 것처럼, 정반대의 문제도 일으킬 수 있습니다. 즉 가장 관련성 높은 문서를 포함하지 못하는 것입니다. 필요한 사실을 모두 제공하지 못하면, 아무리 좋은 AI 모델이라도 완전히 올바른 답을 얻을 수 없습니다. 누락은 "이 데이터셋의 핵심 연구 주제는 무엇인가?" 같은 질문에서도 발생할 수 있습니다. 이때 답은 오해를 낳게 되는데, 벡터 유사도 검색은 기대되는 답을 생성하는 데 필요한 포괄적 데이터를 제공하기보다, 질문과 의미적으로 가장 유사한 조각들을 반환하기 때문입니다.

이 한계들을 예제로 설명해 봅시다. 우리 질문이 "로리첸(Lauritsen)은 사이클로트론 연구와 어떻게 관련되어 있는가?"라고 해 봅시다. 직관적으로는, 가장 유사한 문서라면 최소한 로리첸이라는 이름은 언급할 거라고 기대합니다. 틀렸습니다. 임베딩은 그렇게 동작하지 않습니다. 임베딩은 텍스트의 전체적인 의미를 인코딩하고, 그것을 압축된 요약으로 표현합니다. 가장 유사한 문서가 우리가 묻는 개체를 언급한다는 보장조차 할 수 없습니다! 질문과 문서의 의미적 유사성은 다른 언어적 의미나 패턴을 집어낼 수 있습니다. 실제로 이 질문을 임베딩하고 코사인 유사도(cosine similarity) 기준으로 가장 유사한 세 문서를 찾아보면, 그중 오직 하나만 "Lauritsen"(과 "cyclotron")을 포함하고, 나머지 두 개는 "cyclotron"만 언급합니다. 따라서 이 둘은 질문과 무관한데도, 첫 번째 문서와 거의 같은 유사도 점수를 가지고 있습니다. 표 13.1이 상위 세 문서를 개괄합니다.

**표 13.1** 질문 "로리첸은 사이클로트론 연구와 어떻게 관련되어 있는가?"와 가장 유사한 상위 세 문서, 그리고 각 문서가 "Lauritsen"과 "cyclotron"을 언급하는지 여부

| 문서 | 코사인 유사도 | Lauritsen | Cyclotron |
|---|---|---|---|
| Tuesday, January 3, 1939. Professor Karl Lark-Horovitz, Purdue University. The Van de Graaff machine at Purdue was built in two or three months and is now operating, producing 600 micro-amperes at 850 kilovolts. Thus as a neutron generator it is equivalent to several pounds of radium. The Purdue group has devised ... | 0.906 | True | True |
| Tuesday, May 2, 1939. Dr. Irving Langmuir, General Electric Company. L. very strongly favors continued support for Dr. Dorothy Wrinch. He bases this primarily on two considerations. First, and giving due recognition to the fact that W. is a difficult person whose scientific behavior is not always what it should be, L. considers it unquestionably true that W. has been responsible for stimulating ... | 0.903 | False | False |
| Dr. Dorothy M. Wrinch, April 3, 1939 (continued). In connection with the X-ray structure problem Langmuir obtained from Clowes some money which W. used for computing. She is not sure that she should have accepted funds from elsewhere, but WW assures her that ... | 0.902 | False | True |

이 한계들을 어떻게 극복할 수 있을까요? 이 책의 독자라면 우리의 답이 전혀 놀랍지 않을 것입니다. 바로 KG를 그림 속으로 끌어들이는 것입니다. RAG에 대한 그래프 기반 접근은 흔히 **그래프 RAG(Graph RAG)** 라고 불립니다. 벡터 기반 RAG와 관련된 도전 과제들을 완화하거나 심지어 완전히 해결하는 것 외에도, 이 접근에는 추가적인 이점이 있습니다. KG는 중앙 지식 저장소 역할을 하면서, 원본 텍스트뿐 아니라 다양한 문서 메타데이터와 신뢰도 높은 구조화 데이터 소스(테이블, CSV, 온톨로지 등)까지 통합합니다. 이들이 함께 어우러지면 훨씬 폭넓은 사례를 풀어낼 수 있습니다. KG는 지식을 사람이 이해할 수 있는 형태로 표현하기 때문에 최신 상태로 유지하기가 쉽습니다. 그리고 도메인 전문가가 기존 지식을 검증하거나 새 지식을 입력할 수 있어, AI 시스템의 출력 품질에 직접 영향을 줄 수 있습니다. 이렇게 높아진 투명성은 결국 사용자의 신뢰로 이어집니다.

#### 13.4.3 그래프 RAG — 지식 그래프를 끌어들이다

KG와 LLM 사이의 시너지 개발은 빠르게 진행되고 있습니다 [4]. 록펠러 아카이브 센터 KG는 **LLM으로 증강된 KG(LLM-augmented KG)** 의 좋은 예입니다. 우리는 OpenAI의 ChatGPT 모델 위에서 프롬프트 엔지니어링을 활용해 이 KG를 만들었는데, 이 모델은 개체와 관계를 추출했고, 나아가 모델 내부 지식을 이용해 파악된 정보를 보완하기까지 했습니다. 그 결과로 만들어진 KG는 **텍스트 속성 그래프(text-attributed graph)** 와 **텍스트 짝지음 그래프(text-paired graph)** 의 결합입니다 [5](그림 13.3 참조).

![그림 13.3 록펠러 아카이브 센터 KG는 텍스트 속성 그래프와 텍스트 짝지음 그래프를 결합합니다. 문서는 노드로 표현되며, 각 노드는 저자·문서 유형·날짜 같은 메타데이터를 나타내는 속성을 가집니다. 추출된 개체와 관계는 원본 문서까지 거슬러 추적할 수 있어서, AI 에이전트를 위한 KG 기반 문서 선택 도구를 설계할 수 있게 해 줍니다.](images/ko/figure-13-3-ko.png)

텍스트 속성 그래프란 노드와 관계가 텍스트 속성을 가진 그래프이고, 텍스트 짝지음 그래프란 노드·관계·그래프가 그것들의 후손에 해당하는 문서들과 연결되어 있는 그래프입니다. 우리 예제 KG의 설계는, 데이터와 지식 모델링의 여러 측면을 활용하는 그래프 RAG 시스템을 구현할 수 있게 해 줍니다.

- **메타데이터(metadata)** — 문서는 대개 발행일·유형·출처·저자 같은 메타데이터를 함께 가집니다. 이 모든 정보가 KG 안에 속성과 관계로 존재하며, 내용에 더해 맥락 검색 시스템을 설계하는 데 쓰일 수 있습니다. 예를 들어 문서들의 그래프에서 **커뮤니티(community)**, 즉 서로 밀접하게 연결된 문서 집단을 식별할 수 있고, 각 커뮤니티를 요약본으로 대표하거나 가장 최근 문서(예를 들어 어떤 법이나 규정의 최신 버전)로 대표할 수 있습니다.
- **KG 검색기(KG retriever)** — KG는 훌륭한 맥락 자원입니다. 노드와 유형이 지정된 관계는 LLM이 답을 생성하는 데 쓸 수 있는, 압축되고 정확하며 최신인 정보를 제공합니다. 모델은 사용자의 질문과 KG의 스키마를 받아, 필요한 정보를 검색하는 Cypher 쿼리를 생성할 수 있습니다. 또한 KG 검색기 도구는 사용자가 묻는 개체들을 입력으로 받아, 그 개체들을 (예를 들어 모든 최단 경로를 통해) 잇는 부분 그래프(subgraph)를 반환할 수 있습니다. 그러면 그 데이터 중 어느 부분이 관련 있는지는 최종 LLM이 판단하게 됩니다.
- **KG 강화 문서 검색기(KG-enhanced document retriever)** — 우리의 KG가 텍스트 짝지음 그래프라면, 이것을 문서 검색기로 쓸 수 있으며, 이는 벡터 검색 기반 검색기보다 더 정확합니다. 예를 들어 사용자가 물은 개체를 모두 언급하는 문서만 검색하도록 하여, 앞서 논의한 벡터 검색의 실패 중 하나를 제거할 수 있습니다. 또는 KG가 제공할 수 있는 것보다 어떤 관계에 대해 더 상세한 정보가 필요할 때, 그 관계를 언급하는 문서만 검색하여 주의 분산 현상을 피할 수 있습니다.
- **결합 검색(combined retrieval)** — 때로는 질문이 여러 데이터 소스에 걸쳐 있습니다. 이런 경우, 질문을 두 부분으로 나누는 AI 에이전트를 설계할 수 있습니다. 하나는 KG 정보를 검색하고(KG 검색기), 다른 하나는 이 정보를 이용해 가장 관련성 높은 문서를 찾습니다. 최종 답을 생성하기 전에 두 맥락이 병합됩니다. 예를 들어 이런 질문을 생각해 보세요. "범죄 조직 $\mathbf{X}?^{\mathfrak{n}}$ 의 우두머리가 올해 한 거래는 무엇인가?"(원문의 이 수식 표기는 OCR 과정에서 뭉개진 것으로, 실제로는 "범죄 조직 X의 우두머리"를 뜻합니다.) 금융 문서에 그 사람의 직함이 "X의 우두머리"라고 적혀 있을 가능성은 거의 없습니다. 그래서 우리는 법 집행 기관이 구축한 KG에서 그 사람의 이름을 추출한 다음, 그 이름을 이용해 금융 문서 데이터베이스를 검색해야 합니다.

이제 그래프 RAG가 무엇인지 이해했으니, 록펠러 아카이브 센터 KG 위에 간단한 에이전트를 하나 구현해 봅시다(그림 13.4 참조). 우리의 그래프 RAG 에이전트는 세 가지 도구를 손에 쥐게 됩니다. KG 검색기, KG 강화 문서 검색기, 그리고 다른 도구들이 값진 맥락을 반환하지 못했을 때를 대비한 백업용 **의미 검색기(벡터 검색 도구)** 입니다.

![그림 13.4 외부 데이터 소스(KG와 벡터 데이터베이스)에 근거를 둔 KG 기반 RAG 에이전트](images/ko/figure-13-4-ko.png)

전체 코드는 책의 깃허브 저장소에 있습니다. 여기서는 KG 강화 문서 검색기에 초점을 맞춥니다. 이 도구는 두 개체 사이의 특정 관계가 논의되는 문서를 식별하는 사례를 위해, 파라미터를 받는 도구로 구현되어 있습니다. 이 도구 덕분에 "인물 X는 인물 Y에 대해 무엇이라고 말했는가?" 같은 질문에 답할 수 있게 됩니다.

#### 리스트 13.5 그래프 RAG 에이전트용 KG 강화 문서 검색기 도구

아래 코드(원문 OCR 보존)에서 도구는 질문에 언급된 두 개체, 그 개체들의 클래스(예를 들어 Person), 그리고 관계 유형을 입력으로 받습니다. 이 값들은 사용자의 질문을 바탕으로 AI 에이전트가 제공합니다.

```python
from pydantic import BaseModel, Field
from langchain_community.graphs import Neo4jGraph

# 미리 준비된(precanned) 문서 선택 쿼리
RE_SELECTOR_QUERY = """MATCH (p:Page)-[:MENTIONS_ENTITY]->(m1:Ent...
WHERE e1.name = "{e1}" ...
RETURN DISTINCT p.id AS id, p.text AS text
"""

# Neo4j 그래프 데이터베이스 연결을 초기화한다
graph = Neo4jGraph(
    url=os.environ['NEO4J_URL'],
    username=os.environ['NEO4J_USER'],
    password=os.environ['NEO4J_PWD'],
    database=os.environ['NEO4J_DB']
)


# 새 도구의 입력 스키마(함수 인자) 정의
class REDiarySelectorInput(BaseModel):
    entity_source: str = Field(description="Source entity of the relationship as mentioned in the question.")
    entity_source_class: str = Field(description=
        "Class of the source entity of the relationship. "
        "Available option is only one, 'Person'.")
    entity_target: str = Field(description="Target entity of the relationship as mentioned in the question.")
    entity_target_class: str = Field(description=
        "Class of the target entity of the relationship. "
        "Available options are Person, Organization, Occupation and Title.")
    relationship: str = Field(description=
        "Relationship class between source and target entity. "
        "Available options: TALKED_ABOUT, TALKED_WITH, WORKS_WITH, WORKS_ON, HAS_TITLE")


# KG 강화 문서 검색기 함수
def kg_doc_selector(entity_source: str, entity_source_class: str,
                    entity_target: str, entity_target_class: str, relationship: str) -> List[AnyStr]:
    query = RE_SELECTOR_QUERY.format(e1=entity_source,
        e1_class=entity_source_class,
        e2=entity_target, e2_class=entity_target_class,
        rel_class=relationship)
    print(f"kg_doc_selector's query:\n{query}\n")
```

```python
    try:
        res = graph.query(query)
        print(f"kg_doc_selector found {len(res)} matching documents")
    except Exception as e:
        print(f"Cypher execution exception: {e}")
        return []
    return [x['text'] for x in res[:3]]
```

이 도구는 질문에 언급된 두 개체, 그 클래스(예를 들어 Person), 그리고 관계 유형을 입력으로 받습니다. 이 값들은 AI 에이전트가 사용자의 질문을 바탕으로 제공합니다. 핵심 선택 함수는 이 값들을 이용해, 미리 준비된 Cypher 쿼리를 완성합니다. 이 쿼리는 Neo4j 데이터베이스에 대해 실행되고, 그 결과 문서들이 에이전트에게 반환됩니다.

> **참고** 문서 검색기 Cypher 쿼리를 질문에 따라 자동으로 생성하는, 더 범용적인 도구를 설계할 수도 있습니다. 하지만 그렇게 하면 Cypher 쿼리가 복잡할 때 또 하나의 실패 지점을 들여오게 됩니다. 바로 이 때문에, 운영 중인 수많은 그래프 RAG 시스템은 다양한 KG 관련 도구를 담고 있으며, 그중 상당수는 자주 반복되는 유형의 질문을 위해 미리 준비해 둔 Cypher 쿼리에 기반합니다.

#### 13.4.4 추론 에이전트 — ReAct로 생각하고 행동하기

이제 이 모든 것을 하나의 에이전트로 통합할 시간입니다. LangChain 라이브러리는 이 작업을 아주 쉽게 만들어 주는, 미리 만들어진 에이전트 몇 가지를 제공합니다. 우리에게는 명확한 실행 순서가 없는 여러 도구가 있으므로, **ReAct(Reason and Act, 추론과 행동)** 에이전트 [6]를 사용합니다. ReAct는 복잡한 환경에서 문제 해결 능력을 높이기 위해 추론 능력과 행동 능력을 통합한 프레임워크입니다. ReAct 프레임워크는 어떤 작업을 할지 반복적으로 추론하고, 행동하고, 그 결과를 관찰하는 동적인 피드백 루프를 돌면서, 실시간 결과를 바탕으로 자신의 접근을 다듬어 갑니다.

에이전트는 우리가 제공한 도구들의 제약 안에서 원래 질문을 받아, 다음에 할 작업(실행할 도구)을 계획하고, 그것을 실행한 뒤 그 결과에 대해 추론합니다. 얻은 정보가 만족스럽지 않으면 또 다른 도구를 써서 행동합니다. 원래 질문에 답하기에 충분하다고 판단되는 맥락 정보를 얻으면, 루프를 끝냅니다. 다음의 축약된 코드가 이런 에이전트를 정의합니다.

#### 리스트 13.6 그래프 RAG 방식을 구현한 ReAct 에이전트

```python
from langchain.tools import StructuredTool
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from tools import REDiarySelectorTool, kg_doc_selector, REDiarySelectorInput
from definitions import KG_SCHEMA
```

아래는 모든 도구 정의를 하나의 리스트에 모으는 부분입니다(원문 OCR 보존). 각 도구에는 적절한 이름과 설명이 붙어 있어, 모델이 어떤 상황에서 어떤 도구를 호출할지 판단하도록 돕습니다.

```python
# 모든 도구 정의를 하나의 리스트에 모은다
tools = [
    StructuredTool.from_function(
        func=kg_doc_selector,
        name="KG-based-document-selector",
        args_schema=REDiarySelectorInput,
        # 도구 설명: 두 개체 간 상호작용에 대한 상세 정보를 묻는 질문일 때
        # 문서(일기 항목) 검색에 사용한다. ... 전체 KG 스키마:
        description=f"Use it for document (diary entries) retrieval when the question asks for detailed information regarding interaction between two entities ... Full KG schema:\n{KG_SCHEMA}"
    ),
    <KG RETRIEVER>,      # 원본 텍스트가 필요 없는 질문을 위한 KG 검색기 구조화 도구
    <VECTOR SEARCH>,     # 벡터 검색 기반 검색 도구(백업)
]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = hub.pull("hwchase17/structured-chat-agent")
agent = create_structured_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5,
    return_intermediate_steps=True, verbose=True)

# 모델, 에이전트의 프롬프트, 도구들을 함께 묶어
# 에이전트와 그 실행기를 정의한다
question = "What did August Krogh say about Lawrence Irving?"
response = agent_executor.invoke({"input": question})
```

각 도구의 설명(`description`)에는 KG 스키마가 담겨 있어, 에이전트가 어떤 도구를 쓸지 판단할 수 있게 해 줍니다. 구조화된 ReAct 에이전트는 자신의 도구에 여러 개의 입력 파라미터를 전달하는 것을 지원합니다. 도구들은 모델이 어떤 상황에서 어떤 도구를 호출할지 판단하도록 돕는, 적절히 선택된 이름과 설명을 가지고 있습니다. 에이전트를 설계할 때 이 설명에 주의를 기울이세요. 잘만 쓰면 시스템의 안정성과 예측 가능성을 크게 높일 수 있습니다. 시스템이 기대만큼 일관되게 동작하지 않는다면, LangChain이 여러 에이전트와 도구에서 제공하는 기본 프롬프트를 주저 없이 덮어써 버리세요. 그리고 언제나 테스트하고, 테스트하고, 또 테스트하세요. 같은 설정과 같은 입력 질문으로도 반복해서 말입니다. 이 과정이 애플리케이션을 개선할 통찰을 드러내 줄 수 있습니다.

이 예제는 KG 기반 그래프 RAG 시스템을 어떻게 만드는지 보여 주는 하나의 예시일 뿐이며, 이것이 운영 수준의 시스템이라고 주장하는 것은 아닙니다. 우리는 여러 가지를 개선할 수 있습니다. 예를 들어 벡터 검색 도구를 위한 문서 재순위(re-ranking) 전략을 개발하거나, KG 검색기에 Cypher 자가 교정(self-correction) 루프를 더하거나, 다른 전형적인 사용자 질문을 지원하는 도구를 추가하는 것입니다. 더 자세한 내용은 [7]을 참고하세요.

#### 13.4.5 우리 KG와 대화해 보자 — 실전 테스트

이제 그래프 RAG 에이전트가 준비되었으니, 실제로 시험해 봅시다. 여러분이 도로시 M. 링크(Dorothy M. Wrinch, 1894\~1976)에 대한 정보를 얻으려고 록펠러 아카이브 센터를 방문한 연구자라고 상상해 보세요. 그녀는 수학자이자 생화학 이론가였습니다. 그녀의 연구는 꽤 잘 알려져 있지만, 여러분은 당대의 동료들이 그녀를 어떻게 인식했는지 알고 싶습니다. KG에서 그녀를 찾아, 들어오는 `TALKED_ABOUT` 관계들을 살펴보고, 그 관계들이 추출된 원본 문서를 찾아낼 때까지 파고들어 읽어 볼 수도 있습니다. 아니면, 여러분의 새 에이전트를 쓸 수도 있습니다. 그림 13.5는 "동료 연구자들은 도로시 M. 링크에 대해 무엇이라고 말했는가?"라고 물었을 때 무슨 일이 일어나는지 보여 줍니다.

> **참고** 여러분이 직접 시도해 보면, 사용하는 OpenAI GPT 모델 버전이나 다른 모델 제공자에 따라 출력이 다를 수 있습니다.

![그림 13.5 우리 그래프 RAG 에이전트의 내부 단계 예시. 에이전트는 하나의 생각(thought)에서 시작해, KG 검색기를 사용해 도로시 M. 링크에 대해 이야기한 사람들의 목록을 얻습니다. 그런 다음 각 사람에 대해 KG 강화 문서 검색기를 사용해 관련 문서를 식별합니다. 마지막으로, 에이전트는 이 모든 것을 맥락으로 삼아 최종 답을 생성합니다.](images/ko/figure-13-5-ko.png)

에이전트는 자신이 가진 모든 도구를 고려합니다. 그리고 가장 좋은 행동 방침이, 먼저 KG에서 링크에 대해 이야기한 사람들의 목록을 가져온 다음, 특화된 KG 강화 문서 검색기를 사용해 무슨 말이 오갔는지 세부 정보를 담은 관련 문서를 얻는 것임을 정확히 파악합니다. 에이전트는 총 네 번의 생각(Thought)/행동(Action)/관찰(Observation) 단계를 거칩니다. 검색된 모든 맥락을 바탕으로 생성된 최종 답은 나무랄 데가 없습니다. 요점을 곧장 짚고, 사실에 충실하며, 에이전트에게 제공된 비공개 문서에 분명하게 근거하고 있습니다.

표 13.2는 두 개의 질문을 더 보여 줍니다. 첫 번째는 도로시 M. 링크에 관한 질문의 변형인데, 이번에는 두 명의 특정 인물에 대해 묻기 때문에 에이전트가 KG 강화 문서 검색기 도구를 곧바로 사용할 수 있습니다. 만약 같은 질문을 생짜 LLM에게 던졌다면, "어거스트 크로그(August Krogh)가 로런스 어빙(Lawrence Irving)에 대해 무엇이라고 말했는지에 관해서는 구체적인 정보가 없습니다" 같은 답을 얻거나, 아니면 LLM이 답을 지어냈을 것입니다. 하지만 우리의 그래프 RAG 시스템에서는 그들의 관계에 대한 정확한 통찰을 얻습니다.

**표 13.2** 그래프 RAG 질의응답 예시

| 질문 | 그래프 RAG 답변 |
|---|---|
| "어거스트 크로그는 로런스 어빙에 대해 무엇이라고 말했는가?" | 어거스트 크로그는 노르웨이에서 이루어진 심해 잠수 동물의 호흡 생리에 관한 숄란데르(Scholander)의 연구에 대해 열의를 가지고 이야기했으며, 숄란데르와 로런스 어빙을 한자리에 모으는 일의 중요성을 강조했다. 크로그는 어빙이 훌륭한 연구를 해 왔고, 노르웨이에서는 구할 수 없는 미국의 시설을 숄란데르에게 제공해 줄 수 있다고 언급했다. 그들은 메인주 부스베이 하버(Boothbay Harbor)에서, 그리고 어쩌면 플로리다의 수족관에 있는 돌고래들과 함께 연구할 가능성을 탐색했다. |
| "하버드 대학교와 존스 홉킨스 대학교 사이에 공유된 연구 주제가 있는가?" | 하버드 대학교와 존스 홉킨스 대학교 사이에 공유된 연구 주제는 천문학과 기후학이다. |

두 번째 질문에는 구조화된 KG 내용만으로 충분히 답할 수 있습니다. KG 검색기 도구가 올바른 Cypher 쿼리(조금 서툴긴 해도 올바른)를 생성하므로, 답은 간단하고 정확합니다. 반면 이 질문을 순전히 벡터 검색만으로 답하려 했다면, 방대한 문서 집합을 맥락으로 제공하고 그 안에 완전한 답이 들어 있기를 바라야 했을 것입니다. 왜냐하면 이것은 여러 문서에 걸쳐 있는 **집계 질문(aggregate question)** 의 예이기 때문입니다. 어떤 일기 항목은 하버드 사람들이 한 연구를 논하고, 다른 항목은 존스 홉킨스의 주제를 서술하지만, 하버드와 존스 홉킨스가 무엇을 공통으로 가지는지 직접 논하는 항목은 하나도 없습니다. KG는 여러 문서에 걸쳐 점들을 잇는 데 탁월하며, 부수적인 이점으로 주의 분산 현상과 환각의 위험을 줄여 주고, 예측을 더 빠르고 저렴하게(더 적은 맥락 데이터로) 만들어 줍니다.

우리는 RAG 시스템의 정확성·신뢰성·안정성을 높이기 위해 다양한 접근을 쓸 수 있습니다. 예를 들어 자가 교정 루프를 더해, 모델이 먼저 Cypher 쿼리를 생성한 다음 LLM에게 그것을 재검토하고 필요하면 후속 단계에서 쿼리를 교정하도록 할 수 있습니다. 또는 초기 맥락 선택 이후에 더 발전된 문서 재순위 단계를 더해, 그 관련성을 높이고 크기를 제한할 수 있습니다. 다음 장에서는 Cypher 생성이라는 주제를 더 깊이 파고들 것입니다.

---

#### 요약 — 이 장의 핵심 정리

- 모든 AI 에이전트의 핵심에는, 에이전트의 두뇌 역할을 하는 LLM 모델, 그 모델을 안내하는 프롬프트, 그리고 에이전트가 바깥세상과 상호작용할 수 있게 해 주는 도구 집합의 조합이 자리합니다.
- 검색 증강 생성(RAG)은 생성 모델과 정보 검색을 결합하여, AI 에이전트 같은 지능형 시스템을 만드는 프레임워크입니다. 그렇게 함으로써 환각·최신성·투명성·데이터 프라이버시 같은 LLM 고유의 문제들을 다룹니다.
- 벡터 기반 RAG 시스템은 여러 단점을 겪습니다. 제한된 추론 능력, 확장성 문제, 그리고 잡음과 관련 정보 누락을 포함한 검색의 부정확성이 그것입니다.
- 그래프 RAG는 KG를 LLM과 통합하여, KG 안의 구조화된 관계적 다중 홉 패턴을 활용함으로써 추론 능력과 정보 검색의 정밀도를 강화합니다. 또한 KG는 질의응답 과정에 더 많은 통제와 투명성을 제공합니다.
- KG를 RAG 시스템에 통합하는 방법은 그래프 설계에 따라 결정됩니다. 가장 유용한 KG는 텍스트 속성 그래프와 텍스트 짝지음 그래프를 결합한 것으로, 이를 통해 잘 정제된 구조화 지식과, 문서 및 그 메타데이터를 함께 활용할 수 있습니다.

---

## 핵심 용어 해설

| 용어 (원어) | 한 줄 정의 |
|---|---|
| 지식 그래프 (Knowledge Graph, KG) | 개념(개체)과 그 사이의 관계를 그래프 구조로 정리한, 검증 가능한 사실 저장소 |
| 거대 언어 모델 (Large Language Model, LLM) | 방대한 텍스트로 학습해 사람처럼 언어를 이해하고 생성하는 인공지능 모델 |
| 검색 증강 생성 (Retrieval-Augmented Generation, RAG) | 외부 소스에서 관련 정보를 검색해 그 맥락 위에서 LLM이 답을 생성하게 하는 기법 |
| 그래프 RAG (Graph RAG) | KG를 맥락 소스로 활용하는 RAG 접근으로, 벡터 검색의 한계를 보완함 |
| 환각 (hallucination) | LLM이 그럴듯하지만 실제로는 틀린 사실을 지어내는 현상 |
| 지식 컷오프 (knowledge cutoff) | LLM 학습 데이터의 시점 한계로, 그 이후의 최신 정보를 알지 못함 |
| AI 에이전트 (AI agent) | 환경과 상호작용하며 자율적으로 판단·행동해 복잡한 작업을 수행하는 시스템 |
| 프롬프트 엔지니어링 (prompt engineering) | 모델에게 던지는 지시문을 잘 설계해 원하는 출력을 이끌어 내는 기술 |
| 임베딩 (embedding) | 텍스트의 의미를 담아낸 고정 길이의 조밀한 벡터 표현 |
| 벡터 데이터베이스 (vector database) | 임베딩 벡터를 저장하고 유사도 기반으로 검색하도록 색인한 데이터베이스 |
| 코사인 유사도 (cosine similarity) | 두 벡터의 방향이 얼마나 비슷한지를 재는 유사도 척도 |
| 청킹 (chunking) | 긴 문서를 임베딩·검색을 위해 작은 조각으로 쪼개는 작업 |
| 다중 홉 관계 (multihop relationship) | 여러 단계를 건너 이어지는, 개체 간의 간접적 연결 관계 |
| 주의 분산 (distraction) | 무관한 문서가 맥락에 섞여 모델의 출력 품질을 떨어뜨리는 현상 |
| 텍스트 속성 그래프 (text-attributed graph) | 노드와 관계가 텍스트 속성을 가진 그래프 |
| 텍스트 짝지음 그래프 (text-paired graph) | 노드·관계·그래프가 원본 문서와 연결된 그래프 |
| KG 검색기 (KG retriever) | 질문과 스키마로 Cypher 쿼리를 만들거나 부분 그래프를 반환해 맥락을 제공하는 도구 |
| KG 강화 문서 검색기 (KG-enhanced document retriever) | KG 관계를 활용해 정확한 문서만 골라내는 검색 도구 |
| ReAct (Reason and Act) | 추론·행동·관찰을 반복하는 피드백 루프로 문제를 푸는 에이전트 프레임워크 |
| 근거화 (grounding) | 모델의 답 생성 범위를 제공된 맥락으로 제한해 환각을 줄이는 것 |
| 사람을 루프 안에 두기 (human in the loop) | 사람이 검증·감독에 개입해 시스템을 보강하는 설계 원칙 |
| LLMOps (LLM operations) | LLM 기반 시스템을 운영·관리·모니터링하는 기술과 관행 |
| 자연어 처리 (Natural Language Processing, NLP) | 컴퓨터가 사람의 언어를 이해하고 다루게 하는 분야 |
