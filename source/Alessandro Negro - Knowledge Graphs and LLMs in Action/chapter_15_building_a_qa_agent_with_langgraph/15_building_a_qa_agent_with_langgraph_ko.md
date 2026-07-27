---
lang: ko
format:
  html:
    toc: true
    embed-resources: true
    theme: cosmo
---

# LangGraph로 QA 에이전트 구축하기

### 이 장에서 다루는 내용


 전문가 모방 접근법 구현

질의응답을 통한 조사 구현

시스템 조정 및 개선

이 장에서는 LLM을 사용하여 지식 그래프를 질의하는 실용적인 애플리케이션을 만듭니다. 14장에서 살펴본 개념과 기법을 종합하고, 그림 15.1의 정신 모델로 이를 보여 주면서 통합 솔루션을 구축하는 방법을 설명합니다. LangGraph를 오케스트레이션 프레임워크로 사용하여 각 단계를 매끄러운 파이프라인으로 결합하는 방법을 보여 줍니다. 이 시스템을 접근하기 쉽고 사용자 친화적으로 만들기 위해 Streamlit을 프런트엔드 인터페이스로 사용합니다. 이 책의 코드 저장소에는 전체 구현과 구성 파일이 포함되어 있으므로, 개념을 따라가며 진행하는 동안 코드를 쉽게 참조할 수 있습니다.

![](images/ko/figure-15-1-ko.png)  
그림 15.1 14장에서 소개한 시스템 아키텍처의 개요입니다. 사용자 입력(질문과 사용자 선택) 및 출력(시각화와 요약)을 처리하기 위해 Streamlit을 사용하여 이를 구현하며, LangGraph가 핵심 파이프라인을 오케스트레이션합니다.

### 15.1 LangGraph 파이프라인 구축


LangGraph는 LLM을 기반으로 하는 상태 보존형(stateful) 다중 행위자 애플리케이션을 구축하도록 설계된 혁신적인 라이브러리입니다. 이 프레임워크는 복잡한 추론과 의사결정 과정을 포함하는 워크플로를 오케스트레이션하는 데 특히 적합하며, 이러한 과정은 우리의 KG 질의 파이프라인의 핵심입니다.

LangGraph가 이러한 개념을 실제로 어떻게 구현하는지 더 잘 이해하기 위해, 익숙한 예시인 기본 검색 증강 생성(retrieval-augmented generation, RAG) 시스템에서 시작해 보겠습니다. 이 시스템은 관련 문서를 검색하고 질문에 대한 답변을 생성합니다. 우리의 전문가 모방 아키텍처보다 단순하지만, 이 예시는 우리가 바탕으로 삼을 핵심 원리를 보여 줍니다.

그림 15.2에 나타난 것처럼, 워크플로는 문서 검색과 답변 생성이라는 두 가지 주요 작업으로 구성됩니다. LangGraph를 독특하게 만드는 것은 컴포넌트 간 통신 방식입니다. 즉, 컴포넌트 간에 데이터를 직접 전달하는 대신, 각 컴포넌트는 공유 상태와 상호작용합니다. 이는 각 에이전트가 이전 작업을 읽고 자신의 결과를 추가할 수 있는 화이트보드와 유사합니다.

상태는 사용자의 질문으로 시작됩니다. 첫 번째 에이전트는 문서 검색을 담당하며, 상태에서 질문을 읽고 자신이 찾은 관련 문서를 추가합니다. 그런 다음 두 번째 에이전트는 원래 질문과 검색된 문서 모두에 접근하여 적절한 답변을 생성할 수 있으며, 이 답변 역시 상태에 추가됩니다.

아키텍처의 핵심에서 LangGraph는 이러한 워크플로를 방향 그래프로 구현하며, 각 노드는 고유한 에이전트 함수를 나타냅니다. 이러한 에이전트 함수는 전역 상태와 상호작용함으로써 자신의 책임을 수행합니다. 즉, 관련 데이터를 읽고 실행 후 자신의 결과로 상태를 갱신합니다. 그래프의 간선은 실행 흐름을 결정하며, 다음에 어떤 노드가 실행되어야 하는지를 지정합니다. 중요한 점은 LangGraph가 동적 간선 해석(dynamic edge resolution)을 지원하여, 임의로 복잡한 로직에 따라 워크플로가 분기될 수 있게 한다는 것입니다. 예를 들어, 이전 에이전트의 출력에 기반하여 다음에 실행할 노드를 선택할 수 있으므로, 워크플로 설계에서 유연성과 적응성을 보장합니다.

![](images/ko/figure-15-2-ko.png)  
그림 15.2 LangGraph에서 에이전트 함수 간의 상태 기반 통신입니다. 에이전트들은 진화하는 상태 객체를 통해 통신하면서도 서로 분리된 상태를 유지합니다. 각 에이전트 함수는 전역 상태를 독립적으로 수신하고 갱신합니다.

이 접근 방식은 워크플로 전반에서 시스템이 일관된 상태를 유지하도록 보장하는 동시에, AI 시스템 설계자가 다양한 애플리케이션을 만들 수 있게 합니다. 여기에는 LLM이 여러 컴포넌트 중 하나에 불과한 라우터 시스템부터, LLM이 자신의 실행 경로를 결정하고 형성할 수 있는 완전 자율 시스템까지 포함됩니다. 이러한 흐름 제어의 유연성 덕분에 LangGraph는 의도 탐지, 스키마 추출, 질의 생성과 같은 여러 전문화된 단계를 조정해야 하는 우리의 전문가 모방 아키텍처를 구현하는 데 이상적입니다.

#### 15.1.1 시스템 아키텍처 개요


LangGraph의 역량이 명확해졌으므로, 이제 전문가 모방 접근법이 실행 가능한 워크플로로 어떻게 변환되는지 살펴보겠습니다. 그림 15.3은 초기 사용자 입력 처리부터 의도 탐지, 질의 생성, 결과 제시에 이르는 우리 KG 질의 시스템의 핵심 컴포넌트를 보여 주며, 각 노드는 개별 에이전트 기능을 나타냅니다. 특히 이 컴포넌트 구조는 LangGraph의 에이전트/상태 아키텍처에 자연스럽게 대응되며, 여기서 각 처리 단계는 워크플로 상태를 수신하고 갱신하는 에이전트 기능으로 구현될 수 있습니다.

![](images/ko/figure-15-3-ko.png)  
그림 15.3 KG 질의 파이프라인의 LangGraph 구현. 실선 화살표는 의도 탐지에서 스키마 추출과 질의 실행으로 이어지는 주요 흐름을 보여 주며, 점선 화살표는 질의 실행 결과에 기반한 조건부 경로를 나타냅니다. 이 방향 그래프 (directed graph) 구조는 우리 전문가 모방 접근법의 각 컴포넌트를 LangGraph 에이전트 기능에 직접 대응시킵니다.

LangGraph 워크플로는 우리 백엔드 시스템의 중심이지만, 여러 지원 컴포넌트와 통합되며 프론트엔드 애플리케이션과 인터페이스합니다. 그림 15.4는 더 넓은 아키텍처 관점을 보여 줍니다. 백엔드의 핵심에는 LangGraph 워크플로가 위치하며, 구성 제공자로부터 프롬프트와 구성을 소비하는 동시에 스키마 제공자를 통해 그래프 스키마 정보에 동적으로 접근합니다. 이러한 지원 컴포넌트는 중요한 설정 작업을 처리합니다. 구성 제공자는 프롬프트 템플릿과 시스템 설정을 관리하고, 스키마 제공자는 LLM 소비를 위해 데이터베이스 스키마를 추출하고 형식화합니다.

질문 처리 인터페이스는 핵심 파이프라인과 프론트엔드 애플리케이션 사이의 다리 역할을 합니다. 이 인터페이스는 LangGraph 워크플로를 이벤트 스트림 (event stream)으로 노출하여 프론트엔드가 파이프라인 진행 상황을 실시간으로 추적할 수 있게 합니다. 이 인터페이스는 들어오는 질문을 처리하고, 이를 워크플로에 통과시키며, 상태 업데이트와 최종 응답을 사용자 인터페이스로 다시 스트리밍합니다.

이 아키텍처는 대화형 AI 시스템에 필요한 유연성을 유지하면서도 관심사를 명확히 분리합니다. 각 컴포넌트에는 명확한 책임이 있습니다. LangGraph는 핵심 질의응답 로직을 처리하고, 제공자는 구성 및 스키마 접근을 관리하며, 처리 인터페이스는 프론트엔드 통신을 처리합니다.

다음 절에서는 이러한 아키텍처 요소 각각을 살펴보되, 구성 관리와 스키마 변환 서비스를 제공하는 지원 컴포넌트부터 시작하겠습니다. 그런 다음 파이프라인 단계 간의 효과적인 통신을 가능하게 하는 상태 관리 설계를 살펴보고, 이어서 우리 질의응답 시스템의 핵심을 형성하는 개별 파이프라인 에이전트의 구현을 탐구하겠습니다. 마지막으로, 파이프라인 통합 계층이 이러한 요소들을 프론트엔드 애플리케이션과 상호작용할 수 있는 응집력 있는 시스템으로 어떻게 결합하는지 살펴보겠습니다.

![](images/ko/figure-15-4-ko.png)  
그림 15.4 LangGraph 파이프라인이 지원 컴포넌트와 통합되는 방식을 보여 주는 백엔드 아키텍처. 구성 제공자는 프롬프트와 설정을 관리하며, 스키마 제공자는 데이터베이스 스키마 접근을 처리합니다. 질문 처리 인터페이스는 이벤트 기반 API를 통해 핵심 파이프라인과 프런트엔드 애플리케이션을 연결합니다.

#### 15.1.2 파이프라인 컴포넌트 구성하기

구성 세부 사항을 살펴보기 전에, 구성 컴포넌트가 우리 시스템 아키텍처에 어떻게 들어맞는지 살펴보겠습니다(그림 15.5 참조). 구성 컴포넌트는 우리의 질의응답 시스템이 의존하는 텍스트 요소, 주로 프롬프트 템플릿과 KG 주석을 저장하는 중앙 집중식 저장소 역할을 합니다. 이러한 관심사의 분리는 종종 길이가 긴 텍스트 요소들이 주요 구현을 어지럽히지 않도록 하여, 깔끔하고 유지보수 가능한 코드를 유지하는 데 도움이 됩니다.

우리 템플릿은 Jinja2 템플릿 언어를 사용하므로 런타임에 동적 콘텐츠를 생성할 수 있습니다. 템플릿 정의를 구성 안에 분리함으로써, 템플릿 작성과 템플릿 렌더링 사이에 명확한 경계를 만듭니다. 그러면 주요 코드는 깔끔한 인터페이스를 통해 이러한 템플릿과 상호작용할 수 있으며, 내부적인 템플릿 구성 과정은 알 필요가 없습니다.

![](images/ko/figure-15-5-ko.png)  
그림 15.5 구성 제공자 컴포넌트를 강조한 시스템 아키텍처 다이어그램. 제공자는 LangGraph 에이전트가 사용자 질문을 처리하는 데 필요한 시스템 구성과 프롬프트 템플릿을 관리합니다.

이 접근 방식은 여러 가지 장점을 제공합니다. 첫째, 모든 조정을 핵심 로직을 건드리지 않고 한곳에서 수행할 수 있으므로 KG 설명과 프롬프트 템플릿을 조정하기가 단순해집니다. 둘째, 향후 확장을 위한 구조화된 기반을 제공합니다. 모든 텍스트 기반 리소스가 한곳에 조직되어 있으면 새로운 처리 단계를 추가하거나 기존 단계를 수정하는 일이 더 간단해집니다. 마지막으로, 전용 구성 컴포넌트를 두면 프롬프트와 주석의 서로 다른 버전을 더 쉽게 관리할 수 있는데, 이는 시스템의 개발과 개선 과정에서 특히 가치가 있습니다. 다음 목록은 우리 구성 구조의 예를 보여 줍니다.

#### 목록 15.1 구성 파일 예시

notes: >   
- 모든 POINTS 속성은 Neo4j Points입니다(\`point.distance()   
및 유사 함수가 이들에 대해 작동합니다)   
- ANPRCameraEvent를 Vehicle과 ANPRCamera 둘 다에   
연결해야 하는 경우가 아니면 확장하지 마십시오   
- 이전 범죄자 또는 알려진 범죄자는   
해당 노드가 범죄와 연결되어 있다는 사실로 정의됩니다   
examples:   
question: 2025년 3월 14일에 발생한 범죄   
answer: MATCH (c:Crime) WHERE c.date starts with "2025-03-14"   
reasoning: >-   
해당 날짜에 발생한 범죄를 찾기 위해, 우리는 범죄 노드의   
<b>date</b> 속성을 활용합니다.   
이 속성은 ISO 문자열 형식이므로,   
접두사 "2025-03-14"를 사용하여 그날 발생한 모든 범죄를 얻을 수 있습니다.   
순회가 없으므로 경로는 반환되지 않습니다

[...]   
question: 20세에서 22세 사이의 남성 알려진 범죄자 한 명을 반환   
answer: >-   
MATCH path = (person:Person)   
-[committed:COMMITTED]->(crime:Crime)   
WHERE (person.sex CONTAINS 'MALE' AND   
person.age >= 20 AND person.age <= 22)   
RETURN path LIMIT 1   
prompts:   
text\_to\_cypher:   
system: >-   
제공된 스키마 정의를 기반으로,   
사용자 질문에 답하는 Neo4j 그래프   
데이터베이스용 Cypher 쿼리를 생성하는 것이 당신의 과제입니다.   
template: templates/text\_to\_cypher.template   
intent\_detection:   
template: templates/intent\_detection.template   
generate\_summary:   
template: templates/summary.template

이 구성은 쿼리 생성을 안내하는 운영상 주석, 올바른 쿼리 구성을 보여 주는 예시, 그리고 프롬프트 템플릿 참조라는 세 가지 핵심 요소를 결합합니다. 주석은 그래프 데이터베이스를 다루는 데 필요한 핵심 도메인 지식과 모범 사례를 담고 있으며, 예시는 LLM이 기대되는 쿼리 패턴을 이해하는 데 도움이 되는 상세한 추론이 포함된 질문–답변 쌍을 제공합니다. 이러한 선언적 지식과 실제 예시의 결합은 LLM이 정확한 쿼리를 생성하도록 안내하는 풍부한 맥락을 만듭니다.

prompts 섹션은 서로 다른 파이프라인 단계(의도 탐지, 쿼리 생성, 요약 생성)를 위한 외부 템플릿을 참조하여, 실제 프롬프트 템플릿을 그 구성과 분리해 둡니다. 이러한 분리는 구성 구조와 프롬프트 내용 모두의 유지관리와 버전 관리를 더 쉽게 해 줍니다.

각 템플릿은 전체 구성에 영향을 주지 않고 독립적으로 수정할 수 있으므로, 개발 중에 서로 다른 프롬프트 변형을 실험하기가 더 쉬워집니다. 다음 목록은 구성 컴포넌트가 템플릿 로딩과 동적 콘텐츠 생성을 어떻게 관리하는지 보여 줍니다.

```python
Listing 15.2 Configuration component
class ChainConfiguration:
def __init__(self):
self.base = Path(__file__).parent
self.config = self.load()
def load(self):
config_file = self.base / "chain_config.yaml"
return yaml.load(config_file.open(), Loader=yaml.FullLoader)
def get_prompt(self, name, **kwargs):
system = self.config["prompts"][name].get("system")
```

```python
template_file = (
self.base / self.config["prompts"][name]["template"])
template = template_file.read_text()
prompt = jinja2_formatter(template, **kwargs)
return system, prompt
def getAnnotations(self, reload=True):
if reload:
self.config = self.load()
return {
"notes": self.config["notes"],
"examples": self.config["examples"]}
```

ChainConfiguration 클래스는 우리 구성 요소에 접근하기 위한 깔끔한 인터페이스를 제공합니다. 이 클래스는 두 가지 주요 메서드를 제공합니다. 하나는 프롬프트 템플릿을 검색하고 형식화하기 위한 get\_prompt이고, 다른 하나는 주석과 예시에 접근하기 위한 getAnnotations입니다. 이 구현은 구성 저장과 사용 사이의 명확한 분리를 유지하면서 모든 구성 요소에 쉽게 접근할 수 있도록 보장합니다.

#### 15.1.3 스키마 변환 서비스


다음으로, 우리 시스템 아키텍처에서 스키마 제공자 (Schema provider) 구성 요소와 그래프 데이터베이스의 상호작용을 살펴보겠습니다(그림 15.6 참조). 스키마 제공자는 전문가를 모방하는 질문-응답 시스템에서 핵심적인 구성 요소입니다. 우리의 목표는 스키마 추출을 자동화하는 것이지만, 14장에서 살펴본 근본적인 과제에 직면합니다. 즉, 개념 스키마 (conceptual schema)에 접근해야 하지만, 프로그래밍 방식으로는 기술 스키마 (technical schema)에만 접근할 수 있습니다.

![](images/ko/figure-15-6-ko.png)  
그림 15.6 그래프 데이터베이스에 연결하여 기술 스키마 정보를 추출하고 LLM 친화적인 형식으로 변환하는 스키마 제공자 구성 요소를 강조한 시스템 아키텍처 다이어그램

이를 해결하기 위해 우리는 두 가지 핵심 구성 요소로 이루어진 구성 기반 변환 접근법을 개발했습니다. 첫 번째 구성 요소는 개념 모델에서 제외해야 하는 요소를 식별하는 건너뛰기 목록 (skip list)입니다. 이러한 요소에는 비즈니스 개념을 나타내지 않는 기술적 노드와 관계, 내부 ID와 타임스탬프 같은 구현별 속성, 그리고 LLM 프롬프트에 불필요한 복잡성을 더할 요소가 포함됩니다. 두 번째 구성 요소는 필터링된 스키마를 풍부하게 하는 설명 섹션으로, 노드와 관계에 대한 비즈니스 수준의 설명, 속성에 대한 맥락 정보, 도메인별 용어와 제약 조건을 추가합니다.

우리는 이 구성을 YAML 형식으로 저장하여, 데이터 모델이 발전함에 따라 쉽게 유지보수하고 업데이트할 수 있게 합니다. 스키마 제공자는 원시 기술 스키마를 LLM 친화적인 형식으로 변환하기 위해 3단계 과정을 따릅니다. Neo4j의 apoc.meta.schema를 사용하여 기술 스키마를 추출하고, 건너뛰기 목록을 사용하여 기술적 요소를 필터링하며, 남은 요소를 비즈니스 설명으로 풍부하게 합니다.

이제 이것이 실제로 어떻게 구현되는지 살펴보겠습니다. 먼저 다음 목록과 같이 기술 스키마 정보와 풍부화된 스키마 정보를 모두 표현할 수 있는 데이터 모델이 필요합니다.

리스팅 15.3 스키마 제공자: 데이터 모델   
@dataclass   
class Property: ≤H   
"""선택적 설명이 있는 노드 또는 관계 속성을 나타냅니다"""   
name: str 노드를 나타내는 데이터 클래스   
type: str 또는 관계 속성   
description: str = None   
def \_\_str\_\_(self):   
"""속성을 다음 형식의 문자열로 나타냅니다:   
property\_name:TYPE /\* 선택적 설명 \*/ """   
ret = f"{self.name}: {self.type}"   
if self.description is not None:   
ret += f" /\* {self.description} \*/"   
return ret   
@dataclass 나타내는 데이터 클래스   
class Node: < 노드 유형   
"""노드 유형을 나타냅니다.""" 전역 수준에서 모든   
items = {} < 노드를 추적합니다   
name: str   
properties: list[Property]   
description: str = None   
노드 설명을 사용하여 노드를 인스턴스화합니다   
@classmethod apoc.meta.schema로부터   
def mk\_node(cls, name, value): <   
"""주어진 이름과 속성을 사용하여 딕셔너리로부터 새 노드를 생성합니다.   
인수:   
name (str): 노드의 이름.

```python
value (dict): the node description as
returned by `apoc.meta.schema`
II II II
properties = [Property(name=k, type=v["type"])
for k, v in value["properties"].items()]
properties = sorted(properties, key=lambda x: x.name)
node = Node(name=name,
properties=properties)
for rel_name, rel_value in value["relationships"].items():
Relationship.mk_rels(source=name, name=rel_name,
value=rel_value)
cls.items[node.name] = node < Stores the newly created
def drop_properties(self, skipProperties): node instance in the
"""Drops specified properties from the node. Node.items dictionary
Args:
skipProperties (list): A list of property names Recomputes
to be dropped. node properties
IIII II by filtering out
self.properties = [prop for prop in self.properties properties in
if prop.name not in skipProperties] the skip list
def _str__(self): < Assembles node
"""Represents the node as string in the format: components
(:NodeType /* node class description */ { to form the
property_one:TYPE /* property one description */, desired node
property_two:TYPE /* property two description */, description
format
})
II II II
descr = ("" if self.description is None
else f"/* {self.description} */ ")
return (
f"(:{self.name} {descr}{{\n " +
",\n ".join(str(prop) for prop in self.properties) +
"\n})\n"
```

이 데이터 모델은 우리의 스키마 변환 프로세스의 기반을 제공합니다. Property 클래스는 설명이 포함된 개별 속성을 처리하며, Node 클래스는 전체 구조와 필터링 기능을 관리합니다. 이 데이터 모델을 바탕으로, 다음 리스팅은 Neo4jSchema 클래스가 우리의 핵심 스키마 관리 기능을 어떻게 구현하는지 보여줍니다.

#### 리스팅 15.4 스키마 제공자 메인 클래스


```python
class Neo4jSchema:
Initializes the schema using
[...]
a technical schema from
def get_schema(self): < apoc.meta.schema
with self.driver.session() as session:
```

```python
result = list(session.run(
Parses ))[0]["value"] "CALL apoc.meta.schema({sample:-1})" Calls apoc.meta.schema
on the full database
apoc.meta.schema without sampling
results using list [Node.mk_node(k, v) for k, v in result.items()
comprehension if v["type"] == "node"]
Converts the technical schema
@staticmethod to conceptual by applying
def apply_configuration(config: dict = None): < filters and descriptions
Uses schema if config is None:
config_file = Path(__file__).parent / "schema_config.yaml"
config.yaml from
the package config = yaml.load(config_file.open(),
directory if no Loader=yaml.FullLoader)["schema"]
configuration
is provided items = {node.name: node for node in Node.items.values()
if node.name not in config["skip"]["classes"]}
Node.items = items < Recomputes the node types,
filtering out skip-list nodes
for node in Node.items.values():
Removes node.drop_properties(config["skip"]["properties"])
properties in the
skip list from for node in Node.items.values():
Node objects node.description = (config["descriptions"]["classes"]
.get(node.name))
for prop in node.properties:
Uses the property_description =(config["descriptions"]["properties"]
description from .get(node.name, {})
descriptions.classes .get(prop.name)) <
.<class_name>
prop.description = property_description
if exists Looks for
Filters out skip = config["skip"] descriptions.properties.<class
relationships with relationships = {rel_name: rel name>.<property_name>
the source node for rel_name, rel in Relationship.items.items()
in the skip list if rel.source not in skip["classes"]
if rel.dest not in skip[“classes"] if rel.name not in skip["relationships"] Filters out relationships
Filters out relationships with thedestination node in the skip list
list
for rel in Relationship.items.values():
rel.drop_properties(config["skip"]["properties"]) 4
Removes properties in the skip
Relationship.items = relationships
list from Relationship objects
def str__(self): <
Creates a Markdown
ret = ["### Graph Schema Overview\n", representation of the
"#### Node Types"] schema with node types
ret += [str(node) for node in Node.items.values()] and relationships
ret.append("#### Relationships\n")
ret += [str(rel) for rel in Relationship.items.values()]
return "\n".join(ret)
```

get\_schema 메서드는 기술적 스키마를 가져오며, apply\_configuration은 우리의 구성에 따라 변환 프로세스를 처리합니다. 이 구현은 질의 생성에 필요한 모든 정보를 유지하면서도 LLM이 우리의 데이터 모델에 대해 깔끔하고 개념적인 관점을 제공받도록 보장합니다.

실제로 이 변환된 스키마는 우리의 LLM 구성 요소에 여러 중요한 목적을 수행합니다. 이는 LLM이 도메인 모델을 개념적 수준에서 이해하고, 적절한 엔터티와 관계 이름을 사용하여 질의를 생성하며, 질의 구성 과정에서 비즈니스 규칙과 제약 조건을 고려할 수 있게 합니다. 이러한 접근 방식은 기술적 정확성을 유지하는 것과 LLM에 우리의 데이터 모델에 대한 접근 가능하고 비즈니스 지향적인 관점을 제공하는 것 사이에서 효과적인 균형을 만듭니다.

#### 15.1.4 상태 관리 설계


LangGraph에서 에이전트 통신의 초석은 상태 객체이며, 이는 에이전트가 읽고 쓸 수 있는 공유 메모리 공간 역할을 합니다. 각 에이전트는 이 상태의 특정 부분을 채울 책임이 있으며, 이를 통해 파이프라인 전반에 걸쳐 명확한 책임 사슬이 형성됩니다. 이 상태 객체의 구조를 살펴보겠습니다.

목록 15.5 파이프라인 에이전트의 상태   
class AgentState(TypedDict):   
question: str   
output\_type: str   
output\_type\_reason: str   
schema: str   
query: str   
query\_reasoning: str   
query\_message: str   
results\_error: list   
summary: str   
summary\_reason: str   
summary\_analysis: bool   
information: str   
retries: int

이 상태 구조는 논리적 구역으로 나눌 수 있습니다.

질문 입력(question)—원래 사용자 요청을 저장합니다.

의도 감지 결과(output\_type, output\_type\_reason)—감지된 시각화 의도와 그 근거가 되는 추론을 포착합니다.

스키마 정보(schema)—LLM 친화적 형식의 그래프 스키마를 포함합니다.

질의 생성(query, query\_reasoning, query\_message)—생성된 Cypher 질의와 관련 메타데이터를 보관합니다.

오류 처리(results\_error)—질의 실행 중 발생한 모든 오류를 추적합니다.

 요약 생성(summary, summary\_reason, summary\_analysis)—생성된 요약과 분석 플래그를 포함합니다.

질의 재시도 메커니즘(information, retries)—실패한 질의에 대한 재시도 로직을 관리합니다.

각 필드는 파이프라인의 진행 상황에 관한 이야기를 전달합니다. 상태는 에이전트 간에 데이터를 전달할 뿐만 아니라, 라우팅 결정을 내리고 오류를 원활하게 처리하는 데 필요한 문맥도 유지합니다.

#### 15.1.5 파이프라인 에이전트 구현


앞서 논의했듯이, 우리의 전문가 모방 접근 방식은 LangGraph 파이프라인에 자연스럽게 대응되며, 여기서 질의응답 과정의 각 단계는 특화된 에이전트로 구현됩니다. 그림 15.7은 이 파이프라인 구조를 다시 보여 주며, 에이전트들이 어떻게 연결되는지와 정보의 흐름이 초기 질문에서 최종 답변까지 어떻게 진행되는지를 강조합니다. 이 절에서는 이러한 에이전트들이 어떻게 만들어지고 LangGraph 상태 객체와 어떻게 상호작용하는지 살펴보겠습니다.

![](images/ko/figure-15-7-ko.png)  
그림 15.7 지식 전문가 모방 그래프 질의 파이프라인의 LangGraph 구현

#### 의도 감지 에이전트


의도 감지 에이전트 (intent detection agent)는 우리 파이프라인의 진입점 역할을 합니다. 즉, 사용자의 질문을 어떻게 시각화해야 하는지를 결정합니다. 이 에이전트는 오직 사용자의 입력 질문에만 기반하여 작동하며, 시각화 의도 정보로 상태를 보강합니다.

이 에이전트는 14장에서 논의한 의도 감지 프롬프트 (intent detection prompt)를 사용하여 질문을 분석하고 가장 적절한 시각화 유형을 결정합니다. 에이전트는 상태의 두 가지 핵심 필드를 업데이트합니다.

output\_type—결정된 시각화 형식(표, 그래프 또는 지도)

output\_reason—선택된 시각화 유형의 근거

목록 15.6의 구현을 살펴보겠습니다.

```python
Listing 15.6 Intent detection agent implementation
def run_prompt(self, prompt, system=""): < Handles prompt execution
messages = [HumanMessage(content=prompt)] and response processing
if self.system or system:
system = self.system if not system else system
messages = [SystemMessage(content=system)] + messages <
message = self.model.invoke(messages) Prepends a system message
to the prompt if provided
```

```python
logger.debug(f" got {message.content}")
payload = message.content Removes JSON code
payload = re.sub(r'^\s*```json\s*|\s*```\s*$', block markers from the
payload, flags=re.DOTALL) < response if present
return json5.loads(payload) <
Parses the response as JSON
using the more lenient JSON5
def intent_detection(self, state: AgentState):
system, prompt = self.config.get_prompt( <
"intent_detection", question=state["question"])
results = self.run_prompt(prompt, system)
return { < Retrieves and renders the
"output_type": results["type"], intent detection prompt
"output_reason": results["reason"]} template from configuration
Maps the response fields to their
corresponding state properties
```

에이전트는 설정에서 의도 감지 프롬프트 템플릿을 가져와 사용자의 질문으로 실행하고, LLM의 응답을 처리합니다. 응답은 시각화 유형과 그 근거를 포함하는 JSON 형식일 것으로 예상되며, 이후 이 값들은 해당 상태 필드에 매핑됩니다.

#### 스키마 추출 에이전트 (SCHEMA EXTRACTION AGENT)


스키마 추출 에이전트는 15.1.3절에서 소개한 Neo4jSchema 객체를 사용하여 우리의 KG와 LLM 구성 요소 사이의 가교 역할을 합니다. 이 에이전트의 주요 책임은 KG 스키마를 이후 에이전트들이 질의 생성과 추론에 사용할 수 있는 LLM 친화적인 형식으로 변환하는 것입니다. 다음 목록은 이 에이전트가 프롬프트 실행과 응답 처리를 어떻게 수행하는지 보여 주며, 특히 JSON 파싱과 오류 처리 (error handling)에 주목합니다.

![](images/ko/listing-15-7-ko.png)

핵심 작업은 Neo4jSchema 인스턴스가 수행합니다. 에이전트는 먼저 neo4j\_schema 객체가 제공되었는지 확인한 다음, 현재 스키마를 가져오고 우리가 정의한 모든 설정을 적용합니다. 그 결과 생성된 스키마는 LLM이 효과적으로 처리할 수 있는 문자열 형식으로 변환됩니다. 또한 에이전트는 재시도 카운터를 0으로 초기화하여, 파이프라인의 이후 단계에서 발생할 수 있는 질의 재시도에 대비해 상태를 준비합니다.

#### 텍스트-Cypher 에이전트

텍스트-Cypher 에이전트는 그래프의 스키마와 시각화에서 현재 선택된 요소를 모두 고려하여 사용자의 자연어 질문을 Cypher 질의로 변환합니다. 이러한 맥락 인식 덕분에 사용자는 선택된 노드나 관계를 명시적으로 설명하지 않고도 참조할 수 있으며, 14.7.2절에서 논의했듯이 질의를 더 자연스럽고 간결하게 만들 수 있습니다. 다음 목록에 표시된 것처럼, 에이전트는 프롬프트를 실행하기 전에 상태에 두 가지 정보를 추가합니다. 하나는 질의 생성을 안내하는 데 도움이 되는 설정 제공자의 주석이고, 다른 하나는 시각화 인터페이스의 현재 선택 상태입니다.

```python
Listing 15.8 Text-to-Cypher agent
def text_to_cypher(self, state: AgentState):
extra = {
"annotations": self.config.getAnnotations(),
"selection": self.selection
}
params = dict(state) | extra
system, prompt = self.config.get_prompt("text_to_cypher", **params)
logger.debug(f"prompt: {prompt}")
results = self.run_prompt(prompt, system)
return {"query": results["query"],
"query_reasoning": results["reasoning"],
"query_message": json.dumps(results)}
```

에이전트는 현재 상태를 추가 맥락(extra)과 병합하고, 적절한 프롬프트 템플릿을 가져와 채운 다음, 이를 LLM을 통해 처리합니다. 결과에는 생성된 Cypher 질의, 그 구성의 근거가 되는 추론, 그리고 디버깅 목적의 전체 LLM 응답이 포함됩니다. 이들은 각각 상태의 query, query\_reasoning, query\_message 필드에 저장됩니다.

#### 질의 실행 에이전트


다음에 제시된 질의 실행 에이전트는 시각화 요구에 기반하여 견고한 오류 처리와 동적 결과 형식을 제공합니다.

```python
def query_execution(self, state: AgentState):
try:
results = self.neo4j_schema.run(state["query"])
if state["output_type"] in {"graph", "map"}:
self.results = list(results)
else:
self.results = results.to_df()
results_error = None
information = ""
except neo4j.exceptions.ClientError as e:
```

```python
self.results = None
results_error = str(e)
logger.info(f"got error: {e}")
information = f"""We tried:
{state['query']}
and we got:
{str(e)}

retries = state.get("retries", 0) + 1
return {"results_error": results_error,
"retries": retries,
"information": information}
```

에이전트의 논리는 직관적입니다. 에이전트는 상태에 저장된 질의를 실행하려고 시도하고, 감지된 의도에 기반하여 결과를 처리합니다. 그래프 또는 지도 시각화를 다룰 때는 결과를 레코드 목록이라는 원래 형식으로 보존합니다. 그러나 표 형식 출력의 경우에는 Neo4j의 내장 변환 기능을 사용하여 결과를 pandas DataFrame으로 변환합니다.

오류 처리는 이 에이전트의 핵심적인 측면입니다. 질의 실행이 실패하면(일반적으로 구문 오류 또는 스키마 불일치로 인해 발생합니다), 에이전트는 여러 작업을 수행합니다. 즉, 오류 세부 정보를 포착하고, 디버깅 목적으로 실패를 기록하며, 시도한 질의와 오류 설명을 모두 포함하는 오류 메시지를 구성하고, 시도 횟수를 추적하기 위해 재시도 카운터를 증가시킵니다.

에이전트는 세 가지 정보로 상태를 업데이트합니다. results\_error 필드는 실행이 실패한 경우 오류 메시지를 포함하며, 그렇지 않은 경우 None으로 유지됩니다. retries 필드는 실행 시도 횟수를 추적하고, information 필드는 잠재적인 재시도 시도를 위해 오류에 관한 상세한 맥락을 제공합니다. 이 오류 정보는 실행 후 라우팅 논리에 중요하며, 이를 다음에 살펴보겠습니다.

#### 쿼리 후 실행


그림 15.8은 전체 파이프라인의 맥락에서 강조 표시된, 이 컴포넌트가 구현하는 라우팅 논리를 보여 줍니다. 우리가 논의한 다른 컴포넌트와 달리, 쿼리 후 실행은 에이전트가 아니라 LangGraph 파이프라인에서 동적 에지 (dynamic edge)로 라우팅 논리를 구현합니다(다음 목록 참조).

![](images/ko/figure-15-8-ko.png)  
그림 15.8 QA 파이프라인의 쿼리 후 실행 라우팅 논리로, 재시도, 요약, 직접 완료를 위한 결정 경로를 보여 줍니다.

목록 15.10 쿼리 후 실행 동적 에지   
def post\_query\_execution(self, state: AgentState): 재시도 논리로 쿼리 실행   
실패를 처리합니다   
if state["results\_error"] is not None: < 최대 세 번 시도   
if state["retries"] < 3:   
logger.info(f"{state['retries']} runs, we retry")   
return "retry"   
else:   
logger.info(f"{state['retries']} runs are enough")   
return "END"   
if state["output\_type"] in ("map", "graph"):   
요약으로 라우팅합니다   
logger.info("summarizing..") 지도/그래프 출력의 경우;   
return "summarize" 그렇지 않으면 완료합니다   
else:   
logger.info("no summarization is needed")   
return "END"

라우팅 논리는 두 가지 결정 경로를 따릅니다. 첫째, 상태의 results\_error 필드를 확인하여 쿼리 실행 실패를 처리합니다. 오류가 발생한 경우, 이 컴포넌트는 쿼리를 실행하기 위해 최대 세 번의 시도를 허용하는 재시도 메커니즘을 구현합니다. 이를 통해 우리 시스템은 일시적 실패나 LLM이 올바른 쿼리를 생성하는 데 여러 번의 시도가 필요할 수 있는 경우에 대해 복원력을 갖게 됩니다.

둘째, 성공한 쿼리의 경우 라우팅 결정은 output\_type에 포착된 시각화 의도에 따라 달라집니다. 지도 또는 그래프 시각화를 다룰 때 이 컴포넌트는 흐름을 요약 단계로 라우팅합니다. 이러한 시각화 유형은 추가적인 맥락과 설명의 이점을 얻기 때문입니다. 그러나 일반적으로 그 자체로 설명이 되는 표 형식 결과의 경우, 파이프라인은 직접 종료될 수 있습니다.

이러한 동적 라우팅 기능은 LangGraph의 핵심 기능으로, 실행 결과와 사용자 의도 모두에 기반하여 복잡한 흐름 제어를 구현할 수 있게 해 줍니다. 이 컴포넌트의 단순성은 전체 파이프라인 동작을 조율하는 데 있어 그 중요성을 겉으로 드러내지 않을 뿐입니다.

#### GENERATE-SUMMARY 에이전트


우리 파이프라인의 마지막 에이전트는 그래프 및 지도 시각화를 위한 요약을 생성합니다(목록 15.11 참조). 이 에이전트는 쿼리 결과와 스키마 선택을 결합하여 요약 생성을 위한 포괄적인 컨텍스트를 만듭니다. 또한 이러한 요소들을 기존 상태와 병합하여 요약 프롬프트 템플릿의 매개변수를 채웁니다. 구성 제공자는 적절한 프롬프트 템플릿을 제공하며, 이후 이 템플릿은 LLM을 통해 실행됩니다.

#### 목록 15.11 요약 생성 에이전트


```lua
def generate_summary(self, state: AgentState):
extra = {
"records": self.results,
"selection": self.selection
}
params = dict(state) | extra
system, prompt = self.config.get_prompt(
"generate_summary", **params)
logger.debug(prompt)
results = self.run_prompt(prompt, system)
return {"summary": results["summary"],
"summary_reason": results["reasoning"],
"summary_analisys": results["results_analysis"]}
```

에이전트의 출력은 실제 요약 텍스트, 요약의 근거, 추가 분석이 수행되었는지를 나타내는 플래그라는 세 가지 구성 요소로 상태를 보강합니다. 이로써 검색된 정보의 의미 있는 요약으로 원래 사용자 질문을 변환하는 우리의 파이프라인이 완성됩니다.

#### 파이프라인 조립

우리의 전문가 모방 파이프라인 구현은 LangGraph 워크플로의 구성으로 완성됩니다. 다음 목록은 우리의 에이전트들을 응집력 있는 그래프 구조로 연결합니다.

#### 목록 15.12 LangGraph 파이프라인 그래프 구축


```python
class Agent:
def __init__(self, model):
self.neo4j_schema: Neo4jSchema = None
self.selection = []
self.results = None
self.config = ChainConfiguration()
graph = StateGraph(AgentState)
graph.add_node("intent_detection", self.intent_detection)
graph.add_edge("intent_detection", "schema_extraction")
graph.add_node("schema_extraction", self.schema_extraction)
graph.add_edge("schema_extraction", "text_to_cypher")
graph.add_node("text_to_cypher", self.text_to_cypher)
graph.add_edge("text_to_cypher", "query_execution")
graph.add_node("query_execution", self.query_execution)
graph.add_conditional_edges("query_execution",
self.post_query_execution,
{"retry": "text_to_cypher",
"summarize": "generate_summary",
"END": END})
graph.add_node("generate_summary", self.generate_summary)
graph.add_edge("generate_summary", END)
graph.set_entry_point("intent_detection")
self.graph = graph.compile(checkpointer=self.memory)
self.model = model
```

그래프 구성은 우리의 질의응답 워크플로를 반영하는 명확한 순차적 패턴을 따릅니다. LangGraph의 StateGraph 클래스는 기반을 제공하며, 파이프라인 전체에서 타입 안전성을 보장하기 위해 우리의 AgentState 타입으로 초기화됩니다.

각 에이전트는 그래프의 노드로 추가되며, 간선은 이들 사이의 표준 흐름을 정의합니다. 이 그래프 구조는 우리의 전문가 모방 접근법을 구현하고, 단일 통합 파이프라인 안에서 오류와 다양한 시각화 요구사항을 원활하게 처리할 수 있는 유연성을 제공합니다.

#### 15.1.6 파이프라인 통합 계층


LangGraph는 복잡한 워크플로를 구축하기 위한 강력한 기능을 제공하지만, 실제로 애플리케이션이 이러한 파이프라인과 어떻게 상호작용할지 고려해야 합니다. 가장 단순한 접근법은 LangGraph의 호출 모드를 사용하는 것으로, 초기 상태를 제공하고 파이프라인이 완료되면 최종 결과를 받는 방식입니다. 그러나 이는 이상적이지 않은 사용자 경험으로 이어질 수 있습니다. 사용자는 내부에서 무슨 일이 일어나고 있는지에 대한 피드백 없이 오랜 시간 기다릴 수 있기 때문입니다. 그림 15.9는 LangGraph 파이프라인과 프런트엔드 애플리케이션 간의 실시간 상호작용을 가능하게 하는 통합 아키텍처를 보여줍니다.

![](images/ko/figure-15-9-ko.png)  
그림 15.9 LangGraph 상태 업데이트와 프런트엔드 상호작용 사이를 매개하는 질문 처리 인터페이스를 보여주는 파이프라인 통합 아키텍처

LangGraph는 중간 단계에 대한 가시성을 제공하는 스트림 실행 모드를 제공하지만, 이러한 업데이트를 직접 관리하면 애플리케이션 로직이 복잡해질 수 있습니다. 스트리밍의 장점과 사용 편의성 사이의 균형을 맞추기 위해, 우리는 파이프라인 실행을 프런트엔드가 쉽게 소비할 수 있는 일련의 명확히 정의된 이벤트로 변환하는 인터페이스 계층을 개발했습니다.

이 인터페이스의 핵심은 질문을 처리하고 일련의 이벤트를 산출하는 생성기 함수 (generator function)입니다. 생성기 함수는 단순하고 선형적인 코드 흐름을 유지하면서 중간 결과를 생성할 수 있게 해주기 때문에 이 작업에 매우 적합합니다. 파이프라인 실행은 강력한 이벤트 타이핑과 포괄적인 상태 추적을 갖춘 깔끔한 이벤트 스트림으로 변환되며, 이는 다음 목록에 나와 있습니다.

#### Listing 15.13 질문 처리 인터페이스 함수


```python
Configures the pipeline Sets up the internal selection list of
execution with a unique ID dictionaries when a non-empty
selection is provided
def processQuestion(question, selection=None):
> config = { "configurable": {"thread_id": uuid.uuid4().hex}}
if selection is not None: <
pipeline.selection = [{"labels": list(node.labels)[0],
"properties": dict(node)}
for node in selection] Invokes the pipeline in
else: stream mode, where
pipeline.selection = [] each update contains
only the changed
input = {"question": question}
portion of the state
results = pipeline.graph.stream(input,
config=config,
Processes each pipeline event stream_mode="updates") < First update: tells the
in this loop until completion user that the pipeline
yield "update", "*detecting intent...*", input running the first step
for result in results:
Extracts the > node, value = list(result.items())[0]
agent’s name logger.info(f"got results: {node}, keys: {list(value.keys())}")
and state current_state = pipeline.graph.get_state(config).values 4
updates from match node: Extracts the current full state
the LangGraph case "intent_detection":
result format yield "update", "*extracting schema...*", current_state
case "schema_extraction":
In case of intent yield "update", "*generating query...*", current_state
detection or > case "text_to_cypher":
schema extraction, yield "update", "*executing the query...*", current_state
notifies the user of yield "result", ("Reasoning", value["query_reasoning"]),
the next step current_state
case "query_execution":
Surfaces the if value["results_error"]: #J
text-to-Cypher yield "result", ("ERROR", value["results_error"]),
reasoning as an current_state
intermediate result else: <
output_type = current_state["output_type"]
In case of an error in the yield output_type, pipeline.results, current_state
execution of the query, if output_type in {"graph", "map"}:
surfaces the error yield "update", "*summary generation...*",
message as an current_state Otherwise, emits a
graph/table/map event
with the results as payload
```

```python
case "generate_summary": < Surfaces the
yield "result", ("Summary", value["summary"]), summary as an
current_state intermediate result
logger.info("no more results sendin END")
current_state = pipeline.graph.get_state(config).values
yield "END", current_state, current_state < Emits an END event containing
the final agent state
When the pipeline completes,
fetches the final agent state
```

이 함수는 초기 구성과 상태를 설정하는 것으로 시작한 다음, 의도 감지 (intent detection)가 시작되었음을 사용자에게 알리기 위해 첫 번째 업데이트 이벤트를 산출합니다. 파이프라인이 각 노드를 거쳐 처리되는 동안, 생성기는 프런트엔드에 정보를 전달하기 위해 적절한 이벤트를 산출합니다. 패턴 매칭 구조의 각 경우는 특정 파이프라인 단계에 대응하며, 현재 작업과 일치하는 이벤트를 생성합니다.

생성기가 생성하는 각 이벤트는 일관된 구조를 따릅니다. 즉, 응답 유형, 응답 페이로드, 그리고 파이프라인의 현재 상태를 포함하는 삼중항입니다. 응답 유형은 세 가지 범주로 나뉩니다.

업데이트 이벤트는 사용자에게 파이프라인의 진행 상황을 알려줍니다. 이러한 이벤트는 “의도 감지 중” 또는 “쿼리 생성 중”과 같은 단순한 상태 메시지를 전달하여, 사용자가 현재 어떤 단계가 실행되고 있는지 이해하도록 돕습니다.

 결과 이벤트는 추론 단계, 잠재적 오류, 또는 생성된 요약과 같은 텍스트 출력을 전달합니다. 이는 파이프라인의 의사결정 과정에 대한 더 깊은 통찰을 제공하고, 시스템이 어떻게 결론에 도달했는지 사용자가 이해하도록 돕습니다.

시각화 이벤트는 그래프, 지도, 차트, 또는 표와 같은 구조화된 출력을 나타냅니다. 이러한 이벤트는 쿼리 결과의 시각적 표현을 생성하는 데 필요한 데이터를 전달하여, 프런트엔드가 정보를 가장 적절한 형식으로 제시할 수 있게 합니다.

각 이벤트에 현재 파이프라인 상태를 포함함으로써, 우리는 프런트엔드가 이 정보를 어떻게 사용할지에 대해 가정하지 않고도 완전한 맥락을 제공합니다. 이 접근 방식은 관심사의 깔끔한 분리를 유지합니다. 즉, 인터페이스 계층은 파이프라인 실행을 잘 정의된 이벤트 스트림으로 변환하는 데 집중하고, 표현 방식에 관한 결정은 프런트엔드에 맡깁니다.

그 결과, 질문 응답 과정 전반에서 사용자에게 정보를 제공하고 사용자의 참여를 유지하면서도 깔끔한 아키텍처 경계를 보존하는 인터페이스가 만들어집니다. 복잡한 파이프라인 실행을 단순한 이벤트 스트림으로 변환함으로써, 우리는 유지보수성과 유연성을 희생하지 않으면서 풍부한 상호작용 경험을 지원하는 기반을 만들었습니다.

### 15.2 Streamlit 애플리케이션

전문가 모방 질문 응답을 위한 LangGraph 파이프라인을 구축했으므로, 사용자가 시스템의 역량과 효과적으로 상호작용하고 이를 검증할 수 있는 인터페이스가 필요합니다. 이 인터페이스는 우리의 전문가 모방 접근법에서 비롯되는 몇 가지 요구사항을 지원해야 합니다.

인터페이스는 사용자가 노드와 관계를 탐색하고 선택할 수 있도록 상호작용형 그래프 시각화를 지원해야 합니다. 또한 파이프라인이 다양한 단계를 거쳐 질문을 처리하는 동안 실시간 피드백을 제공해야 합니다. 자연어 상호작용을 위해서는 채팅과 유사한 인터페이스가 필수적이며, 시스템은 선택된 그래프 요소와 처리 맥락에 관한 복잡한 상태 정보를 유지해야 합니다.

Streamlit의 기능은 이러한 요구사항과 잘 부합합니다. 채팅 인터페이스에 대한 기본 지원은 사용자 메시지와 시스템 응답을 모두 갖춘 질문 응답 상호작용을 구현하기 위한 기반을 제공합니다. 이 프레임워크의 내장 데이터 시각화 기능은 사용자 정의 컴포넌트를 통한 확장성과 결합되어 효과적인 그래프 표현을 만들 수 있게 합니다. 무엇보다도 Streamlit의 Python 우선 접근법은 우리의 LangGraph 파이프라인과의 원활한 통합을 보장합니다. 프런트엔드와 백엔드가 동일한 Python 환경에서 작동하므로, 복잡한 API를 구축하거나 언어 간 직렬화를 처리할 필요가 없습니다.

우리의 파이프라인은 질문을 처리하면서 진행 상황과 중간 결과에 관한 업데이트를 생성합니다. Streamlit의 세션 상태 시스템은 자동 UI 업데이트와 결합되어, 이벤트 처리 메커니즘을 구축하지 않고도 이러한 변화를 실시간으로 반영할 수 있게 합니다. 사용자는 전문가 모방 시스템이 자신의 질문을 정확히 어떻게 처리하는지 확인할 수 있습니다.

이러한 특성 덕분에 Streamlit은 우리 시스템의 프로토타이핑과 테스트에 특히 적합합니다. 빠른 반복 주기는 다양한 유형의 질문이 어떻게 처리되는지, 여러 시각화 옵션이 어떻게 작동하는지를 신속하게 검증할 수 있음을 의미합니다. 이 프레임워크의 낮은 구현 부담은 프런트엔드 복잡성을 다루기보다 핵심 전문가 모방 기능을 테스트하고 개선하는 데 집중할 수 있게 합니다. 프로덕션 배포에서는 더 특화된 인터페이스가 필요할 수 있지만, Streamlit은 우리 시스템의 역량을 개발하고 시연하는 데 정확히 필요한 것을 제공합니다.

#### 15.2.1 애플리케이션 개요


우리의 다음 단계는 전문가 모방 그래프 탐색을 완전히 지원하는 기능적 인터페이스를 설계하는 것입니다. 이 인터페이스는 “자연스러운 상호작용”과 “실시간 피드백” 같은 추상적 요구사항을 서로 매끄럽게 함께 작동하는 구성요소로 변환해야 합니다. 그림 15.10은 우리 애플리케이션의 주요 인터페이스 레이아웃을 보여줍니다.

인터페이스 구성요소는 우리의 전문가 모방 파이프라인의 역량에 직접 대응됩니다. 각 요소는 시스템의 특정 측면을 지원하도록 설계되어 있습니다.

애플리케이션의 중심에는 KG의 노드와 관계를 시각화하는 그래프 캔버스가 있습니다. 이 중심 뷰는 자연스러운 탐색과 질문 워크플로를 지원하는 다양한 상호작용 영역으로 보완됩니다.

선택 열은 질문을 더 자연스럽고 맥락 인식적으로 만듭니다. 사용자는 그래프에서 특정 노드를 선택한 다음, 자연어를 사용하여 질문에서 이러한 선택 항목을 참조할 수 있습니다. 예를 들어 특정 노드가 선택된 상태에서 사용자는 각 자산을 명시적으로 지정해야 하는 대신 “이 자산들과 관련된 회사는 무엇입니까?”라고 질문할 수 있습니다. 이 선택 메커니즘은 우리 시스템의 맥락 인식성을 테스트하고, 파이프라인이 시각적 맥락을 얼마나 잘 이해하여 질의 생성 과정에 통합하는지 검증하는 데 중요합니다.

![](images/ko/figure-15-10-ko.png)  
그림 15.10 선택 기능, 인터랙티브 그래프 시각화, 실시간 응답 추적을 갖춘 질의응답 시스템을 보여주는 애플리케이션 인터페이스 레이아웃

하단의 질문 입력 영역을 통해 사용자는 자연어로 질문을 제기할 수 있습니다. 이러한 질문은 단순한 사실 확인에서 복잡한 관계 분석에 이르기까지 다양할 수 있으며, 그 과정에서도 자연스러운 대화와 유사한 상호작용 방식을 유지합니다.

오른쪽의 기록 영역은 질의응답 과정에 대한 포괄적인 뷰를 제공합니다. 우리의 전문가 모방 접근법은 다양한 유형의 응답을 생성할 수 있으므로, 이 영역은 여러 형식을 표시하도록 적응합니다. 답변에 지리 정보가 포함된 경우에는 인터랙티브 지도로 제시됩니다. 시스템이 표 형식 데이터가 가장 유익하다고 판단하는 경우에는 잘 정리된 표를 표시합니다. 중요한 점은, 시스템이 각 질문을 처리하는 동안 이 영역이 실시간으로 업데이트되어, 파이프라인을 통해 이용 가능해지는 중간 단계와 최종 결과를 보여준다는 것입니다.

기록 영역의 실시간 업데이트는 이중 목적을 수행합니다. 즉, 사용자에게 진행 상황을 계속 알려주는 동시에 전문가 모방 파이프라인의 추론 과정이 보이도록 합니다. 이러한 투명성은 사용자가 자신의 질문이 어떻게 처리되고 있는지, 그리고 특정 시각화나 응답 형식이 왜 선택되었는지를 이해하는 데 도움이 됩니다.

이 설계는 사용자가 KG를 탐색하고, 자신이 보는 내용에 대해 질문하며, 전달되는 정보에 적합한 풍부한 다중 형식 응답을 받을 수 있는 유연한 경험을 제공합니다. 각 질문은 독립적인 상호작용으로 존재하지만, 지속적인 업데이트와 보존된 기록은 매끄러운 탐색 경험을 만들어 냅니다.

#### 15.2.2 LangGraph 통합

Streamlit과 우리 LangGraph 파이프라인의 통합은 사용자에게 실시간 대화형 경험을 제공합니다. 이제 이러한 시스템들이 어떻게 함께 작동하는지 살펴보겠습니다.

이 통합은 이벤트 기반 패턴을 따릅니다. 사용자가 Send 버튼을 클릭하자마자 질문은 질문 처리 인터페이스를 거쳐 LangGraph 파이프라인으로 이동합니다. 최종 결과를 기다리는 대신, 우리 시스템은 파이프라인의 각 에이전트가 질문을 처리할 때마다 즉각적인 피드백을 제공합니다. 이러한 실시간 가시성은 사용자가 자신의 질문이 어떻게 분석되고 답변되는지 이해하는 데 도움을 줍니다.

다음 목록에 제시된 것처럼, 이러한 정보 흐름을 관리하기 위해 우리는 두 가지 목적을 수행하는 MessageHistory 객체를 구현합니다. 첫째, 상호작용의 전체 기록을 유지하여 사용자가 과거의 질문과 답변을 검토할 수 있게 합니다. 둘째, 파이프라인의 현재 상태를 저장하여 어떤 에이전트가 질문을 능동적으로 처리하고 있는지, 어떤 중간 결과가 생성되었는지를 추적합니다.

목록 15.14 메시지 기록 구현   
class MessageHistory: 메시지는 딕셔너리의 목록으로 저장되며,   
def init\_\_(self): 마지막 항목은 항상   
self.messages = [{}] < 현재 메시지를 나타냅니다.   
현재 메시지를 나타내는 딕셔너리는   
def update(self, message, finalize=False): 새 데이터로   
self.messages[-1].update(message) ≤ 업데이트됩니다.   
if finalize: < 메시지가 완료되면,   
self.messages.append({})   
다음 메시지를 저장하기 위해   
새 빈 딕셔너리가 추가됩니다.   
@staticmethod   
def display\_message(msg): < 단일 메시지가 인터페이스에   
with st.chat\_message("user"): 어떻게 표시되는지 정의합니다.   
st.markdown(msg["question"])   
with st.chat\_message("assistant"): <   
if "query\_reasoning" in msg:   
st.markdown(f"##### Reasoning\n\n\*\*output type\*\*:\   
{msg['output\_type']}\`\n\n\ 표시는   
{msg['query\_reasoning']}") 딕셔너리의 키에 따라   
"map" 키에 저장된 그래프 데이터를 if "table" in msg: 렌더링해야 할 섹션을   
지도 시각화 라이브러리와 st.table(msg["table"]) 나타내는 방식으로 조정됩니다.   
호환되는 형식으로 변환합니다 if "map" in msg:   
map\_ = folium.Map()   
V nodes\_to\_map(msg["map"], map\_) 생성된 Cypher 쿼리를   
st\_folium(map\_) 텍스트-Cypher 과정 세부 정보와 함께   
if "query" in msg: < 접을 수 있는 섹션에 표시합니다.   
with st.expander("Query...", expanded=False):   
st.markdown(f"\`\`\`cypher\n\n{msg['query']}\n\`\`\`")   
st.json(msg["query\_message"])   
요약이 있는 경우 if "summary" in msg:   
디버깅 목적을 위해 접을 수 있는 섹션에 st.markdown(f"##### Summary\n\n{msg['summary']}")   
생성 세부 정보를 추가합니다 with st.expander("extra...", expanded=False):   
V st.json({   
"summary\_reason": msg["summary\_reason"],   
"summary\_analisys": msg["summary\_analisys"]

```python
})
st.json(msg, expanded=False) < Includes the complete
message state for debugging
def display_messages(self): < Displays
for message in self.messages:
messages
if not message: sequentially
continue
self.display_message(message)
```

MessageHistory 클래스는 메시지 딕셔너리의 목록을 유지하며, 각 메시지는 단순 텍스트부터 복잡한 시각화까지 서로 다른 유형의 내용을 포함할 수 있습니다. update 메서드는 메시지를 점진적으로 구성할 수 있게 하며, 이는 우리 파이프라인 처리의 단계적 특성을 반영합니다. display\_message 메서드는 Streamlit의 컴포넌트를 사용하여 다양한 콘텐츠 유형을 렌더링합니다. 즉, 텍스트에는 Markdown을, 구조화된 데이터에는 표를, 지도에는 Python의 folium 라이브러리를 사용합니다. 이 구현은 정보 계층을 구성하며, 쿼리와 요약 추론 같은 상세 정보에는 확장 가능한 섹션을 사용합니다.

사용자 입력과 LangGraph 파이프라인 간의 통합은 임시 상태 업데이트와 영구 상태 업데이트를 모두 관리하는 반응형 패턴을 사용합니다. 질문을 처리할 때 시스템은 즉각적인 피드백을 보여 주는 동시에 영구적인 대화 기록도 유지해야 합니다. 이는 두 가지 상호 보완적인 메커니즘을 통해 달성됩니다.

임시 플레이스홀더를 사용하는 이벤트 처리는 파이프라인이 질문을 처리하는 동안 실시간 업데이트를 보여 줍니다. 이러한 업데이트는 즉각적인 피드백을 제공하지만 일시적이며, Streamlit 플레이스홀더를 사용하여 현재 파이프라인 상태를 표시합니다.

MessageHistory는 대화의 영구 상태를 누적합니다. 업데이트를 직접 보여 주는 대신, END 이벤트를 받을 때까지 각 메시지의 전체 상태를 수집합니다. 그런 다음 페이지는 MessageHistory의 표시 로직을 사용하여 다시 렌더링되며, 임시 업데이트를 대화의 최종적이고 지속적인 버전으로 대체합니다.

다음 목록에 구현된 이 접근법은 사용자가 즉각적인 피드백과 상호작용 기록의 영구적인 기록을 모두 볼 수 있도록 보장합니다.

#### 목록 15.15 사용자 입력 처리기


```ini
[...] Displays the user’s question in the
if question := st.chat_input("What is up?"): chat history under the “user” role
with chat:
with st.chat_message("user"): < Creates a placeholder in the
st.markdown(question) “assistant” section for
with st.chat_message("assistant"): < displaying real-time updates
placeholder = st.empty()
Extracts selected
selection = [state.canvas["byId"][int(item)] nodes from the canvas
for item in state.selection] state using their IDs
```

```python
for response_type, response, current_state in \
chain.processQuestion(question=str(question), Updates MessageHistory
Sends the question selection=selection): with the current
and selection to state.messages.update(current_state) < pipeline state
the pipeline and match response_type: < Routes event handling based
begins processing on the response type
case "update": < For “update” events, displays
placeholder.markdown(response) the Markdown-formatted
For “graph” or “map” response in the placeholder
events, updates the case "graph" | "map":
canvas visualization placeholder.markdown("*updating canvas...*")
with the graph data > store_to_canvas(response) For map-type
if response_type == "map": responses, stores
state.messages.update( node data for
{"map":state.canvas["nodes"]}) map visualization
Stores tabular data in
MessageHistory for table case "table" | "chart": Renders the pandas
or chart responses state.messages.update({"table": response}) DataFrame as a
placeholder.table(response) < Streamlit table
Creates a new > with st.chat_message("assistant"):
placeholder to preserve placeholder = st.empty() Formats and displays
the table display result events with
case "result": < title and content
title, content = response
response = f"##### {title}\n\n{content}" Creates a new
placeholder.write(response) placeholder to preserve
with st.chat_message("assistant"): < the result display
placeholder = st.empty()
Stores the final state in Triggers an interface redraw to
MessageHistory and marks case "END": show the complete response
the message as complete state.messages.update(current_state, finalize=True)
st.rerun() <
```

사용자가 질문을 제출하면 시스템은 채팅 인터페이스에 자리표시자 요소를 생성하고, 파이프라인이 질문을 처리하는 동안 이를 점진적으로 업데이트합니다. match 문은 다양한 유형의 응답을 처리합니다. 텍스트 응답에 대한 채팅 인터페이스 업데이트, 캔버스에 렌더링되는 그래프 및 지도 시각화, Streamlit의 테이블 컴포넌트를 사용하여 표시되는 표 형식 데이터, 제목과 내용이 포함된 형식화된 결과, 그리고 모든 UI 요소가 적절히 업데이트되도록 재실행을 트리거하는 END 이벤트를 처리합니다. MessageHistory와 이 이벤트 처리 시스템의 결합은 사용자에게 반응성이 높고 상호작용적인 경험을 제공합니다.

### 15.3 전문가 모방형 조사


우리의 전문가 모방형 시스템이 실제로 어떻게 작동하는지 보여주기 위해, 현실적인 조사 워크플로를 따라가 보겠습니다. 조사자가 시스템의 맥락 이해 능력과 의미 있는 통찰 제공 능력을 활용하면서, 자연어 질의를 사용해 범죄, 감시 카메라, 차량 간의 연결을 어떻게 탐색할 수 있는지 살펴보겠습니다.

우리의 조사는 범죄 사건에 초점을 맞춘 지식 그래프 (KG)의 하위 집합을 사용합니다. 그림 15.11에 나타난 것처럼, 스키마는 위치, 설명, 날짜-시간과 같은 속성을 포함하는 Crime 노드를 공간적 관계를 통해 ANPRCamera 노드, 즉 자동 번호판 인식 (automatic number plate recognition) 카메라와 연결합니다. 카메라는 차량을 감지할 때 CameraEvents를 생성하며, 각 목격의 타임스탬프와 위치를 모두 기록합니다. 이러한 이벤트는 Vehicle 노드에 연결되며, Vehicle 노드는 모델, 색상, 번호판 번호와 같은 속성을 저장합니다. 차량은 다시 소유자를 나타내는 Person 노드와 연결되며, 이 소유자들은 COMMITTED 관계를 통해 Crime 사건과 연결될 수 있습니다.

![](images/ko/figure-15-11-ko.png)  
그림 15.11 조사 질의를 위해 Crime, ANPRCamera, CameraEvent, Vehicle, Person 노드가 어떻게 상호 연결되는지를 보여주는 초점화된 스키마 시각화

겉보기에는 단순한 이 구조를 통해 공간 분석, 시간적 패턴, 관계 탐색을 결합하는 정교한 질의가 가능합니다. 이제 조사자가 우리의 시스템을 사용해 무단침입 사건과 관련된 관심 차량을 어떻게 식별할 수 있는지 살펴보겠습니다.

#### 15.3.1 초기 사건 식별


우리의 조사는 현재 수사 중인 범죄를 식별하도록 시스템에 요청하는 것에서 시작합니다. 이는 명시적 제약(“현재 수사 중”)과 암묵적 기대(분석에 의미 있는 범죄를 반환)를 결합한 자연어 질의를 이해하고 변환하는 능력을 보여줍니다. 시스템은 전문가 모사 파이프라인 (expert-emulating pipeline)을 통해 이 요청을 처리하며, 먼저 단일 노드와 그 속성을 표시하는 데 그래프 시각화가 가장 적합하다는 점을 감지합니다. 질의 생성 컴포넌트는 우리가 진행 중인 수사를 찾고 있음을 이해하고, 생성된 Cypher 질의에 관련 제약을 포함합니다.

시스템에 첫 번째 질문을 던져 보겠습니다. “현재 수사 중인 범죄 노드 하나를 반환하라”. 그림 15.12는 범죄 무단침입 사건을 나타내는 범죄 노드를 표시하는 시스템의 응답을 보여줍니다.

(a)  
![](images/ko/figure-15-12a-ko.png)

(b)  
![](images/ko/figure-15-12b-ko.png)  
그림 15.12 현재 수사 중인 범죄 노드를 보여주는 초기 질의 응답. 인터페이스는 (a) 선택 패널(범죄 노드를 더블 클릭하여 채워짐)과 중앙의 현재 노드에 대한 세부 정보를 표시하는 정보 패널이 있는 캔버스를 보여주며, (b) 질의 처리 세부 정보를 포함한 채팅 인터페이스를 보여줍니다.

범죄 노드를 클릭하면 사건에 대한 상세 정보가 캔버스 영역에 나타나며, 여기에는 "EB"로 시작하는 부분 번호판을 가진 검은색 차량을 언급하는 서술문이 포함됩니다. 이 정보는 조사가 진행됨에 따라 가치 있게 활용될 것입니다.

채팅 인터페이스에서는 시스템의 추론 과정을 볼 수 있으며, 시스템이 왜 이 범죄를 선택했는지, 그리고 우리가 진행 중인 수사를 받도록 보장하기 위해 질의를 어떻게 구조화했는지 설명합니다. 이러한 투명성은 사용자가 자신의 자연어 질문이 어떻게 해석되고 실행되는지 이해하는 데 도움이 됩니다.

우리 시스템이 생성한 요약은 범죄 무단침입으로서의 분류와 서술문에 차량 정보가 존재한다는 점을 포함하여 해당 범죄의 핵심 측면을 강조합니다. 이는 우리의 요약 생성 컴포넌트가 속성 필드에 묻혀 있을 수 있는 관련 세부 정보를 추출하고 강조할 수 있음을 보여줍니다.

이 초기 상호작용은 자연어 이해, 적절한 시각화 선택, 노드 속성의 지능적 요약이라는 우리 시스템의 여러 기능을 보여줍니다. 그러나 더 중요하게는, 조사를 진행하면서 이 맥락을 활용할 점점 더 복잡한 질의를 위한 기반을 마련합니다.

#### 15.3.2 감시 범위의 공간 분석


범죄 노드를 식별했으므로, 다음 논리적 단계는 해당 지역의 감시 범위를 확인하는 것입니다. 우리 시스템의 공간 추론 기능을 통해 정확한 좌표나 거리 계산을 지정할 필요 없이 인근 ANPR 카메라에 대해 질의할 수 있습니다.

우리는 범죄 노드를 더블클릭하여 선택 항목에 추가한 다음, 다음 프롬프트를 입력합니다. “선택된 범죄로부터 1 km 이내에 위치한 모든 ANPR 카메라 노드를 반환하십시오.” 여기서 “선택된 범죄”를 직접 참조할 수 있다는 점에 주목하십시오. 시스템은 현재 선택 항목으로부터 이 맥락을 이해합니다. 그림 15.13은 이 상호작용을 포착하며, 시스템이 그래프 시각화와 인터랙티브 지도를 모두 통해 어떻게 응답하는지를 보여줍니다.

이제 캔버스에는 범죄 노드와 함께 새로 발견된 ANPR 카메라 노드가 노란색으로 포함되어 있습니다. 이 응답이 특히 강력한 이유는 시스템이 그래프 보기와 함께 지도 시각화를 제공하기로 결정했다는 점입니다. 지도에는 두 개의 마커가 표시되는데, 하나는 사건 위치를, 다른 하나는 카메라를 나타내며, 각각 그래프의 해당 노드와 일치하는 색상을 사용합니다.

카메라의 위치는 전략적으로 가치가 있어 보입니다. 이 카메라는 무단 침입이 발생한 지역의 진입 지점 또는 이탈 지점일 수 있는 교차로 근처에 위치해 있습니다. 이중 시각화를 통해 즉시 명확해진 이러한 공간적 통찰은 이 카메라의 데이터가 우리 조사에 유용한 단서를 제공할 수 있음을 시사합니다.

이 단계는 우리 시스템의 여러 정교한 기능, 즉 공간 질의 처리, 선택 인식 자연어 이해, 지능적 시각화 선택을 보여줍니다. 당면한 데이터에 가장 적합한 시각화를 선택하는 적응형 응답은 전문가 모방 접근법의 핵심 특징입니다.

(a)  
![](images/ko/figure-15-13a-ko.png)  
(b)

![](images/ko/figure-15-13b-ko.png)  
그림 15.13 범죄 위치 근처의 ANPR 카메라를 보여주는 공간 질의 응답 (a). 시스템은 범죄와 인근 ANPR 카메라 사이의 공간적 관계를 표시하기 위해 지도 시각화 (b)를 자동으로 선택했습니다.

#### 15.3.3 차량 패턴 탐지


이제 사건 보고서의 설명과 일치하는 차량을 검색할 수 있습니다. ANPR 카메라의 노드를 더블 클릭하여 이를 포함하고, 범죄 노드와 함께 현재 선택 항목에 추가합니다.

다음 프롬프트는 이 선택 항목을 사용하면서 구체적인 차량 기준을 추가합니다. “2023년 6월 15일에 선택된 카메라가 탐지한 차량을 반환하라. 차량은 검은색이며 번호판은 EB로 시작해야 한다.” 선택된 요소(“선택된 카메라”)에 대한 참조와 차량의 외관 및 번호판에 관한 명시적 제약을 어떻게 결합할 수 있는지 주목하십시오. 그림 15.14는 시스템이 일치하는 차량과 해당 탐지 이벤트를 포함하도록 시각화를 확장하는 방식을 보여줍니다.

이제 그래프는 탐지 이벤트를 통해 ANPR 카메라에 연결된 여러 차량 노드를 표시하며, 질의의 추론 과정이 채팅 인터페이스에 보입니다. 시스템은 시간적 제약(“2023년 6월 15일”), 차량의 물리적 설명(“검은색”), 부분 번호판 번호(“EB”)를 이해했으며, 이 모든 요소를 하나의 일관된 질의에 통합했습니다.

요약은 일치하는 각 차량을 강조하여 잠재적 용의자에 대한 개요를 제공합니다. 이는 노드 선택, 시간적 제약, 속성 일치를 결합하는 다면적 질의를 처리할 수 있는 우리 시스템의 능력을 보여줍니다.

(a)  
![](images/ko/figure-15-14a-ko.png)

![](images/ko/figure-15-14b-ko.png)  
그림 15.14 일치하는 차량과 해당 탐지 이벤트를 보여주는 차량 질의 결과. 각 경로는 완전한 차량 탐지 기록을 나타내며, 이벤트 노드에는 타임스탬프가 표시됩니다. 시스템의 응답에는 (a) 그래프 시각화와 (b) 각 차량 속성에 대한 상세 요약이 모두 포함됩니다.

이 단계는 범죄와 카메라 위치에 관한 고립된 데이터 포인트를 우리 사건과 연결될 수 있는 차량 집합으로 변환함으로써, 수사에서 중요한 진전을 나타냅니다. 그러나 수사 목표에 관한 추가 맥락을 제공하면 더 많은 통찰을 얻을 수 있을까요? 다음 프롬프트에서 이를 살펴보겠습니다.

#### 15.3.4 맥락 인식 요청 정제


선택된 노드에서 정보를 추출하고 맥락 인식 (context-aware) 요약을 수행하는 전문가 모방 시스템 (expert-emulating system)의 능력을 활용하여 접근 방식을 정제해 보겠습니다. 차량 설명과 날짜 제약을 명시적으로 진술하는 대신, 시스템이 선택된 범죄 노드에서 이러한 세부 정보를 추출하도록 할 수 있습니다. 또한 우리의 역할과 수사 의도에 관한 맥락을 제공함으로써, 시스템이 더 분석적인 통찰을 생성하도록 유도할 수 있습니다.

우리는 이러한 수사 맥락을 반영하도록 프롬프트를 다음과 같이 바꿉니다. “나는 수사관이며 선택된 범죄를 조사하고 있습니다. 설명과 부합하고 사건 당일 선택된 카메라에 감지된 모든 차량 노드가 필요합니다. 그중 사건에 연루되었을 가능성이 현저히 더 높아 보이는 것이 있습니까?” 그림 15.15에 표시된 응답은 추가 맥락이 시스템의 분석을 어떻게 변화시키는지 보여 줍니다.

(a)  
![](images/ko/figure-15-15a-ko.png)

(b)  
![](images/ko/figure-15-15b-ko.png)  
그림 15.15 수사 맥락이 포함된 동일한 차량 데이터에 대한 향상된 분석 (a). 시스템은 (b)의 응답에 관심 패턴을 식별하는 분석 섹션을 추가하며, 추가 맥락이 동일한 기본 데이터에 대해 더 통찰력 있는 요약으로 이어지는 방식을 보여 줍니다.

쿼리는 이전과 동일한 차량을 반환하지만, 이제 요약에는 결과에 대한 더 심층적인 분석이 포함됩니다. 시스템은 흥미로운 패턴을 식별합니다. 일치하는 차량 중 하나가 사건 발생 시간 전후에 두 번 감지되었으며, 이는 추가 조사가 필요한 해당 지역의 잠재적 순회를 시사합니다.

이 향상된 응답은 시스템이 노드 속성에서 제약을 자율적으로 추출하고 적용하여 명시적 재진술의 필요성을 제거할 수 있음을 보여 줍니다. 또한 이 경우 수상한 패턴을 찾는 수사관으로 우리 자신을 식별하는 것처럼 수사 맥락을 제공하면, 시스템이 더 관련성 높고 통찰력 있는 요약을 생성할 수 있음을 보여 줍니다.

시스템은 단순히 기준을 매칭하는 수준을 넘어, 수상한 행동을 나타낼 수 있는 시간적 패턴을 능동적으로 분석하는 단계로 나아갔습니다. 이는 최종 수사 단계, 즉 이러한 차량 중 알려진 범죄자와 연관이 있는 차량이 있는지 검토하는 단계의 기반을 마련합니다.

#### 15.3.5 과거 기록 분석

수상한 이동 패턴을 보이는 차량을 발견한 것을 바탕으로, 차량 소유자의 범죄 이력을 고려함으로써 수사를 더욱 풍부하게 만들 수 있습니다. 이러한 유형의 배경조사는 표준적인 수사 관행이며, 우리 시스템은 이를 분석에 자연스럽게 통합할 수 있습니다.

우리는 프롬프트 (prompt)를 약간 수정합니다. “저는 수사관이며 선택된 범죄를 조사하고 있습니다. 사건 당일 선택된 카메라에 감지되었고 설명과 부합하는 모든 차량 노드가 필요합니다. 이 차량들 중 일부는 전과자가 소유하고 있을 수 있습니다. 사건에 연루되었을 가능성이 가장 높은 차량은 무엇입니까?”

그림 15.16은 우리 수사에서 중요한 돌파구를 보여 줍니다. 시스템은 소유 관계와 범죄 기록을 포함하도록 분석 범위를 확장하여, 이전에 수상한 이동으로 표시된 차량이 관련 범죄 이력이 있는 개인의 소유임을 발견합니다.

특히 주목할 점은 이 사람의 기록에 이전의 범죄적 무단침입 유죄 판결이 포함되어 있다는 것입니다. 이는 우리가 현재 조사하고 있는 범죄와 동일한 유형의 범죄입니다. 요약 에이전트 (summary agent)는 이러한 연결의 중요성이 높아졌음을 인식하고, 일반 요약과 분석 섹션 모두에서 이를 강조합니다.

이 마지막 단계는 시스템이 다음의 모든 작업을 수행할 수 있음을 보여 줍니다.

여러 차례의 쿼리 (query) 정교화 전반에 걸쳐 맥락적 인식을 유지합니다.

다양한 유형의 증거(공간적, 시간적, 역사적 증거)를 통합합니다.

의미 있는 패턴과 연결을 식별합니다.

수사 의사결정을 직접적으로 지원하는 방식으로 결과를 제시합니다.

이 수사를 통해 우리는 자연어 상호작용, 지능형 요약, 맥락 인식 분석이 결합되어 실제 수사 워크플로를 어떻게 지원하는지 살펴보았습니다.

![](images/ko/figure-15-16-ko.png)  
그림 15.16 범죄 이력을 드러내는 최종 수사 통찰. (a) 그래프는 차량 소유자가 이전의 범죄적 무단침입을 포함한 여러 과거 범죄와 연결되어 있음을 보여 주도록 확장됩니다. (b) 요약은 과거 범죄에 대한 상세한 분석을 제공하며, 시스템이 시간적, 공간적, 역사적 증거를 하나의 일관된 수사 서사로 통합할 수 있음을 보여 줍니다.

### 15.4 향후 방향과 개선 사항


14장과 15장 전반에 걸쳐 우리는 KG를 위한 질의응답 시스템의 개발을 살펴보았습니다. 이를 순수한 언어 모델링 문제로 다루기보다는, 인간 전문가가 그래프 데이터베이스를 다루는 방식을 모방하는 시스템, 즉 스키마 맥락을 이해하고 추론된 단계들을 통해 쿼리를 구성하는 시스템을 구축했습니다. 우리의 구현은 이러한 전문가 모방 접근법이 특화된 에이전트들의 파이프라인을 통해 어떻게 실현될 수 있는지를 보여 주며, 각 에이전트는 과정의 서로 다른 측면을 처리합니다.

그러나 이 구현은 즉시 사용 가능한 완제품 솔루션이라기보다는 기반으로 보아야 합니다. 그 가치는 무엇을 수행하는가뿐만 아니라 어떻게 구축되었는가에도 있습니다. 이 기반의 강점은 특히 관측 가능성 (observability)과 전문가 모방에 접근하는 방식에서 드러나는 기저 아키텍처에서 비롯됩니다.

시스템의 투명성은 단순히 디버깅을 위한 것이 아닙니다. 이는 피드백을 수집하고 개선점을 식별하기 위한 자연스러운 지점을 만들어 냅니다. 각 구성 요소의 의사결정 과정이 가시적이므로, 시스템이 무엇을 하는지뿐만 아니라 왜 그렇게 하는지도 이해할 수 있습니다.

마찬가지로 중요한 것은 설계의 기저에 놓인 전문가 모방 패턴입니다. 새로운 과제나 개선 기회에 직면했을 때, 팀은 단순하지만 강력한 질문에서 출발할 수 있습니다. “전문가라면 이 상황에서 무엇을 할 것인가?” 시스템 설계에 대한 이러한 인간 중심적 접근은 개선을 억지스럽기보다 자연스럽게 느껴지게 합니다. 개선은 전문가 행동을 이해하고 모델링하는 데서 나타납니다.

다음 절들에서는 이 기반 위에 구축할 수 있는 여러 경로를 살펴보겠습니다. 이는 시스템의 관측 가능하고 전문가를 모방하는 특성이 어떻게 특정 유형의 개선을 가능하게 하는지를 보여 주는 예입니다.

#### 15.4.1 사용으로부터의 학습


시스템 진화의 자연스러운 출발점은 우리의 핵심 설계 원칙 중 하나인 포괄적 관측 가능성 (observability)을 활용하는 것입니다. 시스템이 처리하는 모든 질의는 사용자 의도, 질의 패턴, 결과의 효과성에 관한 풍부한 정보를 생성합니다. 이러한 관측 가능한 데이터는 여러 수준에서 체계적 개선의 기회를 만들어 냅니다.

이 관측 가능성을 가장 직접적으로 적용할 수 있는 영역은 “불만과 유사한” 질문을 수집하고 분류하는 것입니다. 이는 사용자가 시스템의 응답이 자신의 요구를 충족하지 못했다고 나타내는 프롬프트를 의미합니다. 이를 실패로 간주하기보다, 우리의 관측 가능한 파이프라인은 이러한 사례들로부터 의미 있는 패턴을 추출할 수 있게 해 줍니다. LLM 기반 분석을 사용하여 이러한 패턴을 분류함으로써, 개발 우선순위와 자원 배분을 안내하는 전략적 도구가 될 고충 지점의 동적 대시보드에서 공통적인 사용자 과제를 드러낼 수 있습니다.

성공적인 상호작용의 수집도 마찬가지로 가치가 있을 것입니다. 시스템이 사용자가 특히 유용하다고 느끼는 질의를 생성할 때, 추론 과정 (chain of reasoning)을 보존할 수 있습니다. 그런 다음 이러한 성공 패턴을 체계적으로 분석하여 무엇이 그것들을 효과적으로 만드는지 식별할 수 있으며, 이는 우리의 예시 데이터베이스를 강화하기 위한 기반을 형성합니다.

사용자 질문의 패턴을 분석함으로써, 특화된 처리가 도움이 될 수 있는 유사 질의의 클러스터도 식별할 수 있습니다. 이러한 이해는 하위 프로세스를 최적화하는 데 사용될 수 있습니다. 예를 들어, 스키마에서 가장 관련성이 높은 부분만 선택하거나 특정 유형의 질문에 대해 더 초점화된 예시를 제공하는 방식입니다.

이러한 사용으로부터의 학습은 우리의 기반 아키텍처가 사전에 정해진 규칙이 아니라 실제 사용 패턴에 기반하여 시스템이 진화할 수 있게 하는 방식을 보여 줍니다. 개선은 이해 가능하고 전문가와 유사한 행동에 기반을 둔 상태로 자연스럽게 나타납니다.

#### 15.4.2 핵심 역량 강화


“전문가라면 어떻게 할 것인가?”를 일관되게 질문함으로써, 우리는 관찰 가능한 파이프라인 아키텍처를 사용하면서도 인간 전문가의 패턴과 부합하는 개선 사항을 식별하고 구현할 수 있습니다. 향상의 핵심 영역은 스키마 처리이며, 여기서 우리는 인간 전문가가 지식 그래프(KG)의 구조에 대한 이해를 구축하는 방식을 모방할 수 있습니다. 전문가들은 데이터가 어떻게 구조화되고 사용되는지 이해하기 위해 예비 질의를 실행하는 경우가 많으며, 이는 기본 스키마 정의를 넘어서는 것입니다. 우리는 데이터 패턴을 분석하고 추가 맥락으로 기본 스키마를 보강하는 스키마 강화 에이전트 (schema enrichment agents)를 구현함으로써 이러한 행동을 모델링할 수 있습니다. 이렇게 강화된 스키마 정보는 이후 단계로 흘러가 질의 생성과 결과 해석을 개선할 수 있습니다.

대규모 KG의 경우, 전문가들은 일반적으로 서로 다른 추상화 수준의 정신적 모델을 사용합니다. 다층 스키마 관리 접근법은 이러한 정신적 과정을 반영할 수 있습니다. 예를 들어, 우리의 조사용 KG에서 최하위 계층은 모든 상세 노드와 관계(범죄, 차량, 카메라 이벤트, 사람)를 포함하고, 상위 계층은 이를 차량 모니터링이나 형사 사법 기록과 같은 도메인 중심 뷰로 조직할 것입니다. 시스템은 광범위한 이해(예: 어떤 도메인을 질의할 것인지)를 위해 이러한 상위 수준 뷰를 사용하는 동시에, 구체적인 질의 생성을 위해 상세한 설명을 보존할 수 있습니다. 이러한 계층적 접근법은 인간 전문가가 필요에 따라 서로 다른 세부 수준을 확대하거나 축소하듯이, 정밀성을 희생하지 않으면서 복잡성을 관리합니다.

#### 15.4.3 고급 진화 경로


더 야심적인 진화 경로는 시스템의 역량을 크게 향상시킬 수 있습니다. 이러한 고급 접근법은 언어 모델 응용에서 새롭게 등장하는 기법을 통합하면서도 우리의 핵심 원칙을 유지합니다.

특히 유망한 방향은 스키마 이해와 질의 생성을 위해 순수한 문맥 내 학습 (in-context learning)을 넘어서는 것입니다. 프롬프트에 스키마 정보와 예시를 임베딩하는 현재의 접근법은 효과적이지만, 더 큰 KG에서는 확장성 문제에 직면합니다. 미세조정 (fine-tuning) 접근법은 대안적인 경로를 제공합니다. 스키마 자체를 학습 데이터로 사용함으로써, 그래프 구조와 관계를 더 깊고 효율적으로 이해하는 KG 인식 에이전트를 개발할 수 있습니다.

연구에 따르면 문맥 내 학습은 유연하지만, 과업 특화 적응 (task-specific adaptation) 접근법에 비해 일관되게 낮은 성능을 보입니다 [1], [2]. 이러한 성능 격차는 두 접근법이 동일한 예시 집합에 접근할 수 있는 경우에도 지속됩니다. 이러한 발견은 미세조정된 구성 요소로 이동하는 것이 시스템의 성능을 크게 향상시키는 동시에 잠재적으로 계산 오버헤드를 줄일 수 있음을 시사합니다.

미세조정된 구성 요소로의 전환이 우리의 전문가 모방 아키텍처를 포기해야 함을 의미하지는 않습니다. 대신 관찰 가능한 파이프라인 구조를 유지하면서 문맥 내 학습 구성 요소를 미세조정된 대안으로 선택적으로 대체하거나 보강할 수 있습니다.

이 진화 경로는 또한 더 정교한 질의 계획의 가능성을 열어 줍니다. 미세조정된 에이전트는 질의 패턴과 다양한 그래프 구조와의 관계를 더 미묘하게 이해할 수 있으며, 이는 잠재적으로 더 효율적이고 효과적인 질의 생성으로 이어질 수 있습니다. 시스템은 그래프 지식의 더 강력한 기반 위에 구축되면서도 단계별 추론 접근법을 유지할 수 있습니다.

#### 요약


전문가 모방 접근법은 인간 전문가가 그래프 데이터베이스와 상호작용하는 방식을 모방합니다.

LangGraph의 상태 기반 설계는 각 구성 요소가 독립적인 추론을 유지하는 모듈식이고 관찰 가능한 AI 파이프라인의 생성을 가능하게 합니다.

파이프라인 통합 아키텍처는 처리 업데이트를 대화형 사용자 인터페이스로 실시간 스트리밍할 수 있게 합니다.

 문맥 인식 질의 생성은 자연스러운 상호작용을 위해 스키마 지식, 사용자 선택, 대화 이력을 결합합니다.

다단계 분석 워크플로는 공간적, 시간적, 이력적 분석을 하나의 일관된 프로세스 안에 통합할 수 있습니다.

 메시지 이력 관리는 상태 유지 대화와 실시간 진행 상황 업데이트를 가능하게 합니다.
