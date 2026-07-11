---
lang: ko
format:
  html:
    toc: true
    embed-resources: true
    theme: cosmo
---

# 개방형 LLM과 도메인 온톨로지를 활용한 NED

### 이 장에서 다루는 내용


전통적인 개체명 중의성 해소 (named entity disambiguation, NED) 도구의 한계 이해

범용 LLM과 도메인 온톨로지를 결합하여 개체명 중의성 해소 수행

최단 경로 탐지, 경로-텍스트 변환, 텍스트 경로 요약을 통한 다단계 중의성 해소 수행

7장은 개체명 중의성 해소에 초점을 맞추며, spaCy 프레임워크를 기반으로 구축된 특화된 자연어 처리 (natural language processing, NLP) 도구인 scispaCy의 역할을 강조합니다. 이 도구는 생의학 도메인의 사전학습 모델을 제공함으로써 문서와 출판물을 처리하도록 설계되었습니다.

### 8.1 전통적 개체명 중의성 해소 (named entity disambiguation, NED) 시스템의 한계 이해


scispaCy는 통합 의학 언어 시스템 (Unified Medical Language System, UMLS)과 같은 어휘집 및 온톨로지 (ontology)를 통합하며, 이는 텍스트의 언급 (mention)을 중의성 해소하는 데 유용한 정준 개체를 제공합니다. 그러나 이 접근법에는 몇 가지 한계가 있습니다.

특정 응용 도메인, 즉 생의학 분야를 위해 설계되었습니다.

새로운 개체와 용어를 통합하기 위해 참조 지식 베이스를 확장하고 업데이트하는 데 어려움이 있습니다.

지식 베이스에서 이용 가능한 방대한 정보를 충분히 활용하지 못합니다.

중의성 해소 작업에 개체 간의 기존 관계와 경로를 사용하지 않습니다.

마지막 지점의 영향을 이해하기 위해, 7장 초반에 논의한 예를 다시 살펴보겠습니다. 이 예는 유럽질병예방통제센터 (European Centre for Disease Prevention and Control, ECDC)의 다음 인용문을 사용했습니다.

4월 13일 주에 벨리즈는 처음으로 모기 매개 지카 바이러스 전파를 보고했습니다. 선천성 지카 증후군 및 기타 신경학적 합병증의 관찰된 증가에 대한 업데이트 지카 바이러스 감염과 잠재적으로 관련된 소두증 및 기타 태아 기형.

scispaCy는 “Zika”라는 용어 주변의 문맥 단어를 사용하여 올바르게 중의성이 해소된 세 개체를 탐지합니다. 결과는 다음 목록에 표시되어 있습니다.

![](images/b8bc9f6588d7a9c102e924f2ea970951ada5988f330bed8f7f0a75f3eaef45af.jpg)

하지만 이제 “congenital” 및 “syndrome”과 같은 주변 단어를 포함하지 않는 다음의 약간 다른 예를 테스트해 보겠습니다.

Zika는 Flaviviridae 바이러스 과에 속하며, Aedes 모기에 의해 전파됩니다. Zika disease 및 chikungunya fever와 같은 기타 증후군의 영향을 받는 개인은 viral myalgia, infectious edema, infective conjunctivitis와 같은 증상을 자주 경험합니다. Zika의 심각한 결과는 임신 중 태반 장벽을 통과할 수 있는 능력 때문에 발생하며, 소두증과 선천성 기형을 유발합니다.

이전 예와 비교하면, 중의성 해소 단계를 뒷받침하는 단어가 없습니다. scispaCy 출력을 살펴보겠습니다.

인식된 개체: Zika 0 4   
순위화된 대상 후보:   
- C0276289 Zika Virus Infection   
- C0318793 Zika Virus   
- C4687930 Zika Virus Antibody Measurement   
인식된 개체: Zika disease 109 121   
순위화된 대상 후보:   
인식된 개체: Zika 278 282   
순위화된 대상 후보:   
- C0276289 Zika Virus Infection   
- C0318793 Zika Virus   
- C4687930 Zika Virus Antibody Measurement

이 모델은 첫 번째 문장과 세 번째 문장에서 “Zika”를 “C0276289 Zika Virus Infection”이라는 개체로 중의성 해소하지만, 두 번째 문장의 언급에 대해서는 대상 개체를 탐지하지 못합니다.

이 장에서는 개방형 대규모 언어 모델 (large language models, LLMs)과 도메인 온톨로지를 사용하는 새로운 접근법으로 이러한 한계를 다룹니다. scispaCy와 같은 도메인 기반 도구와 달리, 이 접근법은 풍부한 온톨로지를 사용할 수 있는 다른 응용 도메인에서도 사용할 수 있습니다.

### 8.2 도메인 온톨로지 수집


중의성 해소 (disambiguation) 과정을 추진하기 위해, 7장에서 소개한 SNOMED (Systematized Nomenclature of Medicine) 온톨로지를 사용합니다. 다시 상기하면, SNOMED는 450,000개가 넘는 개념과 풍부한 관계 유형 집합을 포괄하는 다국어 임상 용어 저장소입니다. 우리의 예시 시나리오에서는 다시 다음 두 파일을 사용합니다: sct2\_Description\_Full-en\_US1000124\_20220901.txt(개체 이름과 별칭, 그리고 개체 간 관계) 및 sct2\_Relationship\_Full\_US1000124 \_20220901.txt(개체와 관계를 식별하는 숫자 코드).

#### 참고 이 파일들의 예시는 7.5.2절을 참조하십시오.


그림 8.1은 SNOMED의 계층 구조와 노드를 통한 정보 전파를 보여 줍니다. 목록 8.3–8.5는 각각 다음을 수행합니다: sct2\_Relationship\_Full\_US1000124\_20220901.txt로부터 Neo4j에 노드와 관계를 생성하고, sct2\_Description\_Full-en\_US1000124\_20220901.txt로부터 이름과 별칭을 추출하며, 계층 구조를 따라 1단계 노드에서 더 깊은 노드로 정보를 전파합니다.

참고 이 목록들의 주석이 달린 버전과 더 자세한 내용은 7.6.4절을 참조하십시오. 전체 예제 코드는 이 책의 온라인 저장소에서 이용할 수 있습니다.

![](images/2af8bbeaf10df59220ffcc72117479e746d40538db96d847b51f70a8c5b86afc.jpg)  
그림 8.1 SNOMED 계층 구조의 예시입니다. 더 깊은 수준의 노드는 온톨로지의 원형적 개체인 1단계 노드의 정보를 사용하여 분류될 수 있습니다.

#### 리스팅 8.3 SNOMED 수집: 관계 로드


[...]   
class SnomedRelationshipsImporter(BaseImporter):   
[...]   
def set\_constraints(self):   
queries = [   
(   
"CREATE CONSTRAINT IF NOT EXISTS FOR (n:SnomedEntity) "   
"REQUIRE n.id IS UNIQUE"   
),   
(   
"CREATE INDEX snomedNodeName IF NOT EXISTS "   
"FOR (n:SnomedEntity) ON (n.name)"   
),   
(   
"CREATE INDEX snomedRelationId IF NOT EXISTS "   
"FOR ()-[r:SNOMED\_RELATION]-() ON (r.id)"   
),   
(   
"CREATE INDEX snomedRelationType IF NOT EXISTS "   
"FOR ()-[r:SNOMED\_RELATION]-() ON (r.type)"   
),   
(

```python
"CREATE INDEX snomedRelationUmls IF NOT EXISTS "
"FOR ()-[r:SNOMED_RELATION]-() ON (r.umls)"
),
]
for q in queries:
self.connection.query(q, db=self.db)
def import_snomed_rels(self):
query = """
UNWIND $batch as item
MERGE (e1:SnomedEntity {id: item.sourceId})
MERGE (e2:SnomedEntity {id: item.destinationId})
MERGE (e1)-[:SNOMED_RELATION {id: item.typeId}]->(e2)
FOREACH(ignoreMe IN CASE WHEN item.typeId = '116680003'
➥THEN [true] ELSE [] END|
MERGE (e1)-[:SNOMED_IS_A]->(e2)
)
" II "I
size = self.get_csv_size(snomedRels_file)
self.batch_store(snomed_rels_query, self.get_rows(snomedRels_file),
size=size)
```

#### 목록 8.4 SNOMED 수집: 이름과 별칭 로딩


```python
[...]
class SnomedNamesImporter(BaseImporter):
[...]
def import_snomed_names(self, snomedNames_file):
snomed_names_concepts_query = """
UNWIND $batch as item
MATCH (e1:SnomedEntity)
-[r:SNOMED_RELATION {id: item.conceptId}]->
(e2:SnomedEntity)
WHERE item.conceptId <> '116680003' AND r.id = item.conceptId
SET r.type = CASE
WHEN r.type IS NULL THEN item.termAsType
ELSE r.type END,
r.aliases = CASE
WHEN item.termAsType IN r.aliases THEN r.aliases
ELSE coalesce(r.aliases,[]) + item.termAsType END
IIIIII
snomed_names_entities_query = """
UNWIND $batch as item
MATCH (e:SnomedEntity {id: item.conceptId})
SET e.name = CASE
WHEN e.name IS NULL THEN item.term
ELSE e.name END,
e.aliases = CASE
```

WHEN item.term in e.aliases THEN e.aliases   
ELSE coalesce(e.aliases, []) + item.term END   
II I II   
size = self.get\_csv\_size(snomedNames\_file)   
self.batch\_store(   
snomed\_names\_concepts\_query,   
self.get\_rows(snomedNames\_file),   
size=size)   
self.batch\_store(   
snomed\_names\_entities\_query,   
self.get\_rows(snomedNames\_file),   
size=size)   
[...]

목록 8.5 SNOMED 수집: 1수준 노드에서의 레이블 전파

[...]   
class SnomedLabelPropagator():   
[...]   
def get\_rows(self):   
propagation\_query = """   
MATCH p=(n:SnomedEntity)<-[:SNOMED\_IS\_A]-(m:SnomedEntity)   
WHERE n.id= "138875005" // 루트 노드   
WITH distinct m as first\_node   
CALL apoc.path.expandConfig(first\_node, { // #A   
relationshipFilter: '<SNOMED\_IS\_A',   
minLevel: 1,   
maxLevel: -1,   
uniqueness: 'RELATIONSHIP\_GLOBAL   
}) yield path   
UNWIND nodes(path) as other\_level // #B   
WITH first\_node, collect(DISTINCT other\_level) as uniques   
UNWIND uniques as unique\_other\_level   
WITH first\_node,unique\_other\_level   
WHERE not first\_node.name in   
coalesce(unique\_other\_level.type,[])   
RETURN unique\_other\_level.id as id, first\_node.name as label - C   
" I II   
with self.\_driver.session(database=self.\_database) as session:   
result = session.run(query=propagation\_query)   
for record in iter(result):   
yield dict(record)   
[...]

SNOMED\_IS\_A 관계는 엔터티 간의 계층적 연결을 사용하여 트리 구조 전반에 의미 유형을 전파하는 데 사용되었습니다.

### 8.3 Ollama와 Llama 3.1 8B로 모델 설정하기


이전 장들에서는 OpenAI API를 사용하여 NLP 작업을 수행하는 방법을 살펴보았습니다. 이제 Ollama와 Llama 3.1 8B(Meta에서 공개)를 사용하여 NED 시스템을 로컬에 배포함으로써 그 지식을 확장합니다.

Ollama는 사용자가 로컬 머신에서 직접 LLM을 실행할 수 있게 해 주는 오픈 소스 도구입니다. 모델을 로컬에서 실행하면 외부 제공자에 대한 의존성과 지연 시간을 줄이는 동시에 데이터에 대한 완전한 통제권을 확보할 수 있습니다. Llama 3.1 8B는 80억 개의 매개변수를 가진 오픈 소스 LLM입니다. 이 모델은 최대 128,000토큰의 문맥 길이 (context length)를 지원하며 다국어 정보 처리에 최적화되어 있습니다. 또한 소비자용 하드웨어에서 효율적으로 배포되도록 설계되었습니다.

로컬 머신에 Llama 3.1 8B 모델을 배포하려면 먼저 Ollama를 다운로드하여 설치해야 합니다. 이 도구는 macOS, Linux, Windows와 호환되며, 설치 파일은 https://ollama.com/ 에서 제공됩니다. Ollama는 명령줄과 그래픽 사용자 인터페이스(GUI) 옵션을 모두 제공하며, 우리는 NED 시스템을 위해 Llama 3.1 8B를 다운로드하고 배포하는 데 다음 명령줄 지침을 사용했습니다.

#### 목록 8.6 Llama 3.1 8B를 다운로드하고 설치하기 위한 Ollama 명령어


ollama serve   
ollama pull llama3.1:latest

Ollama는 OpenAI Chat Completions API와의 내장 호환성을 제공하므로, 이전 장들에서와 같은 Python 코드를 사용하여 로컬에 배포된 모델과 직접 상호작용할 수 있습니다. 이는 모델을 우리의 NED 시스템에 통합하는 과정을 단순화합니다. 다음 목록은 우리 모델과 상호작용하는 데 필요한 Python 클래스를 보여줍니다.

#### 목록 8.7 Python에서 우리의 Llama 3.1 8B 모델 실행하기


```python
from openai import OpenAI
class LLM_Model():
def init__(self, url='http://localhost:11434/v1', key="default"):
self.client = OpenAI(
base_url= url, < Default URL through
api_key = key, < which the model is served
“api_key” is required for the OpenAI Chat
def generate(self, messages): Completions API but is not used for open models.
response = self.client.chat.completions.create(
model="llama3.1:latest", < Specifies llama3.1:latest,
messages=messages, downloaded by Ollama
temperature=0,
max_tokens=4000,
top_p=1,
frequency_penalty=0,
presence_penalty=0,
)
#### It assumes as response the ChatGPT API format
return response.choices[0].message.content
```

참고 다음 절의 프롬프트를 사용하여 생성한 예제 결과는 2024년 10월 최신 버전의 Llama 3.1 모델을 사용하여 얻었습니다.

NED에 Llama 3.1과 같은 범용 모델을 사용하면, 도메인 특화 온톨로지 (ontology)와 결합했을 때 LLM이 전문 영역에서도 우수한 성능을 발휘할 수 있는 잠재력을 보여줄 수 있습니다. 다음 절에서는 이 과정을 세분화하고, 시스템이 복잡한 생의학 텍스트에서 엔터티를 해석하고 중의성을 해소하여 정보 추출과 분석을 촉진하는 방법을 보여줍니다.

### 8.4 종단 간 NED 과정


그림 8.2는 입력 문서에서 중의성이 해소된 언급에 이르기까지 우리 예시의 NED 과정을 설명하는 정신 모델 (mental model)을 보여줍니다. 이 과정은 비정형 텍스트를 포함한 입력 문서에서 시작되며, LLM 기반 개체명 인식 (named entity recognition, NER) 구성요소가 이를 분석합니다. 여기서 LLM은 도메인 온톨로지에 내재된 지식을 사용하여 입력 텍스트에서 관련 생의학 엔터티를 식별하고 레이블을 부여합니다. 예를 들어 “Zika”와 같은 용어는 SNOMED에 따라 질병 (Disease) 개념으로 인식될 수 있습니다. 이 단계는 원시 텍스트를 구조화된 데이터로 변환하며, 이는 이후 단계에서 처리됩니다.

LLM 모델은 도메인 온톨로지를 사용하여 입력 텍스트에서 생의학 엔터티를 식별하고 레이블을 부여합니다. 예를 들어 “Zika”와 같은 용어는 SNOMED에서 “Disease” 개념으로 인식됩니다.  
LLM 모델은 중의성 해소를 지원하기 위해 도메인 온톨로지의 후보들 사이에서 최단 경로를 탐지하고, 경로 정보를 번역 및 요약하는 과정을 포함하는 다단계 접근법을 사용하여 가장 정확한 엔터티를 선택합니다.  
![](images/f3fb7c2cc0956dfd64a853aaab4d4d2aae5f9b7608b2b12bfca6d54fee22e54c.jpg)  
그림 8.2 생의학 텍스트 처리를 위해 LLM과 SNOMED와 같은 도메인 특화 온톨로지를 사용하도록 설계된 NED 시스템의 워크플로. 이 워크플로의 각 단계에는 입력 텍스트의 엔터티 중의성을 정확하게 해소하기 위한 상호작용이 포함됩니다.

다음으로 시스템은 NED 후보 선택 (candidate selection, CS) 단계로 이동합니다. 전문 검색 (full-text search) 메커니즘은 식별된 각 엔터티 언급에 대해 가능한 일치 항목 목록을 생성합니다. 예를 들어 “Zika”라는 용어는 SNOMED에서 Zika Virus, Zika Virus Infection, Congenital Zika Virus Infection과 같은 여러 엔터티에 대응할 수 있습니다. 이 단계는 잠재적인 중의성 해소 대상의 풀을 설정하며, 시스템은 다음 단계에서 이를 평가하여 가장 정확한 일치 항목을 찾습니다.

마지막 단계인 NED 후보 중의성 해소 (candidate disambiguation, CD)에서 LLM은 각 언급에 대해 가장 정밀한 엔터티 일치 항목을 결정하도록 선택을 정제합니다. 다단계 접근법은 온톨로지 구조 내 후보들 사이의 최단 경로를 식별한 다음, 중의성 해소를 문맥적 지식으로 지원하고 입력 문서에 언급된 각 엔터티가 온톨로지에서 대응하는 중의성이 해소된 엔터티에 정확히 매핑되도록 관련 경로 세부 정보를 번역하고 요약합니다.

이 LLM 주도 접근법은 중의성 해소 과정의 각 단계에 도메인 특화 온톨로지를 통합합니다. 온톨로지의 계층적 구조와 관계적 구조를 통합함으로써, 모델은 특히 용어가 여러 의미나 연관성을 가질 수 있는 복잡한 사례에서 엔터티 분류와 중의성 해소에 대해 더 많은 정보를 바탕으로 한 결정을 내릴 수 있습니다.

#### 8.4.1 명명된 엔터티 인식


NER의 목표는 구조화되지 않은 텍스트에 언급된 명명된 엔터티 (named entity)를 질병, 유기체, 시술과 같은 사전 정의된 범주로 식별하고 분류하는 것입니다. 이전 장들에서 논의했듯이, 한 가지 실용적인 접근법은 프롬프트 엔지니어링 (prompt engineering)을 사용하는 것이며, 여기서는 우리가 관심을 가지는 엔터티의 유형을 프롬프트에 명시적으로 정의합니다. 흔히 데이터 과학자나 데이터 엔지니어가 도메인 전문가와 협력하여 이러한 엔터티를 식별하고 정의합니다.

우리의 시나리오에서는 SNOMED의 구조화된 의학 지식을 통합하여 LLM이 생의학 텍스트에서 더 정밀하고 문맥을 인식하는 엔터티 인식을 수행할 수 있도록 합니다. 그림 8.3은 NER의 입력과 출력을 보여 줍니다.

NER을 위해서는 온톨로지에서 사전 정의된 모든 범주를 검색해야 합니다. 다음 쿼리는 SNOMED에서 범주를 검색합니다.

#### 리스팅 8.8 SNOMED에서 사전 정의된 범주 검색하기


MATCH (n:SnomedEntity)   
UNWIND n.type as named\_entity   
WITH DISTINCT named\_entity, count(named\_entity) as num\_of\_entities   
ORDER BY num\_of\_entities DESC   
RETURN collect(named\_entity) as named\_entities

NER 작업을 위한 프롬프트에서 이 쿼리의 결과를 사용할 수 있습니다. 리스팅 8.9는 이를 위해 우리가 정의한 프롬프트 메시지의 단순화된 버전이며, 전체 프롬프트는 코드 저장소에 있습니다.

LLM은 도메인 온톨로지를 사용하여 입력 텍스트에서 생의학 엔터티를 식별하고 레이블을 지정합니다. 예를 들어 “Zika”와 같은 용어는 SNOMED에서 “Disease” 개념으로 인식됩니다.

![](images/58956fe3b1376d6493a2d2dec1d27919a332548f2555e57913aefff09afe4332.jpg)  
그림 8.3 NED의 첫 번째 단계는 NER입니다. 우리의 시나리오에서 명명된 엔터티는 온톨로지에서 직접 파생됩니다. SNOMED에서 범주는 온톨로지의 첫 번째 수준 노드에 의해 정의되며, 이 정보는 다른 모든 노드로 전파됩니다.

#### 목록 8.9 NER을 위한 단순화된 프롬프트


```python
system = """You are an assistant capable of extracting named entities in the
➥medical domain.
Your task is to extract ALL single mentions of named entities from the text.
You must only use one of the pre-defined entities from the following list:
{named_entities}.
No other entity categories are allowed.
For each sentence, extract the named entities and present the output in
valid JSON format""".format(named_entities=named_entities))
input = """Risk factors for rhinocerebral mucormycosis include poorly
controlled diabetes mellitus and severe immunosuppression."""
assistant = [
{
"sentence": """Risk factors for rhinocerebral mucormycosis include
poorly controlled diabetes mellitus and severe immunosuppression.""",
"entities": [
{
"id": 0,
"mention": "Risk factors",
```

```jsonl
"label": "Events"
},
{
"id": 1,
"mention": "rhinocerebral mucormycosis",
"label": "Disease"
},
{
"id": 2,
"mention": "poorly controlled diabetes mellitus",
"label": "Disease"
},
{
"id": 3,
"mention": "severe immunosuppression",
"label": "Qualifier value"
}
]
}
]
```

이 프롬프트는 다음과 같이 구성됩니다.

시스템 지시—시스템 메시지는 SNOMED 온톨로지에서 의료 도메인의 범주에 속하는 명명된 엔터티의 출현을 추출하되, 관련이 없거나 범위 밖의 범주는 피하라고 지시합니다.

입력 텍스트—이 예에서 입력 텍스트는 의학적 상태(비뇌형 털곰팡이증)의 위험 요인을 논의하며, 인식되어야 하는 의학 용어인 당뇨병 및 면역억제와 같은 상태를 나열합니다.

어시스턴트 출력—시스템은 다음 필드를 포함하는 구조화된 JSON 출력으로 응답합니다.

sentence—시스템은 각 의미 단위를 개별적으로 분석하도록 하기 위해 텍스트를 문장별로 처리합니다.

– entities—출력에는 식별된 엔터티의 배열이 포함됩니다. 각 항목은 문장 내 엔터티를 고유하게 식별하는 ID, 텍스트에서 발견된 명명된 엔터티에 대한 언급(예: “비뇌형 털곰팡이증”), 그리고 Disease와 같은 SNOMED 범주에 따라 엔터티를 분류하는 레이블을 포함합니다.

다음은 NER 프롬프트로 지시된 LLM에 전달된 입력 텍스트의 예입니다.

#### Listing 8.10 NER을 위한 사용자 메시지


user = """임신 중 태반 장벽을 통과하는 지카의 능력으로 인해 소두증과 선천성 기형이 발생하므로 지카의 중증 결과가 나타납니다""".

그리고 다음은 Llama 3.1이 생성한 결과의 일부입니다.

Listing 8.11 Listing 8.10의 문장에서 나온 NER 출력   
"sentence": """Severe outcomes of Zika are due to its capacity to cross   
the placental barrier during pregnancy, causing microcephaly   
and congenital malformations. """,   
"entities": [   
{   
"id": 0,   
"mention": "Zika",   
"label": "Organism",   
"start": 19,   
"end": 22   
},   
{   
"id": 1,   
"mention": "microcephaly",   
"label": "Clinical finding (finding)",   
"start": 105,   
"end": 116   
},   
{   
"id": 2,   
"mention": "congenital malformations",   
"label": "Clinical finding (finding)",   
"start": 122,   
"end": 145   
}   
]

이 결과로부터 우리는 세 개의 서로 다른 엔터티를 식별하고 분류할 수 있습니다.

 “Zika”—Organism으로 식별되며, 문장에서의 위치는 문자 19부터 22까지에 걸쳐 있습니다.

“Microcephaly”—Clinical finding (finding)으로 레이블이 지정되어, 이것이 의학적 상태임을 나타냅니다. 이 언급은 문자 105부터 116까지에 나타납니다.

 “Congenital malformations”—역시 Clinical finding (finding)으로 레이블이 지정됩니다. 이는 문장에서 문자 122와 145 사이에 위치합니다.

LLM은 문장에서 언급의 시작 문자와 끝 문자를 정확하게 탐지하는 데 어려움을 겪습니다. 이러한 이유로 우리는 다음 Python 함수를 사용하여 후처리에서 start 및 end 필드를 생성했습니다.

Listing 8.12 언급의 시작 문자와 끝 문자를 계산하는 Python 함수

```python
def find_all_mention_indices(self, string, substring):
indices = []
start_index = 0
```

```python
while True:
start_index = string.find(substring, start_index)
if start_index == -1:
break # No more occurrences found
end_index = start_index + len(substring) - 1
indices.append((start_index, end_index))
#### Move start_index forward to search for the next occurrence
start_index += len(substring)
return indices
```

이 함수는 언급의 위치를 식별하며, 이는 전통적인 NER 시스템이 인식하는 정보입니다.

#### 8.4.2 후보 선택

NED 과정의 두 번째 단계는 CS이며, 이는 식별된 각 개체명의 의도된 의미와 일치할 수 있는 관련 엔티티 또는 개념을 식별합니다. 그림 8.4는 이 단계의 입력과 출력을 보여 주며, 후보 엔티티가 어떻게 선택되는지를 강조합니다.

![](images/0e1651bfb34cf2ce2086110dee44e33d3c9ee78cb1208a84970a7beacee6a5b5.jpg)  
그림 8.4 NED의 두 번째 단계는 후보 선택입니다. 이전 단계에서 감지된 각 엔티티 언급에 대해, 이 단계는 해당 언급을 가리킬 수 있는 잠재적 후보를 검색합니다.

다음 listing에 제시된 것처럼, CS의 입력은 NER 과정에 의해 입력 텍스트에 주석 처리된 언급과 도메인 온톨로지 (domain ontology)로 구성됩니다. 출력은 각 언급과 연관된 하나 이상의 후보 엔티티 목록입니다.

이 단계에서는 두 가지 이유로 LLM을 사용하지 않습니다. 첫째, LLM에 내재된 지식에 의존하기보다는 도메인 온톨로지에서 직접 후보를 검색하고자 합니다. 둘째, 온톨로지의 크기 때문에 이를 프롬프트에 전체적으로 로드할 수 없습니다. 따라서 CS를 효율적으로 수행하기 위해 Neo4j의 전문 검색 (full-text search) 기능을 사용하며, 이는 각 언급과 밀접하게 일치하는 온톨로지 내 문자열을 식별할 수 있습니다.

```python
class CandidateSelection:
[...]
def full_text_query(self):
query = """
CALL db.index.fulltext.queryNodes("names", $fulltextQuery,
➥{limit: $limit})
YIELD node
WHERE node:SnomedEntity AND ANY(x IN node.type
➥WHERE x IN $labels)
RETURN distinct node.name AS candidate_name, node.id
➥AS candidate_id
" I "I
return query
def generate_full_text_query(self, input):
full_text_query = ""
words = [el for el in input.split() if el]
if len(words) > 1:
for word in words[:-1]:
full_text_query += f" {word}~0.80 AND "
full_text_query += f" {words[-1]}~0.80"
else:
full_text_query = words[0] + "~0.80"
return full_text_query.strip() [...]
```

쿼리의 검색 공간을 줄이기 위해 \$labels 매개변수를 지정합니다. 이러한 레이블은 NER 단계의 출력에서 수집되며, 시스템이 언급된 엔티티 유형과 관련된 엔티티의 부분집합만 식별하도록 강제합니다.

참고: 전문 검색 메커니즘은 실행 가능한 해결책을 제공하지만, 텍스트 매칭을 통해 식별되지 않을 수 있는 추가 후보를 검색하기 위해 벡터 기반 검색 (vector-based search)을 통합함으로써 향상될 수 있습니다.

입력 용어로 “Zika”를 전달했을 때 쿼리의 JSON 결과는 다음과 같습니다.

#### Listing 8.14 CS 단계에서 업데이트된 명명 엔티티 중의성 해소 (NED) 결과의 예


```json
{
"id": 0,
"mention": "Zika",
"label": "Organism",
"start": 19,
"end": 22,
"candidates": [
{
"snomed_id": "50471002",
"name": "Zika virus"
},
{
"snomed_id": "3928002",
"name": "Zika virus disease"
},
{
"snomed_id": "762725007",
"name": "Congenital Zika virus infection"
}
]
}
```

candidates 필드는 시스템이 각 언급 (mention)에 대해 찾은 잠재적 일치 항목 또는 후보의 목록을 포함합니다. 각 후보는 SNOMED를 기반으로 해당 언급에 대한 가능한 해석을 나타냅니다. 다음 필드는 각 후보의 특징을 설명합니다.

snomed\_id—SNOMED에서 해당 개념의 고유 식별자

name—snomed\_id와 연결된 의료 엔티티의 이름

이 경우 후보는 “Zika virus” (50471002), “Zika virus disease” (3928002), “Congenital Zika virus infection” (762725007)입니다. 이러한 후보는 임상 용어에서 “Zika”가 가질 수 있는 가능한 의학적 의미를 나타내며, 중의성 해소 단계에서 추가 정제를 수행하기 위한 기반을 마련합니다.

#### 8.4.3 후보 중의성 해소


NED 과정의 최종 단계는 CD입니다(그림 8.5 참조). 우리는 문장 내에서 대상 엔티티와 함께 나타나는 다른 의학 엔티티가 제공하는 문맥 정보를 사용하는 전략을 적용합니다. 이러한 엔티티를 도메인 온톨로지 (domain ontology)의 구조화된 지식과 상호 참조함으로써, 선택된 후보를 검증하고 정제하여 가장 정확한 일치를 결정할 수 있습니다.

예를 들어, “Zika”와 “microcephaly”가 모두 언급된 문장을 생각해 보겠습니다. “Zika”와 함께 “microcephaly”가 존재한다는 것은 가치 있는 문맥을 제공합니다. 즉, 이 감염이 소두증을 유발하는 것으로 알려져 있으므로, 선천성 Zika 바이러스 감염과의 연관성을 시사합니다. 중의성 해소 과정은 이러한 동시 출현을 사용하여 “Zika”의 다른 잠재적 의미(예: 일반적인 바이러스 또는 관련 없는 용어)보다 선천성 Zika 바이러스 감염을 우선시할 수 있습니다.

LLM은 도메인 온톨로지의 후보들 사이에서 최단 경로 탐지를 포함하고, 중의성 해소를 지원하기 위해 경로 정보의 번역 및 요약을 함께 수행하는 다단계 접근법을 사용하여 가장 정확한 엔티티를 선택합니다.

![](images/4bb9dfc81ccb754eda80256915d2173043d82ed896c157d11f44b44a07325387.jpg)  
그림 8.5 NED의 세 번째 단계는 후보 중의성 해소입니다. 목표는 그래프 기반 알고리즘(최단 경로 탐지)과 LLM을 결합하여, 이전 단계에서 식별된 모든 가능한 후보 중에서 가장 적합한 일치를 선택하는 것입니다.

입력 문서에서 식별된 각 멘션에 대해 중의성이 해소된 엔티티를 생성하기 위해, 우리는 세 단계를 수행합니다.

1 최단 경로 탐지—우리는 한 문장 내 서로 다른 멘션과 연관된 후보 엔티티들 사이의 최소 길이 경로를 식별합니다. 이러한 연결을 매핑함으로써, 각 멘션의 의도된 의미를 명확히 하는 데 도움이 되는 잠재적 관계를 설정합니다.

2 경로를 텍스트로 번역—텍스트 정보를 처리하는 LLM의 강점을 활용하기 위해, 우리는 후보 엔티티를 연결하는 각 그래프 경로를 자연어 문장으로 번역합니다. 이러한 변환을 통해 LLM은 자신이 효과적으로 처리할 수 있는 형식으로 관계 정보를 해석할 수 있습니다.

3 텍스트 경로 요약—우리는 번역된 경로에서 도출된 모든 텍스트 정보를 종합적인 설명으로 요약합니다. 이 요약은 관계의 핵심을 포착하고, LLM이 더 정확한 중의성 해소 결정을 내리도록 지원합니다.

그림 8.6은 중의성 해소 과정에서 이러한 단계의 개요를 제공합니다. LLM은 경로-텍스트 번역과 텍스트 경로 요약 단계를 강화하여, 관계 정보를 효과적으로 해석하고 압축할 수 있게 합니다. 다음에 논의할 최단 경로 탐지는 Neo4j의 Graph Data Science(GDS) 라이브러리를 사용하여 후보들 사이의 연결을 식별합니다.

1단계. 문장 내에서 식별된 후보들 사이의 최단 경로를 탐지합니다. 예를 들어,   
(Congenital Zika virus infection)-[:OCCURRENCE]->   
(Congenital)<-[:OCCURRENCE]-(Micrencephaly)

(Congenital)<-[:OCCURRENCE]-(Micrencephaly)는 “Zika virus disease”와 “Chikungunya fever” 사이의 경로를 나타냅니다.

![](images/186e3c1da63da97da11cb6d027870990421ac54405f066cd02485e9c3f91a04c.jpg)  
그림 8.6 NED CD 단계: (1) 문장 내 엔티티 언급과 관련된 모든 후보 사이의 최단 경로를 탐지합니다. (2) 탐지된 경로를 자연어 문장으로 변환합니다. (3) 해당 문장들을 중의성 해소에 유용한 텍스트로 요약합니다.

#### 최단 경로 탐지


이 단계의 목표는 CS 단계에서 식별된 각 의료 엔티티 언급과 관련된 모든 가능한 후보 사이의 최단 경로를 탐지하는 것입니다. 다음 목록은 이 작업을 수행하기 위한 쿼리를 보여줍니다.

#### 목록 8.15 SNOMED 온톨로지에서 관련 경로를 추출하기 위한 Python 클래스


```python
class PathExtraction():
def __init__(self, model, store, candidates, named_entities):
self.model = model
self.store = store
self.candidates = candidates
self.named_entities = named_entities
[...]
def get_co_occs_query(self, s1_id, s2_id):
query = f"""
CALL gds.degree.stream('snomedGraph')
YIELD nodeId, score
WITH gds.util.asNode(nodeId).name AS name, score AS degree
ORDER BY degree DESC
LIMIT 350
WITH collect(name) as hub_nodes
MATCH (s1), (s2)
```

```python
WHERE s1.id="{s1_id}" AND
s2.id="{s2_id}"
WITH s1,
s2,
allShortestPaths((s1)-[:SNOMED_RELATION*1..2]-(s2)) AS paths,
hub_nodes
UNWIND paths AS path
WITH relationships(path) AS path_edges, nodes(path) as path_nodes,
hub_nodes
WITH [n IN path_nodes | n.name] AS node_names,
[r IN path_edges r.type] AS rel_types,
[n IN path_edges startnode(n).name] AS rel_starts,
hub_nodes
WHERE not any(x IN node_names WHERE x IN hub_nodes)
WITH [i in range(0, size(node_names)-1) | CASE
WHEN i = size(node_names)-1
THEN "(" + node_names[size(node_names)-1] + ")"
WHEN node_names[i] = rel_starts[i]
THEN "(" + node_names[i] + ")" + '-[:' + rel_types[i] + ']->'
ELSE "(" + node_names[i] + ")" + '<-[:' + rel_types[i] + ']-' END]
➥as string_paths
RETURN DISTINCT apoc.text.join(string_paths, '') AS `Extracted paths`
""".format(s1_id=s1_id, s2_id=s2_id, named_entities=named_entities)
return query
[...]
```

이 쿼리의 핵심 단계는 다음과 같습니다.

차수 계산—이 쿼리는 먼저 CALL Gds.Degree.Stream을 사용하여 그래프에서 가장 높은 “차수”(관계 수)를 가진 노드를 검색합니다. 이러한 노드는 연결성이 높은 허브 노드를 나타내며, 이후 더 의미 있고 덜 일반적인 연결에 집중하기 위해 제외됩니다.

 최단 경로 검색—이 쿼리는 ID를 기반으로 두 엔티티 s1과 s2 사이의 모든 최단 경로를 찾으며, 경로 길이를 하나 또는 두 개의 홉(관계)으로 제한합니다. 이러한 노드 사이의 관계는 “허브” 노드를 제외하고 일반적이거나 지나치게 광범위한 관계를 피하도록 필터링됩니다.

 경로 변환—경로가 식별되면, 쿼리는 이를 풀어 각 경로에 포함된 노드와 관계를 모두 수집합니다. 그런 다음 이러한 경로를 관계의 방향과 유형을 보여 주는 읽기 쉬운 문자열로 형식화합니다(예: (n1)-[:REL\_TYPE]->(n2)).

다음 목록은 탐지된 경로의 예를 보여줍니다.

#### 목록 8.16 Neo4j GDS 라이브러리를 사용하여 탐지된 경로


{   
"id": 1,   
"path": "(Congenital Zika virus infection)-[:OCCURRENCE]->(Congenital)   
➥<-[:OCCURRENCE]-(Micrencephaly)"   
},   
{

```c
"id": 2,
"path": "(Congenital Zika virus infection)-[:OCCURRENCE]->(Congenital)
➥<-[:OCCURRENCE]-(Acrocephaly)"""
},{
"id": 3,
"path": "(Congenital Zika virus infection)-[:OCCURRENCE]->(Congenital)
➥<-[:OCCURRENCE]-(Multiple congenital malformations)"
},
{
"id": 4,
"path": "(Congenital Zika virus infection)-[:OCCURRENCE]->(Congenital)
➥<-[:OCCURRENCE]-(Congenital malformation)"
},
{
"id": 5,
"path": "(Congenital Zika virus infection)-[:OCCURRENCE]->(Congenital)
➥<-[:OCCURRENCE]-([X]Other congenital malformations)"
},
{
"id": 6,
"path": "(Micrencephaly)-[:OCCURRENCE]->(Congenital)
➥<-[:OCCURRENCE]-(Multiple congenital malformations)"
},
{
"id": 7,
"path": "(Micrencephaly)-[:OCCURRENCE]->(Congenital)
➥<-[:OCCURRENCE]-(Congenital malformation)"
},
{
"id": 8,
"path": "(Micrencephaly)-[:OCCURRENCE]->(Congenital)
➥<-[:OCCURRENCE]-([X]Other congenital malformations)"
},
{
"id": 9,
"path": "(Acrocephaly)-[:OCCURRENCE]->(Congenital)
➥<-[:OCCURRENCE]-(Multiple congenital malformations)"
},
{
"id": 10,
"path": "(Acrocephaly)-[:IS_A]->(Craniosynostosis syndrome)
➥-[:IS_A]->(Congenital malformation)"
},
{
"id": 11,
"path": "(Acrocephaly)-[:PATHOLOGICAL_PROCESS_(ATTRIBUTE)]
➥->(Pathological developmental process)
➥<-[:PATHOLOGICAL_PROCESS_(ATTRIBUTE)]-(Congenital malformation)"
},
{
"id": 12,
"path": "(Acrocephaly)-[:OCCURRENCE]->(Congenital)
➥<-[:OCCURRENCE]-(Congenital malformation)"
```

},   
{   
"id": 13,   
"path": "(Acrocephaly)-[:OCCURRENCE]->(Congenital)   
➥<-[:OCCURRENCE]-([X]Other congenital malformations)"   
}   
]

이 JSON 출력에서 각 항목은 ID와 경로를 포함하며, 각 경로는 온톨로지 경로에서의 공출현 (co-occurrence)을 기반으로 생의학 엔터티 간의 관계적 연결을 보여 줍니다. 세부 사항은 다음과 같습니다.

선천성 지카 바이러스 감염 경로—많은 경로는 선천성 지카 바이러스 감염이 소두증, 첨두증, 다발성 선천 기형과 같은 다양한 선천성 질환에 [:OCCURRENCE] 관계 유형을 통해 연결되는 것으로 시작합니다. 이는 선천성 지카 바이러스 감염이 이러한 질환들과, 아마도 원인 또는 발생으로서, 연관되어 있음을 시사합니다.

공유된 선천성 질환—선천성은 소두증과 첨두증 같은 여러 선천성 질환을 연결하는 공통 노드입니다. 이 중심 노드는 이러한 질환들이 유사한 발생 속성을 공유한다는 것을 나타냅니다.

대안적 관계—일부 경로는 [:IS\_A] 및 [:PATHOLOGICAL\_PROCESS\_(ATTRIBUTE)]와 같은 관계를 사용하여 계층적 관계 또는 속성 기반 관계를 보여 줍니다. 예를 들어, 첨두증은 두개골조기유합증 증후군 아래에 분류되며, 이는 선천 기형과 연결됩니다.

#### 경로를 텍스트로 번역하기


이 단계는 그래프 경로를 문장으로 번역하며, 이를 통해 LLM이 복잡한 관계형 데이터를 자신들이 이해하도록 최적화된 형식인 자연어로 처리할 수 있습니다. 이러한 변환은 모델이 엔터티 간의 연결을 해석하기 쉽게 만들어 유사한 용어들을 구별하는 데 도움이 되는 맥락을 제공합니다. 다음은 경로를 문장으로 번역하기 위한 프롬프트의 단순화된 버전입니다.

#### 목록 8.17 경로를 문장으로 번역하기 위한 프롬프트의 단순화된 버전


```python
system = """You are an assistant capable of translating a Neo4j graph path
➥into a clear sentence.
Use the exact entity names from the path while generating the sentence.
The sentences will assist a large language model (LLM) in disambiguating
➥biomedical entities.
Ensure the output is a valid JSON with no extraneous characters."""
input = {
"path": "
➥(Hypertension)-[:RISK_FACTOR_FOR]->(Cardiovascular Disease)
➥<-[:ASSOCIATED_WITH]-(Myocardial Infarction)"
}
assistant = {
"sentence": "Hypertension is a risk factor for cardiovascular
➥disease. Myocardial infarction is also associated with cardiovascular
```

➥질환으로 이어지며, 이는 고혈압이 심혈관 질환과의 연결을 통해   
➥심근경색을 경험할 위험을 증가시킬 수 있음을 나타냅니다."   
}

프롬프트의 세부 사항은 다음과 같습니다.

 시스템 지시—시스템은 Neo4j 그래프 경로를 명확하고 사람이 읽을 수 있는 문장으로 번역합니다.

입력 그래프 경로—이 경우 입력은 복잡한 관계를 나타낼 수 있는 Neo4 데이터베이스의 그래프 경로로 구성됩니다.

 어시스턴트 출력—시스템은 생성된 문장을 보여 주는 유효한 JSON 구조로 응답합니다.

각 그래프 경로는 다음 목록에 표시된 것처럼 하나의 명확한 문장으로 번역됩니다.

#### 목록 8.18 경로를 자연어로 번역한 결과


```jsonl
[{
"sentence": "A Congenital Zika virus infection occurrence is associated
➥with a Congenital occurrence, which in turn is associated with
➥Micrencephaly."
},
{
"sentence": "A Congenital Zika virus infection occurrence is associated
➥with a Congenital occurrence, which in turn is associated with an
➥Acrocephaly occurrence."
},
[...],
{
"sentence": "Micrencephaly occurs in Congenital and Multiple congenital
➥malformations also occur in Congenital."
},
{
"sentence": "Micrencephaly occurs in Congenital and is also an occurrence
➥of Congenital malformation."
},
[...],
{
"sentence": "Acrocephaly occurs in Congenital and Other congenital
➥malformations also occur in Congenital."
}]
```

이러한 JSON 항목은 LLM이 그래프 경로를 해석하고 중의성 해소에 활용할 수 있도록 접근 가능하게 만듭니다.

#### 텍스트 경로 요약하기

선택된 후보들에 대한 최종 중의성 해소를 실행하기 전에, 번역된 그래프 경로를 나타내는 문장들을 요약해야 합니다. 이렇게 하면 모델의 “인지 부하”가 줄어들어(토큰 수 감소), 과도한 세부 정보에 압도되지 않고 엔터티 관계를 더 쉽게 해석하며 가장 정확한 후보를 선택할 수 있습니다. 다음 목록은 프롬프트의 단순화된 버전을 보여 줍니다.

```jsonl
Listing 8.19 Simplified prompt for summarizing textual paths —
system = """You are an assistant that can summarize multiple sentences
➥derived from ontology paths into a short summary. This summary will be
➥used to support a named entity disambiguation task.
Ensure the output is a valid JSON with no extraneous characters."""
input = {
"sentences": [
{
"id": 1,
"sentence": "Hypertension is a risk factor for cardiovascular
➥disease. Myocardial infarction is also associated with cardiovascular
➥disease, indicating that hypertension may increase the risk of
➥experiencing a myocardial infarction through its connection to
➥cardiovascular disease."
},{
"id": 2,
"sentence": "Diabetes mellitus is a complication that arises from
➥an endocrine disorder. Diabetic retinopathy is also associated with
➥endocrine disorders, suggesting that diabetes mellitus can lead to the
➥development of diabetic retinopathy through its link to endocrine
➥dysfunction."
},
{
"id": 3,
"sentence": "Asthma is associated with respiratory disorders.
➥Allergic rhinitis is also linked to respiratory disorders, which
➥implies that individuals with asthma may also experience allergic
➥rhinitis due to their common association with respiratory
➥conditions."
},
{
"id": 4,
"sentence": "Osteoporosis leads to bone weakness. Bone fractures
➥are a result of bone weakness, indicating that osteoporosis can
➥increase the likelihood of bone fractures due to the weakened state
➥of the bones."
}
]
assistant = {
"context": "Hypertension is a risk factor for cardiovascular disease,
➥which in turn increases the likelihood of experiencing a myocardial
➥infarction. Similarly, diabetes mellitus is linked to endocrine disorders,
➥potentially leading to complications such as diabetic retinopathy. Asthma
➥and allergic rhinitis are both associated with respiratory disorders,
➥suggesting a common link between these conditions. Finally, osteoporosis
➥weakens bones, making individuals more susceptible to bone fractures."
}
```

프롬프트의 세부 사항은 다음과 같습니다.

시스템 지시—시스템은 온톨로지 (ontology) 경로에서 파생된 문장들을 요약하되, 요약문에 식별된 모든 엔터티를 유지하도록 지시받습니다. 출력은 유효한 JSON 객체 형식이어야 하며, 각 요약은 context 키 아래에 문자열로 제공되어야 합니다.

입력 문장—입력은 여러 문장으로 구성되며, 각 문장은 의학적 상태와 그 효과 또는 연관성 사이의 관계를 포함합니다.

어시스턴트 출력—시스템은 관련 엔터티의 각 그룹에 대해 유효한 JSON 형식의 단일 요약 문장으로 응답합니다.

출력 구조는 입력 문장의 핵심 관계를 요약하며, 중요한 엔터티와 관계를 보존합니다(목록 8.20).

#### 목록 8.20 요약 단계의 결과


```jsonl
{"context": "A Congenital Zika virus infection occurrence is associated
➥with various congenital malformations, including Micrencephaly,
➥Acrocephaly, Multiple congenital malformations, and Other congenital
➥malformations. These conditions all share a common link to the Congenital
➥entity."}
```

이 결과는 LLM에 복잡한 관계 정보의 정제된 버전을 제공하여, 모호성 해소에 가장 관련성이 높은 맥락에 집중할 수 있게 합니다.

#### 모호성 해소

마지막 단계에서는 선택된 후보들과 요약 단계에서 제공된 텍스트 세부 정보를 포함하여 모호성 해소를 위한 모든 요소를 결합합니다. 다음 목록은 모호성 해소를 위한 프롬프트를 보여줍니다.

#### Listing 8.21 최종 모호성 해소를 위한 프롬프트


system = """당신은 엔터티 모호성 해소를 전문으로 하는 어시스턴트입니다.   
당신의 과제는 주어진 문장에서 언급된 엔터티를 식별하고 정확하게 모호성을 해소하는 것이며,   
➥주변 문장에 존재하는 맥락적 엔터티에 크게 의존해야 합니다.   
1. 원문 문장: 해결해야 하는 모호한 엔터티가 포함된 문장입니다.   
2. 후보 엔터티: 문장에서 추출된 잠재적 엔터티의 목록으로,   
각 엔터티는 여러 가능한 의미 또는 레이블을 가집니다.   
3. 맥락 문장: 언급된 엔터티의 모호성을 해소하기 위한 추가 맥락을 제공하는   
관련 문장 또는 주변 문장의 모음입니다.   
당신의 목표는 맥락 문장에서 언급된 엔터티를   
➥주요 정보원으로 사용하여 원문 문장에 있는 엔터티의   
➥모호성을 해소하는 것입니다. 각 모호한   
➥언급에 대해 후보 엔터티를 분석하고, 맥락 및   
➥맥락 문장이 제공하는 의미와 가장 잘 부합하는 것을 선택하십시오. 출력은 유효한   
➥JSON이어야 합니다."""

```json
input = {
"sentence": "Asthma and allergic rhinitis are commonly addressed
➥together in treatment protocols, given their shared underlying
➥inflammatory processes in allergic individuals.",
"candidates": [
{
"id": 1,
"candidates": [
{
"snomed_id": "233681001",
"name": "Extrinsic asthma with asthma attack"
},
{
"snomed_id": "195967001",
"name": "Asthma"
},
{
"snomed_id": "266361008",
"name": "Intrinsic asthma"
},
{
"snomed_id": "266364000",
"name": "Asthma attack"
},
{
"snomed_id": "270442000",
"name": "Asthma monitored"
},
{
"snomed_id": "170642006",
"name": "Asthma severity"
},
{
"snomed_id": "170643001",
"name": "Occasional asthma"
},
{
"snomed_id": "170644007",
"name": "Mild asthma"
},
{
"snomed_id": "170645008",
"name": "Moderate asthma"
}
]
}
],
"context":"Asthma is associated with respiratory disorders. Allergic
➥rhinitis is also linked to respiratory disorders, which implies that
➥individuals with asthma may also experience allergic rhinitis due to their
➥common association with respiratory conditions."}
}
```

```json
assistant = {
"entities": [
{
"id": 1,
"disambiguation": {
"snomed_id": "195967001",
"name": "Asthma"
}
},
{
"id": 2,
"disambiguation": {
"snomed_id": "61582004",
"name": "Allergic rhinitis"
}
}
]
}
```

#### 프롬프트의 세부 사항은 다음과 같습니다:


시스템 지시—시스템은 모호한 엔터티를 식별하고 정확하게 모호성을 해소하도록 지시됩니다. 원문 문장의 각 엔터티를 분석하고, 요약된 맥락 정보가 제공하는 맥락적 의미와 가장 잘 부합하는 후보를 선택해야 하며, 이러한 맥락적 세부 사항과 부합하는 엔터티를 우선해야 합니다.

입력 구조:

– 원문 문장—기본 문장에는 모호성 해소가 필요한 모호한 엔터티가 포함되어 있습니다.

– 후보 엔터티—각 언급에 대한 잠재적 SNOMED 엔터티 목록으로, 각각 여러 가능한 해석 또는 라벨을 가집니다.

– 맥락 문장—원문 문장에 있는 각 모호한 엔터티의 의미를 명확히 하는 데 도움이 되는 맥락을 제공하는 추가 문장입니다.

어시스턴트 출력:

– id—엔터티 언급의 고유 식별자

– 모호성 해소—선택된 SNOMED 엔터티를 포함하는 객체로, 맥락 정보와 가장 잘 일치하는 snomed\_id 및 name을 포함합니다.

우리는 LLM에 입력 문장 “지카의 중증 결과는 임신 중 태반 장벽을 통과하여 소두증과 선천성 기형을 유발하는 능력 때문입니다.”를 전달하여 다음 결과를 얻었습니다.

#### 목록 8.22 모호성 해소 과정의 결과


{   
"entities": [   
{   
"id": 0,

"disambiguation": {   
"snomed\_id": "762725007",   
"name": "Congenital Zika virus infection"   
}   
},   
{   
"id": 1,   
"disambiguation": {   
"snomed\_id": "204030002",   
"name": "Micrencephaly"   
}   
},   
{   
"id": 2,   
"disambiguation": {   
"snomed\_id": "116022009",   
"name": "Multiple congenital malformations"   
}   
}   
]   
}

각 엔터티는 SNOMED 온톨로지에서 가장 관련성이 높은 개념과 매칭되며, 이를 통해 맥락 정보를 바탕으로 정확한 식별과 분류가 가능합니다.

### 8.5 결론


이 장에서는 개방형 LLM과 도메인 특화 온톨로지를 사용하는 개체명 중의성 해소 (named entity disambiguation, NED) 시스템을 심층적으로 탐구했습니다. SNOMED와 같은 도메인 온톨로지를 Llama 3.1 8B와 같은 개방형 범용 LLM과 통합함으로써, scispaCy와 같은 생의학 도메인의 전통적인 NLP 도구가 지닌 한계 중 일부를 해결할 수 있습니다. 우리의 유연한 접근 방식은 여러 응용 도메인에 걸쳐 적응할 수 있습니다. 즉, Neo4j의 GDS 라이브러리를 활용한 최단 경로 탐지 및 전문 검색을 LLM의 중의성 해소 능력과 결합함으로써, 복잡한 텍스트에서 엔터티를 식별하고 정확하게 중의성을 해소하는 견고한 시스템을 구현할 수 있습니다. 경로-텍스트 변환 (path-to-text translation) 및 텍스트 기반 경로 요약 (textual path summarization)과 같은 기법을 통해, 우리는 LLM이 관계형 데이터를 자연어 형식으로 처리하는 능력을 개선했으며, 이로써 유사한 엔터티를 구별하는 역량을 강화했습니다. 이 프레임워크는 도메인 특화 NED 과제에서 LLM을 활용하는 미래 응용의 토대를 마련합니다.

#### 요약


명명 엔터티 명확화 (Named entity disambiguation)는 복잡한 도메인에서 엔터티를 정확하게 식별하고 구별하는 데 필수적입니다.

scispaCy와 같은 전통적인 NLP 도구는 다양한 도메인에서 사용될 수 없고 엔터티 간 관계를 활용할 수 없으며, 참조 지식을 확장하고 업데이트할 수도 없습니다.

범용 LLM과 도메인 온톨로지 (domain ontology)를 결합하면 이러한 문제를 해결할 수 있습니다. LLM은 온톨로지에 통합된 지속적으로 업데이트되는 지식에 의해 구동될 수 있으며, 그 관계형 구조를 활용할 수 있습니다.

우리는 LLM과 도메인 온톨로지가 관여하는 여러 단계를 포함하여, NED를 위한 유연한 종단 간 프로세스를 배포했습니다.

도메인 온톨로지의 그래프 차원과 결합된 LLM의 역량을 최대한 활용하기 위해, 명확화는 최단 경로 탐지, 경로의 텍스트 변환, 텍스트 경로 요약이라는 세 단계로 나뉩니다.

향후 NED 응용은 이 프레임워크를 사용하고, 엔터티의 관계적 본질을 설명하는 풍부한 온톨로지를 특징으로 하는 다른 도메인에 적응할 수 있습니다.