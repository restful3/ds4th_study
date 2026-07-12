# Disambiguation Flashcards

## Card 1

**Front:** Named Entity Disambiguation (NED)의 주요 역할은 무엇입니까?

**Back:** 텍스트 내의 언급(mention)을 지식 베이스의 정형화된 개체와 연결하여 모호성을 해소하는 것입니다.

---

## Card 2

**Front:** scispaCy는 어떤 오픈 소스 NLP 프레임워크를 기반으로 구축되었습니까?

**Back:** spaCy 프레임워크를 기반으로 구축되었습니다.

---

## Card 3

**Front:** scispaCy가 특화되어 설계된 주요 응용 도메인은 무엇입니까?

**Back:** 생물 의학(Biomedical) 도메인입니다.

---

## Card 4

**Front:** scispaCy가 개체 중의성 해소를 위해 참조하는 대표적인 통합 의료 언어 시스템은?

**Back:** UMLS (Unified Medical Language System)입니다.

---

## Card 5

**Front:** 전통적인 NED 도구인 scispaCy의 주요 한계 중 하나는 특정 _____에만 설계되었다는 점입니다.

**Back:** 응용 도메인 (생물 의학 분야)

---

## Card 6

**Front:** 전통적인 NED 시스템은 지식 베이스에 새로운 용어를 추가하거나 _____하는 데 어려움이 있습니다.

**Back:** 업데이트 (Updating)

---

## Card 7

**Front:** 전통적인 NED 툴은 지식 베이스 내의 개체 간 _____와 경로 정보를 충분히 활용하지 못한다는 단점이 있습니다.

**Back:** 기존 관계 (Existing relationships)

---

## Card 8

**Front:** 본문에서 제시된 'Zika' 사례에서 scispaCy가 중의성 해소에 실패한 이유는 무엇입니까?

**Back:** 'congenital'이나 'syndrome'과 같이 해소를 지원하는 주변 맥락 단어가 부족했기 때문입니다.

---

## Card 9

**Front:** SNOMED (Systematized Nomenclature of Medicine) 온톨로지는 약 몇 개의 개념을 포함하고 있습니까?

**Back:** 450,000개 이상의 개념을 포함합니다.

---

## Card 10

**Front:** SNOMED 데이터 중 개체 이름, 별칭, 관계를 포함하는 텍스트 파일의 명칭은?

**Back:** sct2_Description_Full-en_US1000124_20220901.txt

---

## Card 11

**Front:** SNOMED 데이터에서 개체와 관계를 식별하는 수치 코드를 담고 있는 파일은?

**Back:** sct2_Relationship_Full_US1000124_20220901.txt

---

## Card 12

**Front:** SNOMED 계층 구조에서 정보가 전파되는 방향은 어떠합니까?

**Back:** 첫 번째 수준 노드(First-level nodes)에서 더 깊은 노드(Deeper nodes)로 정보가 전파됩니다.

---

## Card 13

**Front:** SNOMED 계층 구조의 최상위에서 아키타입 개체를 나타내는 노드를 무엇이라 부릅니까?

**Back:** 첫 번째 수준 노드 (First-level nodes)

---

## Card 14

**Front:** 로컬 머신에서 LLM을 직접 실행할 수 있게 해주는 오픈 소스 도구는 무엇입니까?

**Back:** Ollama

---

## Card 15

**Front:** Meta에서 출시한 Llama 3.1 8B 모델의 파라미터 수는 몇 개입니까?

**Back:** 80억 개 (8 billion)

---

## Card 16

**Front:** Llama 3.1 8B 모델이 지원하는 최대 컨텍스트 길이는 얼마입니까?

**Back:** 128,000 토큰

---

## Card 17

**Front:** LLM을 로컬에서 실행할 때 얻을 수 있는 데이터 보안상의 이점은?

**Back:** 사용자의 데이터에 대한 완전한 제어권을 가질 수 있습니다.

---

## Card 18

**Front:** Ollama는 어떤 표준 API와 내장 호환성을 제공하여 모델 통합을 단순화합니까?

**Back:** OpenAI Chat Completions API

---

## Card 19

**Front:** 본문에서 제안하는 NED 프로세스의 3가지 주요 단계는 무엇입니까?

**Back:** NER(명명된 개체 인식), CS(후보 선택), CD(후보 중의성 해소)입니다.

---

## Card 20

**Front:** NER(Named Entity Recognition)의 궁극적인 목표는 무엇입니까?

**Back:** 비정형 텍스트에서 개체를 식별하고 질병, 유기체 등의 사전 정의된 범주로 분류하는 것입니다.

---

## Card 21

**Front:** NER 단계에서 개체 유형을 정의하기 위해 주로 누구와 협업합니까?

**Back:** 데이터 과학자 또는 데이터 엔지니어가 도메인 전문가와 협업합니다.

---

## Card 22

**Front:** 본문의 NER 프롬프트에서 시스템 메시지가 지시하는 추출 대상은 어디에 속해야 합니까?

**Back:** SNOMED 온톨로지의 의료 도메인 범주

---

## Card 23

**Front:** NER 단계의 결과로 출력되는 구조화된 JSON 데이터의 주요 필드 3가지는?

**Back:** sentence(문장), entities(개체 배열), label(범주 레이블)입니다.

---

## Card 24

**Front:** LLM이 NER 수행 시 전통적인 시스템에 비해 정확하게 탐지하기 어려워하는 정보는?

**Back:** 개체 언급의 시작(start) 및 종료(end) 문자 위치

---

## Card 25

**Front:** 후보 선택(Candidate Selection, CS) 단계의 입력값 두 가지는?

**Back:** NER 단계에서 주석 처리된 언급(mention)들과 도메인 온톨로지입니다.

---

## Card 26

**Front:** CS 단계에서 LLM을 직접 사용하지 않고 Neo4j 검색을 사용하는 이유는?

**Back:** 온톨로지에서 직접 후보를 추출해야 하며, 온톨로지의 크기가 너무 커서 프롬프트에 담을 수 없기 때문입니다.

---

## Card 27

**Front:** Neo4j에서 언급된 문자열과 유사한 온톨로지 내 문자열을 찾기 위해 사용하는 기능은?

**Back:** 전체 텍스트 검색 (Full-text search)

---

## Card 28

**Front:** CS 단계의 결과로 각 언급에 대해 제공되는 정보는 무엇입니까?

**Back:** snomed_id와 해당 의료 개체의 이름(name)을 포함한 후보 목록입니다.

---

## Card 29

**Front:** 후보 중의성 해소(Candidate Disambiguation, CD) 단계에서 활용하는 핵심 정보는?

**Back:** 한 문장 내에서 타겟 개체와 함께 나타나는 다른 의료 개체들의 문맥 정보입니다.

---

## Card 30

**Front:** CD 단계의 세부 3단계 프로세스는 무엇입니까?

**Back:** 최단 경로 탐색, 경로의 텍스트 변환, 텍스트 경로 요약입니다.

---

## Card 31

**Front:** 최단 경로 탐색 단계에서 사용하는 Neo4j의 라이브러리 명칭은?

**Back:** GDS (Graph Data Science) 라이브러리

---

## Card 32

**Front:** 최단 경로 탐색 쿼리에서 '허브 노드(hub nodes)'를 제외하는 이유는 무엇입니까?

**Back:** 지나치게 일반적이거나 광범위한 연결을 배제하고 의미 있는 관계에 집중하기 위해서입니다.

---

## Card 33

**Front:** 허브 노드를 식별하기 위해 쿼리에서 먼저 계산하는 지표는 무엇입니까?

**Back:** 노드의 연결 수인 '차수(degree)'입니다.

---

## Card 34

**Front:** 본문의 최단 경로 탐색 시 허용되는 최대 홉(hop) 수는?

**Back:** 2홉 (relationships) 이내

---

## Card 35

**Front:** 그래프 경로를 자연어 문장으로 변환하는 'Path-to-text translation'의 이점은?

**Back:** LLM이 자연어 처리 능력에 최적화되어 있어 복잡한 관계 데이터를 더 잘 해석할 수 있게 합니다.

---

## Card 36

**Front:** Path-to-text 단계의 입력값은 무엇입니까?

**Back:** Neo4j 데이터베이스에서 추출된 그래프 경로(path)입니다.

---

## Card 37

**Front:** 텍스트 경로 요약(Summarizing textual paths) 단계가 필요한 이유는 무엇입니까?

**Back:** 토큰 수를 줄여 모델의 인지 부하를 낮추고 핵심 관계에 집중하게 하기 위함입니다.

---

## Card 38

**Front:** 요약 단계의 결과물은 어떤 JSON 키(key) 아래에 문자열로 제공됩니까?

**Back:** context

---

## Card 39

**Front:** 최종 중의성 해소 단계에서 LLM이 가장 우선순위를 두어 분석해야 하는 정보는?

**Back:** 요약된 문맥 정보와 일치하는 후보 개체

---

## Card 40

**Front:** 최종 Disambiguation 출력에서 각 언급을 식별하기 위해 사용하는 고유 식별자는?

**Back:** id

---

## Card 41

**Front:** 본문에서 'Zika'가 'Microcephaly'와 함께 등장할 때 어떤 개체로 중의성이 해소됩니까?

**Back:** Congenital Zika virus infection (선천성 지카 바이러스 감염)

---

## Card 42

**Front:** SNOMED 온톨로지에서 'is_a' 관계는 주로 어떤 유형의 관계를 나타냅니까?

**Back:** 계층적(Hierarchical) 관계

---

## Card 43

**Front:** Neo4j에서 snomed_id에 대해 고유성을 보장하기 위해 설정하는 것은?

**Back:** CREATE CONSTRAINT (UNIQUE)

---

## Card 44

**Front:** 온톨로지 검색 성능 향상을 위해 'n.name' 속성에 생성하는 것은?

**Back:** CREATE INDEX

---

## Card 45

**Front:** Ollama에서 Llama 3.1 모델을 로컬로 가져오는 명령어는?

**Back:** ollama pull llama3.1:latest

---

## Card 46

**Front:** 본문에서 사용한 Llama 3.1 8B 모델은 어떤 하드웨어 환경을 타겟으로 설계되었습니까?

**Back:** 소비자용 등급 하드웨어 (Consumer-grade hardware)

---

## Card 47

**Front:** LLM 기반 NER 접근법에서 개체 유형(category)은 어디서 가져옵니까?

**Back:** 온톨로지(SNOMED)에서 사전 정의된 범주를 검색하여 가져옵니다.

---

## Card 48

**Front:** Candidate Selection 단계에서 Neo4j의 어떤 함수를 사용하여 문자열을 결합합니까?

**Back:** apoc.text.join

---

## Card 49

**Front:** SNOMED의 첫 번째 수준 노드 중 질병 정보를 전파하는 노드 예시는?

**Back:** Disease

---

## Card 50

**Front:** SNOMED에서 약품 관련 정보를 담당하는 노드의 이름은?

**Back:** Pharmaceutical product

---

## Card 51

**Front:** 전통적 툴과 달리 본문의 접근법이 다른 응용 도메인으로 쉽게 확장 가능한 이유는?

**Back:** 특정 도메인에 고정되지 않고 풍부한 온톨로지만 있으면 범용 LLM을 사용할 수 있기 때문입니다.

---

## Card 52

**Front:** NER 결과물에서 'mention' 필드가 의미하는 바는 무엇입니까?

**Back:** 텍스트에서 실제로 발견된 명명된 개체의 텍스트 문자열입니다.

---

## Card 53

**Front:** CS 단계의 결과로 얻은 후보들 중 중의성 해소의 대상이 되는 개체를 무엇이라 합니까?

**Back:** 대상 개체 (Target entity)

---

## Card 54

**Front:** CD 프로세스 중 'Shortest path search' 쿼리에서 unwind의 역할은?

**Back:** 식별된 경로들을 개별적으로 풀어내어 노드와 관계를 수집할 수 있게 합니다.

---

## Card 55

**Front:** 경로 번역(Path-to-text) 프롬프트의 시스템 지침은 무엇을 요구합니까?

**Back:** 그래프 경로를 명확하고 사람이 읽기 쉬운 문장으로 번역할 것을 요구합니다.

---

## Card 56

**Front:** Summarization 단계에서 시스템은 요약본에 반드시 무엇을 유지해야 합니까?

**Back:** 문장에서 식별된 모든 개체 (Identified entities)

---

## Card 57

**Front:** 최종 중의성 해소 단계의 출력 필드 중 선택된 SNOMED 개체 정보를 담는 객체는?

**Back:** disambiguation

---

## Card 58

**Front:** 본문의 접근법에서 온톨로지는 프로세스의 어느 단계에 통합됩니까?

**Back:** 중의성 해소의 모든 단계 (NER, CS, CD)에 통합됩니다.

---

## Card 59

**Front:** scispaCy가 'Zika'를 탐지할 때 사용한 개체 코드 'C0276289'는 무엇을 의미합니까?

**Back:** Zika Virus Infection

---

## Card 60

**Front:** Ollama를 통해 모델을 배포한 후 로컬에서 서버가 돌아가는 기본 URL은?

**Back:** http://localhost:11434 (또는 Default URL)

---

## Card 61

**Front:** Llama 3.1 8B가 다국어 정보 처리에 최적화되어 있다는 점이 주는 이점은?

**Back:** SNOMED와 같은 다국어 임상 용어 저장소를 처리하는 데 유리합니다.

---

## Card 62

**Front:** NER에서 'sentence' 단위로 텍스트를 처리하는 이유는 무엇입니까?

**Back:** 의미의 개별 단위를 분석하여 정확한 엔터티 식별을 보장하기 위해서입니다.

---

## Card 63

**Front:** CS 단계에서 'snomed_id'는 어떤 역할을 합니까?

**Back:** SNOMED 개념을 고유하게 식별하는 수치 코드 역할을 합니다.

---

## Card 64

**Front:** CD 단계에서 'Acrocephaly'와 'Congenital malformation'을 잇는 관계 유형의 예시는?

**Back:** [:IS_A]

---

## Card 65

**Front:** 최단 경로 탐색 시 'hub_nodes' 필터링에 사용되는 WHERE 절의 조건은?

**Back:** 경로 내 노드 이름이 허브 노드 목록에 포함되지 않아야 함 (not any)

---

## Card 66

**Front:** Path-to-text 단계의 출력이 JSON 형식이어야 하는 이유는?

**Back:** 다음 단계인 요약 및 중의성 해소 단계에서 프로그램적으로 쉽게 파싱하기 위해서입니다.

---

## Card 67

**Front:** 요약된 문맥(context)은 최종적으로 LLM이 무엇을 결정하는 데 도움을 줍니까?

**Back:** 여러 후보 중 실제 의미에 가장 부합하는 후보를 선택하는 결정

---

## Card 68

**Front:** 본문에서 언급된 'cognitive load'를 줄이는 방법은?

**Back:** 텍스트 경로 요약을 통해 토큰 수를 줄이는 것

---

## Card 69

**Front:** NED 시스템 설계 시 범용 LLM(Llama 3.1)과 도메인 온톨로지를 결합했을 때의 핵심 가치는?

**Back:** LLM의 일반적 추론 능력과 온톨로지의 정밀한 도메인 지식을 동시에 활용하는 것

---

## Card 70

**Front:** Neo4j의 GDS 라이브러리를 활용한 학문적/기술적 영역은?

**Back:** 그래프 데이터 과학 (Graph Data Science)

---

## Card 71

**Front:** SNOMED의 'Relationship' 파일에서 관리하는 주요 정보는?

**Back:** 개체 간의 수치 코드로 정의된 연결 및 관계 유형

---

## Card 72

**Front:** LLM 기반 NER에서 'label'을 할당할 때의 기준은?

**Back:** 해당 개체가 속한 SNOMED 온톨로지 카테고리

---

## Card 73

**Front:** CS 쿼리에서 'ORDER BY'와 'LIMIT'을 사용하는 목적은?

**Back:** 유사도가 높은 상위 후보들만을 선택하여 중의성 해소의 효율성을 높이기 위함

---

## Card 74

**Front:** CD의 최종 단계에서 입력되는 'Original sentence'의 역할은?

**Back:** 중의성 해소가 필요한 대상 단어가 포함된 원본 맥락 제공

---

## Card 75

**Front:** Path-to-text 번역 결과의 예시 문장 구조는?

**Back:** 개체 A는 관계 R을 통해 개체 B와 연결된다는 식의 자연어 문장

---

## Card 76

**Front:** 본문의 시스템에서 'Zika'라는 용어가 'Organism' 레이블을 받는 단계는?

**Back:** NER (Named Entity Recognition) 단계

---

## Card 77

**Front:** SNOMED 온톨로지가 'multilingual' 임상 용어 저장소라는 것의 의미는?

**Back:** 여러 언어로 된 임상 용어와 개념을 포함하고 있다는 뜻입니다.

---

## Card 78

**Front:** 본문 시스템에서 Neo4j는 주로 어떤 데이터를 저장하고 쿼리합니까?

**Back:** SNOMED 온톨로지의 그래프 구조와 개체 데이터

---

## Card 79

**Front:** NED 프로세스 전체를 아우르는 mental model의 시작점은?

**Back:** 비정형 텍스트로 구성된 입력 문서 (Input document)

---

## Card 80

**Front:** 본문에 따르면, 미래의 NED 응용 프로그램은 무엇을 통해 다른 도메인에 적응할 수 있습니까?

**Back:** 관계적 특성을 설명하는 풍부한 온톨로지를 활용하는 프레임워크

---
