# Biomedical Flashcards

## Card 1

**Front:** 자연어 처리(NLP)에서 개체명 인식(NER)의 주된 목적은 무엇인가요?

**Back:** 텍스트에서 관련 개체명을 식별하고 사람, 조직, 장소, 질병 등 사전 정의된 범주에 할당하는 것입니다.

---

## Card 2

**Front:** NER만으로는 텍스트를 정확하게 이해하기 부족한 이유는 무엇인가요?

**Back:** NER은 개체를 분류할 뿐, 특정 문맥에서 해당 개체가 가리키는 고유한 의미나 지식 베이스와의 연결을 제공하지 않기 때문입니다.

---

## Card 3

**Front:** 개체명 모호성 해소(NED)의 궁극적인 목표는 무엇인가요?

**Back:** 문맥을 검토하여 엔티티 언급의 모호성을 제거하고, 이를 지식 베이스(Knowledge Base) 내의 특정 엔티티에 연결하는 것입니다.

---

## Card 4

**Front:** 지능형 자문 시스템(IAS)의 핵심 속성 중 하나로, 인간과 여러 상호작용을 통해 정보를 교환하는 능력은?

**Back:** 상호작용성(Interactivity)

---

## Card 5

**Front:** IAS가 사용자에게 유용한 정보를 제공하기 위해 필요한 두 가지 NLP 기능은?

**Back:** 자연어에서 의미 있는 엔티티 탐지 및 다양한 지식 소스로부터 해당 엔티티에 대한 정보 검색입니다.

---

## Card 6

**Front:** 동일한 단어 'Zika'가 문맥에 따라 바이러스 또는 질병을 의미할 때, 이를 구분해 내는 작업을 무엇이라 하나요?

**Back:** 개체명 모호성 해소(Named Entity Disambiguation, NED)

---

## Card 7

**Front:** 의료 도메인과 같은 비즈니스 환경에서 정확한 NED가 필수적인 이유는 무엇인가요?

**Back:** 일반적인 질병과 선천적 형태의 질병을 구분하는 등 미세한 차이가 결정적인 의사결정에 영향을 미치기 때문입니다.

---

## Card 8

**Front:** NED 시스템의 주요 3단계 프로세스는 무엇인가요?

**Back:** 후보 선택(Candidate selection), 후보 순위 지정(Candidate ranking), 온톨로지 통합(Ontology integration)입니다.

---

## Card 9

**Front:** NED의 1단계인 '후보 선택(Candidate selection)'의 역할은?

**Back:** 텍스트 언급과 연관될 가능성이 있는 지식 베이스 내의 타당한 엔티티 집합을 식별하는 것입니다.

---

## Card 10

**Front:** NED의 2단계인 '후보 순위 지정(Candidate ranking)'은 어떤 기준으로 수행되나요?

**Back:** 인식된 개체 주변의 단어(문맥 정보)를 기반으로 각 후보에 점수를 할당하여 가장 높은 점수의 타겟 엔티티를 선택합니다.

---

## Card 11

**Front:** NED의 3단계인 '온톨로지 통합(Ontology integration)'의 목적은 무엇인가요?

**Back:** 여러 온톨로지로부터 정보를 집계하여 타겟 엔티티의 표현을 풍부하게 만드는 것입니다.

---

## Card 12

**Front:** 의료 분야 엔티티의 모호성 해소를 위해 사용할 수 있는 대표적인 Python 라이브러리는?

**Back:** scispaCy

---

## Card 13

**Front:** 지식 베이스(Knowledge Base)가 엔티티 연결 작업에서 수행하는 역할은 무엇인가요?

**Back:** 특정 도메인 내 엔티티들에 대한 구조화된 표현을 수집하고 저장하는 중앙 역할을 합니다.

---

## Card 14

**Front:** UMLS(Unified Medical Language System)는 무엇을 제공하는 자원인가요?

**Back:** 여러 소스로부터 생물 의학 용어, 분류 및 코딩 표준을 제공하여 상호 운용 가능한 정보 시스템 구축을 돕습니다.

---

## Card 15

**Front:** UMLS 파일 중 여러 어휘집의 엔티티 목록과 ID를 포함하는 파일명은?

**Back:** $MRCONSO.RRF$

---

## Card 16

**Front:** UMLS 엔티티들을 범주화하는 의미론적 유형(Semantic Types) 목록이 담긴 파일은?

**Back:** $MRSTY.RRF$

---

## Card 17

**Front:** UMLS의 데이터 포맷에서 각 값들을 구분하는 데 주로 사용되는 기호는?

**Back:** 세로줄 또는 파이프($|$)

---

## Card 18

**Front:** NED 모델의 출력이 지식 그래프(KG) 구축에서 '진입점' 역할을 한다는 의미는?

**Back:** 추출된 엔티티를 통해 외부 온톨로지의 경로를 탐색하고 비구조적 지식과 구조적 지식을 연결할 수 있음을 뜻합니다.

---

## Card 19

**Front:** 지식 그래프를 위한 CRISP-DM 모델의 첫 번째 단계인 '비즈니스 이해'의 목표는?

**Back:** 정책 담당자의 의사결정을 돕기 위해 비구조적 콘텐츠에 대한 접근 및 분석을 용이하게 하는 것입니다.

---

## Card 20

**Front:** 데이터 준비(Data preparation) 단계가 KG 구축 과정에서 중요한 이유는?

**Back:** 문서 처리, 개체명 모호성 해소, 온톨로지 처리 등을 통해 데이터를 '수입(ingestion)' 가능한 형태로 만들기 때문입니다.

---

## Card 21

**Front:** 키워드가 아닌 '의미'를 기반으로 정보를 찾는 검색 방법은?

**Back:** 개념 검색(Conceptual search)

---

## Card 22

**Front:** 개념 검색이 키워드 검색보다 우수한 점은 무엇인가요?

**Back:** 서로 다른 표현(예: '췌장 섬'과 '랑게르한스 섬')이 동일한 엔티티를 가리킴을 인식하거나, 이름은 비슷하지만 의미가 다른 엔티티를 구분할 수 있습니다.

---

## Card 23

**Front:** 도메인 온톨로지에 구조화된 공식 지식을 사용하여 텍스트 정보를 검색하는 방식은?

**Back:** 구조화된 지식 기반 검색(Structured knowledge-based search)

---

## Card 24

**Front:** 온톨로지 경로 탐색을 통해 얻을 수 있는 분석적 이점은?

**Back:** 특정 질환으로 인해 발생하는 다양한 장애를 식별하고, 이를 언급하는 모든 문서를 찾아 통합된 시야를 확보할 수 있습니다.

---

## Card 25

**Front:** ChatGPT와 같은 거대언어모델(LLM)이 UMLS ID 할당 작업에서 보이는 한계는?

**Back:** UMLS와 같은 외부 지식 베이스가 모델 내부에 통합되어 있지 않아 정확한 고유 식별자(ID)를 매핑하는 데 전문성이 부족합니다.

---

## Card 26

**Front:** 지식 그래프 스키마에서 'EntityMention' 노드는 무엇을 나타내나요?

**Back:** 텍스트 내에서 실제로 언급된 엔티티의 문자열과 그 위치 정보를 나타냅니다.

---

## Card 27

**Front:** 지식 그래프 스키마에서 'MedicalEntity' 노드와 'EntityMention' 노드를 분리하여 설계하는 이유는?

**Back:** 동일한 문자열이 다른 엔티티를 가리키거나, 다른 문자열이 동일한 엔티티를 가리키는 경우를 유연하게 표현하기 위함입니다.

---

## Card 28

**Front:** 온톨로지 로딩 및 매핑 시 UMLS가 수행하는 전략적 역할은 무엇인가요?

**Back:** 여러 특정 온톨로지(SNOMED, HPO 등)의 정보에 접근하기 위한 진입점(Entry point) 역할을 합니다.

---

## Card 29

**Front:** 동일한 문장 내에 두 엔티티가 함께 나타나는 관계를 무엇이라 하나요?

**Back:** 공동 출현(Co-occurrence) 관계

---

## Card 30

**Front:** SNOMED 온톨로지에서 'Zika virus disease'와 'Zika virus' 사이의 관계 유형은?

**Back:** $CAUSATIVE\_AGENT$ (원인체)

---

## Card 31

**Front:** 지식 그래프 분석 시 경로 탐색 결과가 너무 일반적인 엔티티로 흐르는 것을 방지하기 위해 제외해야 하는 노드는?

**Back:** 허브 노드(Hub nodes)

---

## Card 32

**Front:** 허브 노드(Hub nodes)를 식별하기 위해 사용하는 그래프 알고리즘은?

**Back:** $gds.degree.stream$ (차수 중심성 측정)

---

## Card 33

**Front:** 텍스트에 아직 명시적으로 정의되지 않은 새로운 패턴을 도메인 온톨로지에 제안하는 유스케이스는?

**Back:** 새로운 지식의 발견(Uncovering new knowledge)

---

## Card 34

**Front:** UMLS ID $C0011311$은 어떤 질병 엔티티를 식별하나요?

**Back:** 뎅기열(Dengue fever)

---

## Card 35

**Front:** NED 모델이 엔티티의 텍스트 내 위치를 식별함으로써 검색 결과에 기여하는 '설명 가능성' 요소는?

**Back:** 엔티티가 언급된 정확한 텍스트 부분(snippet)과 페이지 내 발생 횟수를 제공할 수 있습니다.

---

## Card 36

**Front:** 스키마 내 $HPO\_IS\_A$ 관계는 어떤 온톨로지의 계층 구조를 모델링하나요?

**Back:** 인간 표현형 온톨로지(Human Phenotype Ontology, HPO)

---

## Card 37

**Front:** 지식 그래프에서 'Page' 노드와 'File' 노드를 연결하는 관계 이름은?

**Back:** $CONTAINS\_PAGE$

---

## Card 38

**Front:** 온톨로지 통합 단계에서 'MedicalEntity' 노드에 추가되는 정보의 예는?

**Back:** UMLS 의미론적 유형(Semantic Type) 및 관련 온톨로지 코드 등입니다.

---

## Card 39

**Front:** NED 프로세스 중 문맥 정보를 이용해 엔티티의 불확실성을 수치화하는 단계는?

**Back:** 후보 순위 지정(Candidate ranking)

---

## Card 40

**Front:** 지식 그래프 구축 시 '스키마 정의' 단계가 갖는 의미는?

**Back:** 데이터의 주요 구성 요소(노드 레이블 및 관계 유형)를 정의하는 이론적 모델링 단계입니다.

---

## Card 41

**Front:** UMLS Metathesaurus에서 제공하는 '정규화된 이름(Normalized names)'의 용도는?

**Back:** 다양한 표현의 용어들을 표준화된 하나의 개념으로 매핑하기 위해 사용됩니다.

---

## Card 42

**Front:** NED와 KG 기술을 결합하여 얻을 수 있는 가장 큰 기회는 무엇인가요?

**Back:** 비구조적 데이터로부터 구조화된 지식을 추출하고 통합하여 고급 분석 서비스를 개발할 수 있다는 점입니다.

---

## Card 43

**Front:** SNOMED 온톨로지에서 '뎅기열'이 '플라비바이러스에 의한 질병'임을 나타내는 관계는?

**Back:** $IS\_A$

---

## Card 44

**Front:** 경로 탐색 쿼리에서 $allShortestPaths$ 함수를 사용하는 이유는 무엇인가요?

**Back:** 두 엔티티 사이를 연결하는 가장 짧고 효율적인 관계망을 찾기 위해서입니다.

---

## Card 45

**Front:** 지식 그래프에서 공동 출현(Co-occur) 관계가 생성되는 단위는?

**Back:** 동일한 문장(Sentence)

---

## Card 46

**Front:** 데이터 세트에서 'Islets of Langerhans'에 할당된 SNOMED 코드는?

**Back:** $3928002$

---

## Card 47

**Front:** UMLS의 $MRSTY.RRF$ 파일에서 각 엔티티 ID에 매핑되는 핵심 정보는?

**Back:** 해당 엔티티가 속한 의미론적 범주(TUI 등)입니다.

---

## Card 48

**Front:** NED 모델이 지식 베이스를 참조할 때, 참조되는 엔티티를 무엇이라 부르나요?

**Back:** 그라운드 엔티티(Ground entity)

---

## Card 49

**Front:** NED 시스템 아키텍처에서 '도메인 온톨로지'는 어느 단계에 입력되나요?

**Back:** 온톨로지 통합(Ontology integration) 단계

---

## Card 50

**Front:** 지식 그래프 내에서 파일 경로와 페이지 인덱스를 함께 저장할 때의 이점은?

**Back:** 사용자가 검색된 정보의 출처를 정확히 추적하고 검증할 수 있게 합니다.

---

## Card 51

**Front:** scispaCy 모델이 후보를 선택할 때 사용하는 기반 정보는?

**Back:** 참조 지식 베이스(Reference Knowledge Base)

---

## Card 52

**Front:** 지식 그래프에서 'MedicalEntity'들 사이의 $COOCCUR$ 관계는 무엇을 기반으로 생성되나요?

**Back:** 동일한 문서 페이지의 동일한 문장에서 두 엔티티가 동시에 언급되었는지 여부입니다.

---

## Card 53

**Front:** NED가 수동으로 수행될 때 발생하는 문제점은 무엇인가요?

**Back:** 문서의 양이 증가함에 따라 도메인 전문가가 일일이 처리하는 것이 비현실적이고 비효율적이 됩니다.

---

## Card 54

**Front:** UMLS Metathesaurus가 포함하는 'Source ID'의 역할은?

**Back:** 특정 용어가 원래 어떤 외부 온톨로지에서 유래했는지 식별합니다.

---

## Card 55

**Front:** NED 결과물에서 'Target Entity'는 무엇을 의미하나요?

**Back:** 후보들 중 순위 지정 단계를 거쳐 최종적으로 선택된 정답 엔티티입니다.

---

## Card 56

**Front:** 구조화된 지식 기반 검색에서 '비자명한 관계(Nontrivial relationships)'의 예는?

**Back:** 직접적인 언급이 없더라도 온톨로지 상의 부모-자식 관계를 통해 연결된 정보들입니다.

---

## Card 57

**Front:** NED를 통해 탐지된 UMLS 엔티티로부터 추가 지식을 확장하는 방법은?

**Back:** 해당 엔티티와 연결된 여러 생물 의학 온톨로지들의 문맥적 지식을 탐색하는 것입니다.

---

## Card 58

**Front:** 지식 그래프 스키마에서 $DISAMBIGUATED\_TO$ 관계는 무엇을 연결하나요?

**Back:** 특정 언급($EntityMention$)을 모호성 해소된 결과인 의료 엔티티($MedicalEntity$)에 연결합니다.

---

## Card 59

**Front:** UMLSMetathesaurus에서 'Normalized name'은 어떤 필드에 위치하나요?

**Back:** 일반적으로 엔티티 ID와 온톨로지 정보 옆에 위치하여 해당 엔티티의 표준 명칭을 보여줍니다.

---

## Card 60

**Front:** NED를 활용한 지식 그래프 구축의 마지막 단계는 보통 무엇인가요?

**Back:** 사용자의 유스케이스에 맞는 쿼리(Querying)를 정의하고 실행하는 것입니다.

---

## Card 61

**Front:** ChatGPT가 UMLS 매핑에 대해 답변할 때 강조한 '필요한 전문성'은?

**Back:** 도메인 특정 지식 및 UMLS 자원 활용에 대한 전문 지식입니다.

---

## Card 62

**Front:** 지식 그래프 모델에서 $SnomedEntity$는 어떤 노드의 하위 유형인가요?

**Back:** $MedicalEntity$

---

## Card 63

**Front:** 온톨로지 처리(Ontology processing) 단계에서 주로 수행하는 작업은?

**Back:** 외부 온톨로지 데이터를 로드하고 지식 그래프 내의 엔티티들과 매핑하는 것입니다.

---

## Card 64

**Front:** NED가 제공하는 '불확실성 제거' 기능은 어떤 경우에 가장 유용합니까?

**Back:** 동일한 약어(Abbreviation)나 명칭이 문맥에 따라 전혀 다른 의학적 의미를 가질 때입니다.

---

## Card 65

**Front:** NED 시스템에서 후보 선택 단계가 실패하면 이후 단계에 어떤 영향을 주나요?

**Back:** 올바른 엔티티가 후보군에 포함되지 않으면 순위 지정 단계에서 정답을 찾을 수 없게 됩니다.

---

## Card 66

**Front:** 지식 그래프의 '비구조적 지식과 구조적 지식의 가교' 역할이란?

**Back:** 텍스트(비구조적)에서 추출된 개체를 온톨로지(구조적)의 개념과 연결하여 통합된 뷰를 제공함을 의미합니다.

---

## Card 67

**Front:** UMLS $MRCONSO.RRF$ 파일은 어떤 형식의 파일로 처리될 수 있나요?

**Back:** 구분자로 분리된 값(DSV) 형식이므로 전통적인 CSV 파일처럼 처리 가능합니다.

---

## Card 68

**Front:** 개념 검색이 '설명 가능성(Explainability)'을 향상시키는 방법은?

**Back:** 검색 결과와 함께 해당 엔티티가 나타난 텍스트의 앞뒤 문맥(context snippet)을 제공함으로써 근거를 제시합니다.

---

## Card 69

**Front:** NED 없이 NER만 수행했을 때, 'Zika virus'와 'Zika syndrome'은 어떻게 처리되나요?

**Back:** 두 단어 모두 '질병' 또는 '바이러스'로 분류될 수 있지만, 두 개념 사이의 논리적 차이는 식별되지 않습니다.

---

## Card 70

**Front:** 지식 그래프 구축 프로세스 중 'Ingestion' 단계의 역할은?

**Back:** 처리된 문서와 추출된 엔티티 정보를 실제 그래프 데이터베이스에 적재하는 것입니다.

---

## Card 71

**Front:** UMLS ID $C0011311$의 다른 명칭 예시는?

**Back:** 'Dengue fever', 'Dungero', 'Dandy fever' 등입니다.

---

## Card 72

**Front:** NED 3단계에서 'Enrichment'란 무엇을 의미하나요?

**Back:** 타겟 엔티티에 대해 다른 온톨로지가 가진 속성이나 관계 정보를 추가하여 정보를 더 풍성하게 만드는 것입니다.

---

## Card 73

**Front:** 지식 그래프에서 $HAS\_PHENOTYPIC\_FEATURE$ 관계는 주로 어떤 엔티티들을 연결하나요?

**Back:** 질병 엔티티와 그에 따른 신체적/임상적 증상(표현형) 엔티티를 연결합니다.

---

## Card 74

**Front:** NED의 'Candidate Selection' 단계에서 주로 사용되는 기술적 방법은?

**Back:** 문자열 유사도 매칭이나 사전(Dictionary) 기반의 검색 기법이 사용됩니다.

---

## Card 75

**Front:** 온톨로지 내 'IS\_A' 관계의 핵심 기능은?

**Back:** 개념 간의 계층적 상속 구조(부모-자식 관계)를 정의하여 지식의 추상화를 가능하게 합니다.

---

## Card 76

**Front:** NED 결과 분석 시 'Precision(정밀도)'이 중요한 이유는?

**Back:** 의학 도메인처럼 오류의 비용이 큰 분야에서는 잘못된 엔티티 연결이 위험한 결과를 초래할 수 있기 때문입니다.

---

## Card 77

**Front:** 지식 그래프 검색 쿼리에서 'APOC' 라이브러리의 용도는?

**Back:** 텍스트 결합($apoc.text.join$)과 같은 표준 사이퍼(Cypher) 쿼리 이상의 확장 기능을 제공하기 위함입니다.

---

## Card 78

**Front:** CRISP-DM 모델의 'Data Understanding' 단계에서 SoHO 정책 담당자가 관심을 갖는 데이터는?

**Back:** SoHO 관련 규정, 보고서, 지침 및 의료 온톨로지(SNOMED, HPO)입니다.

---

## Card 79

**Front:** NED 과정에서 '문맥(Context)'이란 구체적으로 무엇을 말하나요?

**Back:** 인식된 엔티티 주변에 위치한 단어들의 집합으로, 의미를 결정하는 힌트가 됩니다.

---

## Card 80

**Front:** 지식 그래프를 활용해 '당뇨병과 관련된 모든 질환'을 찾는 원리는?

**Back:** 당뇨병 노드에서 시작하여 온톨로지 경로를 따라 연결된 모든 질병 노드를 탐색하는 것입니다.

---
